import json
import os
os.environ["NUMEXPR_MAX_THREADS"] = '56'
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import fire
from pathlib import Path
import pandas as pd

import numpy as np
import yaml
import wandb
import time
from easydict import EasyDict
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies.ddp import DDPStrategy
from pl_modules import ModelModule, DataModule, PretuneModule, DDGModule
from collections import defaultdict

torch.set_num_threads(16)

def parse_yaml(yaml_dir):
    with open(yaml_dir, 'r') as f:
        content = f.read()
        config_dict = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
        # args = Namespace(**config_dict)
    return config_dict
def init_pytorch_settings():
    # Multiprocess Setting to speedup dataloader
    torch.multiprocessing.set_start_method('forkserver')
    torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class LightningRunner(object):
    def __init__(self, model_config='./config/models/esm2_rinalmo.yaml', data_config='./config/datasets/rpi.yaml',
                 run_config='./config/runs/finetune_sequence.yaml'):
        super(LightningRunner, self).__init__()
        self.model_args = parse_yaml(model_config)
        self.dataset_args = parse_yaml(data_config)
        self.run_args = parse_yaml(run_config)
        init_pytorch_settings()

    def save_model(self, model, output_dir, trainer):
        print("Best Model Path:", trainer.checkpoint_callback.best_model_path)
        module = ModelModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        if trainer.global_rank == 0:
            best_model = module.model
            (output_dir / 'model_data.json').write_text(json.dumps(vars(self.dataset_args), indent=2))
            torch.save(best_model, str(output_dir / 'model.pt'))
    
    def select_module(self, stage, log_dir):
        if stage=='pretune':
            model = PretuneModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args, run_args=self.run_args)
        elif stage=='dG':
            model = ModelModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args, run_args=self.run_args)
        elif stage=='ddG':
            model = DDGModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args, run_args=self.run_args)
        else:
            raise NotImplementedError
        return model

    def finetune(self, stage='dG'):
        print("Run args:", self.run_args, "\n")
        print("Model args:", self.model_args, "\n")
        print("Dataset args:", self.dataset_args, "\n")
        output_dir, gpus = (self.run_args.output_dir, self.run_args.gpus)
        self.model_args.model.stage = stage
        # Setup datamodule
        run_results = []
        for k in range(self.run_args.num_folds):
            # if k != 4:
            #     continue
            print(f"Training fold {k} Started!")
            output_dir = Path(output_dir)
            log_dir = output_dir / f'log_fold_{k}'
            data_module = DataModule(dataset_args=self.dataset_args, **self.dataset_args, col_group=f'fold_{k}')

            # Setup model module
            model = self.select_module(stage, log_dir)
            # Trainer setting
            name = self.run_args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
            if self.run_args.wandb:
                wandb.init(project='copra', name=name)
                logger = WandbLogger()
            else:
                logger = CSVLogger(str(log_dir))
            # version_dir = Path(logger_csv.log_dir)
            pl.seed_everything(self.model_args.train.seed)
            print("Successfully initialized, start trainer...")
            strategy=DDPStrategy(find_unused_parameters=True)
            # strategy.lightning_restore_optimizer = False
            # Build callbacks list: early stopping + best-checkpoint-by-val_loss
            callbacks = [
                EarlyStopping(monitor="val_loss", mode="min", patience=self.run_args.patience, strict=False),
                # Keep the best checkpoint by validation loss
                ModelCheckpoint(dirpath=(log_dir / 'checkpoint'), filename='{epoch}-{val_loss:.3f}',
                                monitor="val_loss", mode="min", save_last=False, save_top_k=1),
            ]

            # Optionally add a periodic checkpoint callback (e.g., every N epochs)
            periodic_n = getattr(self.run_args, 'save_every_n_epochs', None)
            try:
                # EasyDict may store as int or string — coerce
                if periodic_n is not None:
                    periodic_n = int(periodic_n)
            except Exception:
                periodic_n = None

            if periodic_n and periodic_n > 0:
                callbacks.append(
                    ModelCheckpoint(dirpath=(log_dir / 'checkpoint'), filename='epoch={epoch}',
                                    every_n_epochs=periodic_n, save_top_k=-1, save_last=False)
                )

            trainer = pl.Trainer(
                devices=gpus,
                # max_steps=self.run_args.iters,
                max_epochs=self.run_args.epochs,
                # optional limits for quick debugging / single-step runs
                limit_train_batches=getattr(self.run_args, 'limit_train_batches', 1.0),
                limit_val_batches=getattr(self.run_args, 'limit_val_batches', 1.0),
                logger=logger,
                num_sanity_val_steps=0,
                callbacks=callbacks,
                # gradient_clip_val=self.model_args.train.max_grad_norm if self.model_args.train.max_grad_norm is not None else None,
                # gradient_clip_algorithm='norm' if self.model_args.train.max_grad_norm is not None else None,
                strategy=strategy,
                    precision=self.run_args.get('precision', 32),
                    log_every_n_steps=3,
            )
            # If a checkpoint path is provided, load its weights manually into
            # the LightningModule with strict=False so we only load matching
            # parameter tensors (do not restore optimizer/scheduler state).
            if hasattr(self.run_args, 'ckpt') and self.run_args.ckpt is not None:
                ckpt_path = self.run_args.ckpt
                try:
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    state_dict = ckpt.get('state_dict', ckpt)
                    lsd_result = model.load_state_dict(state_dict, strict=False)
                    print('Loaded checkpoint weights from %s' % ckpt_path)
                    print('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
                    print('Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))
                except Exception as e:
                    print('Failed to load checkpoint weights manually:', e)

            # Run training normally; do not pass ckpt_path to Trainer.fit so
            # Lightning does not attempt to restore optimizer/scheduler state.
            trainer.fit(model=model, datamodule=data_module)
            print(f"Training fold {k} Finished!")
            trainer.strategy.barrier()
            print("Best Validation Results:")
            _ = trainer.test(model=model, ckpt_path="best", datamodule=data_module)
            res = model.res
            run_results.append(res)
            if trainer.global_rank == 0:
                self.save_model(model, output_dir, trainer)
        result_dir = Path(output_dir) / name
        os.makedirs(result_dir, exist_ok=True)
        with open(result_dir / 'res.json', 'w') as f:
            json.dump(run_results, f)
        results_df = pd.DataFrame(run_results)
        print(results_df.describe())

    def test(self, stage='dG'):
        print("Args:", self.run_args, self.dataset_args, self.model_args)
        output_dir, ckpts, gpus = (self.run_args.output_dir, self.run_args.ckpts,
                                   self.run_args.gpus)
        run_results = []
        for k in range(self.run_args.num_folds):
            output_dir = Path(output_dir)
            log_dir = output_dir / f'log_fold_{k}'
            data_module = DataModule(dataset_args=self.dataset_args, **self.dataset_args, col_group=f'fold_{k}')
            # data_module.setup()
            model = self.select_module(stage, log_dir)
            logger = CSVLogger(str(log_dir))
            strategy=DDPStrategy(find_unused_parameters=True)
            # strategy.lightning_restore_optimizer = False
            trainer = pl.Trainer(
                devices=gpus,
                max_epochs=0,
                logger=[
                    logger,
                ],
                callbacks=[
                    TQDMProgressBar(refresh_rate=1),
                ],
                strategy=strategy,
            )

            _ = trainer.test(model=model, ckpt_path=ckpts[k], datamodule=data_module)
            res = model.res
            run_results.append(res)
        if trainer.global_rank == 0:
            results_df = pd.DataFrame(run_results)
            print(results_df.describe())
            

if __name__ == '__main__':
    fire.Fire(LightningRunner)