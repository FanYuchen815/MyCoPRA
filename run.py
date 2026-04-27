"""
运行入口文档
"""

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
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
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
                wandb.init(project='PRORNA_SSDN', name=name)
                logger = WandbLogger()
            else:
                csv_logger = CSVLogger(str(log_dir))
                tb_logger = TensorBoardLogger(str(log_dir), name='tensorboard')
                logger = [csv_logger, tb_logger]
            # version_dir = Path(logger_csv.log_dir)
            pl.seed_everything(self.model_args.train.seed)
            print("Successfully initialized, start trainer...")
            strategy=DDPStrategy(find_unused_parameters=True)
            # strategy.lightning_restore_optimizer = False
            # Build callbacks list: early stopping + best-checkpoint.
            # Choose the checkpoint monitor key depending on the training stage
            if stage == 'pretune':
                monitor_key = "val_loss"
                mode = "min"
                filename = '{epoch}-{val_loss:.3f}'
            elif stage == 'ddG':
                # DDGModule logs per-complex Pearson as 'val/pc_pearson'
                monitor_key = "val/pc_pearson"
                mode = "max"
                filename = '{epoch}-{val_pc_pearson:.3f}'
            elif stage == 'dG':
                # ModelModule logs overall Pearson as 'val/all_pearson'
                monitor_key = "val/all_pearson"
                mode = "max"
                filename = '{epoch}-{val_all_pearson:.3f}'
            else:
                monitor_key = "val_loss"
                mode = "min"
                filename = '{epoch}-{val_loss:.3f}'

            checkpoint_cb = ModelCheckpoint(dirpath=(log_dir / 'checkpoint'), filename=filename,
                                            monitor=monitor_key, mode=mode, save_last=False, save_top_k=1)

            callbacks = [
                EarlyStopping(monitor="val_loss", mode="min", patience=self.run_args.patience, strict=False),
                checkpoint_cb,
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
            # If a checkpoint path is provided, use Lightning's native resume
            # to restore model weights, optimizer state, scheduler state, and epoch.
            ckpt_path = None
            if hasattr(self.run_args, 'ckpt') and self.run_args.ckpt is not None:
                ckpt_path = self.run_args.ckpt
                print(f'Resuming from checkpoint: {ckpt_path}')

            # Run training with optional checkpoint resume
            trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)
            print(f"Training fold {k} Finished!")
            trainer.strategy.barrier()
            print("Evaluating all checkpoints to select best by PCC with SCC>=baseline (best_ckpt)")

            # determine baseline spearman from the original best checkpoint if available
            # selection threshold will be baseline_spearman - 0.03 (allow up to 0.03 drop)
            baseline_spearman = 0.03
            try:
                best_ckpt_path = trainer.checkpoint_callback.best_model_path
                if best_ckpt_path is not None and best_ckpt_path != "":
                    print(f"Testing original best checkpoint to obtain baseline SCC: {best_ckpt_path}")
                    _ = trainer.test(model=model, ckpt_path="best", datamodule=data_module)
                    res_best = getattr(model, 'res', None)
                    if res_best is not None and 'spearman' in res_best:
                        baseline_spearman = float(res_best.get('spearman', baseline_spearman))
                        print(f"Baseline spearman from best checkpoint: {baseline_spearman:.4f}")
            except Exception as e:
                print(f"Could not obtain baseline spearman from best checkpoint, fallback to {baseline_spearman}: {e}")

            # compute selection threshold: allow up to 0.03 drop from baseline
            threshold_spearman = max(0.0, baseline_spearman - 0.03)
            print(f"Using spearman threshold = baseline - 0.03 = {threshold_spearman:.4f}")

            ckpt_dir = log_dir / 'checkpoint'
            ckpt_files = []
            if ckpt_dir.exists():
                for p in ckpt_dir.glob('*.ckpt'):
                    ckpt_files.append(p)

            ckpt_metrics = []
            # If no checkpoints found, fallback to testing the best
            if len(ckpt_files) == 0:
                print("No checkpoint files found, testing best checkpoint...")
                _ = trainer.test(model=model, ckpt_path="best", datamodule=data_module)
                res = model.res
                run_results.append(res)
                chosen_ckpt = trainer.checkpoint_callback.best_model_path
            else:
                for ck in sorted(ckpt_files):
                    print(f"Testing checkpoint: {ck}")
                    try:
                        _ = trainer.test(model=model, ckpt_path=str(ck), datamodule=data_module)
                        res_ck = getattr(model, 'res', None)
                        if res_ck is None:
                            continue
                        pear = float(res_ck.get('pearson', 0.0))
                        spe = float(res_ck.get('spearman', 0.0))
                        ckpt_metrics.append({'ckpt': str(ck), 'pearson': pear, 'spearman': spe})
                    except Exception as e:
                        print(f"Failed to test checkpoint {ck}: {e}")

                # select ckpt with spearman >= threshold_spearman (baseline - 0.03) and highest pearson
                candidates = [c for c in ckpt_metrics if c['spearman'] >= threshold_spearman]
                if len(candidates) > 0:
                    candidates.sort(key=lambda x: x['pearson'], reverse=True)
                    chosen_ckpt = candidates[0]['ckpt']
                    chosen_metrics = candidates[0]
                    print(f"Selected checkpoint {chosen_ckpt} with pearson={chosen_metrics['pearson']:.4f}, spearman={chosen_metrics['spearman']:.4f} (threshold {threshold_spearman:.4f})")
                    # load chosen ckpt and run final test to populate res
                    _ = trainer.test(model=model, ckpt_path=str(chosen_ckpt), datamodule=data_module)
                    res = model.res
                    run_results.append(res)
                else:
                    # fallback: choose checkpoint with highest pearson regardless of spearman
                    if len(ckpt_metrics) > 0:
                        ckpt_metrics.sort(key=lambda x: x['pearson'], reverse=True)
                        chosen_ckpt = ckpt_metrics[0]['ckpt']
                        print(f"No checkpoint met spearman>=threshold ({threshold_spearman:.4f}); fallback to highest PCC checkpoint {chosen_ckpt}")
                        _ = trainer.test(model=model, ckpt_path=str(chosen_ckpt), datamodule=data_module)
                        res = model.res
                        run_results.append(res)
                    else:
                        print("No valid checkpoint metrics found; testing best checkpoint...")
                        _ = trainer.test(model=model, ckpt_path="best", datamodule=data_module)
                        res = model.res
                        run_results.append(res)

            # save the chosen checkpoint's model
            if trainer.global_rank == 0:
                try:
                    # load module from chosen checkpoint and save
                    print(f"Saving chosen checkpoint model: {chosen_ckpt}")
                    module = ModelModule.load_from_checkpoint(chosen_ckpt)
                    best_model = module.model
                    (output_dir / 'model_data.json').write_text(json.dumps(vars(self.dataset_args), indent=2))
                    # save standard model file
                    torch.save(best_model, str(output_dir / 'model.pt'))
                    # derive a PCC string for filenames
                    p_val = metrics_to_save.get('pearson', None) if isinstance(metrics_to_save, dict) else None
                    try:
                        p_str = f"PCC={float(p_val):.4f}" if p_val is not None else "PCC=NA"
                    except Exception:
                        p_str = "PCC=NA"
                    # also save model with PCC in filename for quick reference
                    try:
                        torch.save(best_model, str(output_dir / f"model_{p_str}.pt"))
                    except Exception as e:
                        print(f"Failed to save model with PCC in filename: {e}")
                    # record chosen checkpoint metrics (pearson / spearman)
                    metrics_to_save = None
                    if 'chosen_metrics' in locals() and isinstance(chosen_metrics, dict):
                        metrics_to_save = {'pearson': float(chosen_metrics.get('pearson', 0.0)), 'spearman': float(chosen_metrics.get('spearman', 0.0)), 'ckpt': str(chosen_ckpt)}
                    else:
                        # try to use res if available
                        try:
                            res_for_metrics = res if 'res' in locals() else getattr(module, 'res', None)
                            if res_for_metrics is None:
                                res_for_metrics = getattr(model, 'res', None)
                            metrics_to_save = {'pearson': float(res_for_metrics.get('pearson', 0.0)) if res_for_metrics is not None else 0.0,
                                               'spearman': float(res_for_metrics.get('spearman', 0.0)) if res_for_metrics is not None else 0.0,
                                               'ckpt': str(chosen_ckpt)}
                        except Exception:
                            metrics_to_save = {'pearson': None, 'spearman': None, 'ckpt': str(chosen_ckpt)}

                    try:
                        # save with and without PCC in filename
                        (output_dir / 'chosen_metrics.json').write_text(json.dumps(metrics_to_save, indent=2))
                        chosen_name = output_dir / f"chosen_metrics_{p_str}.json"
                        (chosen_name).write_text(json.dumps(metrics_to_save, indent=2))
                        print(f"Saved chosen metrics to: {output_dir / 'chosen_metrics.json'} and {chosen_name}")
                    except Exception as e:
                        print(f"Failed to write chosen metrics: {e}")
                except Exception as e:
                    print(f"Failed to save chosen checkpoint model: {e}")
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
            csv_logger = CSVLogger(str(log_dir))
            tb_logger = TensorBoardLogger(str(log_dir), name='tensorboard')
            logger = [csv_logger, tb_logger]
            strategy=DDPStrategy(find_unused_parameters=True)
            # strategy.lightning_restore_optimizer = False
            trainer = pl.Trainer(
                devices=gpus,
                max_epochs=0,
                logger=logger,
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