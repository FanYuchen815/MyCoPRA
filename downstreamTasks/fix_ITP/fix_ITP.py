"""固定 ITP 权重的训练/验证 pipeline。

参考 run.py 中的训练流程；不同点：将 SSDN 的 ITP 固定为相等权重（w_seq=w_struct=w_mix=1/3）。

用法示例：
  python downstreamTasks/fix_ITP/fix_ITP.py finetune --model_config config/models/prorna_ssdn.yml \
		--data_config config/datasets/PRA310.yml --run_config config/runs/finetune_struct.yml
"""
import os
import time
import json
from pathlib import Path
import yaml
from easydict import EasyDict
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
import wandb

# ensure repository root is on sys.path so local packages (pl_modules, models, data, etc.) can be imported
import sys
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

from pl_modules import ModelModule, DataModule


def parse_yaml(yaml_dir):
	with open(yaml_dir, 'r') as f:
		content = f.read()
		return EasyDict(yaml.load(content, Loader=yaml.FullLoader))


def init_pytorch_settings():
	try:
		torch.multiprocessing.set_start_method('forkserver')
		torch.multiprocessing.set_sharing_strategy('file_system')
	except Exception:
		pass
	torch.set_num_threads(4)
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True


class FixITPRunner(object):
	def __init__(self, model_config='./config/models/prorna_ssdn.yml', data_config='./config/datasets/finetune.yml', run_config='./config/runs/finetune_sequence.yaml'):
		self.model_args = parse_yaml(model_config)
		self.dataset_args = parse_yaml(data_config)
		self.run_args = parse_yaml(run_config)
		init_pytorch_settings()

	def _force_fixed_itp(self, equal_weight=True):
		# Ensure fusion config exists
		try:
			fusion = self.model_args.model.fusion
		except Exception:
			if not hasattr(self.model_args.model, 'fusion'):
				self.model_args.model.fusion = EasyDict()
			fusion = self.model_args.model.fusion

		# disable learnable ITP and set fixed weights
		fusion.use_itp = False
		if equal_weight:
			weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
		else:
			weights = fusion.get('itp_weights', [0.5, 0.3, 0.2])
		fusion.itp_weights = weights
		# write back
		self.model_args.model.fusion = fusion

	def finetune(self, stage='dG'):
		print("Run args:", self.run_args)
		print("Model args:", self.model_args)
		print("Dataset args:", self.dataset_args)

		output_dir, gpus = (self.run_args.output_dir, self.run_args.gpus)
		self.model_args.model.stage = stage

		# enforce fixed ITP with equal weights
		self._force_fixed_itp(equal_weight=True)

		run_results = []
		for k in range(self.run_args.num_folds):
			print(f"Training fold {k} Started!")
			output_dir = Path(output_dir)
			log_dir = output_dir / f'log_fold_{k}'
			data_module = DataModule(dataset_args=self.dataset_args, **self.dataset_args, col_group=f'fold_{k}')

			# Setup model module
			model = ModelModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args, run_args=self.run_args)

			# Logger
			name = self.run_args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
			if self.run_args.get('wandb', False):
				wandb.init(project='PRORNA_SSDN', name=name)
				logger = WandbLogger()
			else:
				csv_logger = CSVLogger(str(log_dir))
				tb_logger = TensorBoardLogger(str(log_dir), name='tensorboard')
				logger = [csv_logger, tb_logger]

			pl.seed_everything(self.model_args.train.seed)
			print("Successfully initialized, start trainer...")
			strategy = DDPStrategy(find_unused_parameters=True)

			# monitor settings (reuse run.py defaults)
			monitor_key = "val/all_pearson"
			mode = "max"
			filename = '{epoch}-{val_all_pearson:.3f}'

			checkpoint_cb = ModelCheckpoint(dirpath=(log_dir / 'checkpoint'), filename=filename,
											monitor=monitor_key, mode=mode, save_last=False, save_top_k=1)

			callbacks = [
				EarlyStopping(monitor="val_loss", mode="min", patience=self.run_args.patience, strict=False),
				checkpoint_cb,
			]

			periodic_n = getattr(self.run_args, 'save_every_n_epochs', None)
			try:
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
				max_epochs=self.run_args.epochs,
				limit_train_batches=getattr(self.run_args, 'limit_train_batches', 1.0),
				limit_val_batches=getattr(self.run_args, 'limit_val_batches', 1.0),
				logger=logger,
				num_sanity_val_steps=0,
				callbacks=callbacks,
				strategy=strategy,
				precision=self.run_args.get('precision', 32),
				log_every_n_steps=3,
			)

			ckpt_path = None
			if hasattr(self.run_args, 'ckpt') and self.run_args.ckpt is not None:
				ckpt_path = self.run_args.ckpt
				print(f'Resuming from checkpoint: {ckpt_path}')

			trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)
			print(f"Training fold {k} Finished!")
			trainer.strategy.barrier()

			# test best checkpoint and collect metrics
			try:
				_ = trainer.test(model=model, ckpt_path="best", datamodule=data_module)
				res = getattr(model, 'res', None)
				run_results.append(res)
			except Exception as e:
				print(f"Testing best checkpoint failed: {e}")

		result_dir = Path(output_dir) / name
		os.makedirs(result_dir, exist_ok=True)
		with open(result_dir / 'res.json', 'w') as f:
			json.dump(run_results, f)
		print("Finished all folds.")


if __name__ == '__main__':
	import fire
	fire.Fire(FixITPRunner)

