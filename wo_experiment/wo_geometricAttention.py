"""
Ablation script: disable geometric attention by forcing its output to zero.

使用方法示例：
  mamba activate /root/miniconda3/envs/copra_h
  python wo_experiment/wo_geometricAttention.py finetune --model_config ./config/models/prorna_ssdn.yml \
	  --data_config ./config/datasets/PRI30k.yml --run_config ./config/runs/pretune_struct.yml --stage pretune
  python wo_experiment/wo_geometricAttention.py finetune --model_config ./config/models/prorna_ssdn.yml \
	  --data_config ./config/datasets/finetune.yml --run_config /root/autodl-tmp/CoPRA/config/runs/finetune_from_pretrained.yml --stage dG

此脚本参照 `run.py` 的调用方式，先对 `models.encoders.geometric_attention.GeometricAttention`
的 `forward` 做 monkey-patch（令输出恒为 0），然后调用 `run.LightningRunner.finetune` 运行训练。
"""

import sys
import torch
import fire
import importlib
from pathlib import Path

# Ensure project root is on sys.path so `run.py` can be imported when this
# script is executed from the `wo_experiment` directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from run import LightningRunner


def patch_geometric_attention_zero():
	"""Monkey-patch GeometricAttention.forward -> all-zeros output."""
	try:
		mod = importlib.import_module('models.encoders.geometric_attention')
	except Exception as e:
		raise RuntimeError(f'无法导入 models.encoders.geometric_attention: {e}')

	def zero_forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms):
		# 尝试从 pos_atoms 或 aa 推断 batch/length
		device = None
		dtype = torch.float32
		try:
			if pos_atoms is not None:
				N, L, A, _ = pos_atoms.shape
				device = pos_atoms.device
			elif aa is not None:
				N, L = aa.shape
				device = aa.device
			else:
				# fallback
				return torch.tensor([], device='cpu')
		except Exception:
			return torch.tensor([], device='cpu')

		out_dim = getattr(self, 'out_dim', None)
		if out_dim is None:
			out_dim = getattr(self, 'geo_dim', 0)

		return torch.zeros((N, L, L, out_dim), device=device, dtype=dtype)

	setattr(mod.GeometricAttention, 'forward', zero_forward)
	print('已将 GeometricAttention.forward monkey-patch 为全零输出')


class AblationRunner:
	def __init__(self, model_config='./config/models/prorna_ssdn.yml',
				 data_config='./config/datasets/PRI30k.yml',
				 run_config='./config/runs/pretune_struct.yml'):
		self.model_config = model_config
		self.data_config = data_config
		self.run_config = run_config

	def finetune(self, stage='dG', zero_geo=True):
		"""Run finetune/pretune with geometric-attention disabled when `zero_geo=True`.

		参数 `stage` 与 `run.py` 保持一致（例：'pretune', 'dG' 等）。
		"""
		if zero_geo:
			patch_geometric_attention_zero()

		runner = LightningRunner(model_config=self.model_config,
								 data_config=self.data_config,
								 run_config=self.run_config)
		runner.finetune(stage=stage)


if __name__ == '__main__':
	fire.Fire(AblationRunner)

