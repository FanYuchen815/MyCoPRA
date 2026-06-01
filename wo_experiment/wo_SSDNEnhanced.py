"""
Ablation script: skip SSDNEnhanced fusion (ablate fusion)

将 `models.components.ssdn.SSDNEnhanced.forward` 替换为直接对 `seq_emb` 和 `struct_emb`
进行 pooling -> `fusion`，从而跳过所有交互层（`c_former` 整体行为退化为 pooling）。

使用示例：
  mamba activate /root/miniconda3/envs/copra_h
  python wo_experiment/wo_SSDNEnhanced.py finetune \
	--model_config ./config/models/prorna_ssdn.yml \
	--data_config ./config/datasets/PRI30k.yml \
	--run_config ./config/runs/pretune_struct.yml \
	--stage pretune

消融选项：调用时可传 `ablate_fusion=True`（默认 True 当运行此脚本时）。
"""

import sys
import importlib
import fire
import torch
from pathlib import Path

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from run import LightningRunner


def patch_ssdn_enhanced_pooling():
	"""Monkey-patch SSDNEnhanced.forward to skip layers and only pool+fusion."""
	try:
		mod = importlib.import_module('models.components.ssdn')
	except Exception as e:
		raise RuntimeError(f'无法导入 models.components.ssdn: {e}')

	if not hasattr(mod, 'SSDNEnhanced'):
		raise RuntimeError('模块 models.components.ssdn 中未找到 SSDNEnhanced')

	cls = mod.SSDNEnhanced

	def ablated_forward(self, seq_emb, struct_emb, key_padding_mask=None, need_attn_weights=False, attn_mask=None, atom_coords=None, residue_mask=None, chain_mask=None):
		# Compute interaction weights via ITP if present (keep gating behavior consistent)
		try:
			interaction_weights = self.itp(seq_emb, struct_emb)
		except Exception:
			# fallback to uniform
			B = seq_emb.size(0)
			device = seq_emb.device
			interaction_weights = torch.zeros((B, 3), device=device, dtype=seq_emb.dtype)
			interaction_weights[:, 0] = 1.0

		# Skip all entanglement layers: directly pool seq_emb and struct_emb
		seq_pool = seq_emb.mean(dim=1)      # [B, E]
		struct_pool = struct_emb.mean(dim=(1, 2))  # [B, P]

		# trim to min dim and fuse (reuse existing fusion projection)
		min_dim = min(seq_pool.shape[-1], struct_pool.shape[-1])
		seq_pool = seq_pool[:, :min_dim]
		struct_pool = struct_pool[:, :min_dim]

		# Use module's fusion layer if available, else simple concat+linear
		try:
			fused = self.fusion(torch.cat([seq_pool, struct_pool], dim=-1))
		except Exception:
			import torch.nn as nn
			linear = nn.Sequential(nn.Linear(min_dim * 2, seq_pool.shape[-1]), nn.ReLU(), nn.Linear(seq_pool.shape[-1], seq_pool.shape[-1]))
			# move fallback module to input device to avoid CPU/GPU mismatch
			linear = linear.to(seq_pool.device)
			fused = linear(torch.cat([seq_pool, struct_pool], dim=-1))

		return fused, seq_emb, struct_emb

	setattr(cls, 'forward', ablated_forward)
	print('已将 SSDNEnhanced.forward monkey-patch 为 skip-layers pooling+fusion')


class AblationRunner:
	def __init__(self, model_config='./config/models/prorna_ssdn.yml',
				 data_config='./config/datasets/PRI30k.yml',
				 run_config='./config/runs/pretune_struct.yml'):
		self.model_config = model_config
		self.data_config = data_config
		self.run_config = run_config

	def finetune(self, stage='dG', ablate_fusion=True):
		"""Run finetune/pretune with SSDNEnhanced fusion ablated when `ablate_fusion=True`."""
		if ablate_fusion:
			patch_ssdn_enhanced_pooling()

		runner = LightningRunner(model_config=self.model_config,
								 data_config=self.data_config,
								 run_config=self.run_config)
		runner.finetune(stage=stage)


if __name__ == '__main__':
	fire.Fire(AblationRunner)

