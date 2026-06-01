"""
Ablation script: Single-stream `seq_only` — EntanglementAttention only processes sequence stream.

此脚本会 monkey-patch `models.components.ssdn.EntanglementAttention.forward`，使其
跳过结构相关的 cross-attention/struct 更新，仅保留序列自注意与（可选）门控，
并保持返回签名兼容 `(seq_out, struct_out)`。

使用示例：
  mamba activate /root/miniconda3/envs/copra_h
  python wo_experiment/wo_seq_only.py finetune \
	--model_config ./config/models/prorna_ssdn.yml \
	--data_config ./config/datasets/PRI30k.yml \
	--run_config ./config/runs/pretune_struct.yml \
	--stage pretune
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


def patch_entanglement_seq_only():
	try:
		mod = importlib.import_module('models.components.ssdn')
	except Exception as e:
		raise RuntimeError(f'无法导入 models.components.ssdn: {e}')

	if not hasattr(mod, 'EntanglementAttention'):
		raise RuntimeError('模块 models.components.ssdn 中未找到 EntanglementAttention')

	cls = mod.EntanglementAttention

	def seq_only_forward(self, seq_emb, struct_emb, interaction_weights):
		# Minimal seq-only behavior: compute seq self-attention, apply interaction weight and optional gate.
		B = seq_emb.shape[0]
		device = seq_emb.device
		dtype = seq_emb.dtype

		# seq self-attn
		try:
			seq_self, _ = self.seq_self_attn(seq_emb, seq_emb, seq_emb)
		except Exception:
			# fallback: zero update
			seq_self = torch.zeros_like(seq_emb)

		seq_self = self.seq_norm(seq_emb + seq_self)

		# derive pools for gating computation (keep consistent with original)
		seq_pooled = seq_emb.mean(dim=1)
		try:
			struct_pool_tokens = struct_emb.mean(dim=2)
		except Exception:
			struct_pool_tokens = torch.zeros((B, seq_emb.shape[-1]), device=device, dtype=dtype)

		# interaction weights
		try:
			w_seq, w_struct, w_mix = interaction_weights.unbind(dim=-1)
			w_seq = w_seq.view(B, 1, 1)
		except Exception:
			w_seq = torch.ones((B, 1, 1), device=device, dtype=dtype)

		seq_entangled = w_seq * seq_self

		# gate
		if getattr(self, 'use_gate', False) and hasattr(self, 'seq_gate') and self.seq_gate is not None:
			gate_seq = self.seq_gate(torch.cat([seq_entangled.mean(dim=1), struct_pool_tokens.mean(dim=1)], dim=-1)).unsqueeze(1)
		else:
			gate_seq = torch.ones((B, 1, 1), device=device, dtype=dtype)

		seq_out = seq_emb + gate_seq * seq_entangled

		# struct_out unchanged (skip struct self-attention / cross updates)
		struct_out = struct_emb

		return seq_out, struct_out

	setattr(cls, 'forward', seq_only_forward)
	print('已将 EntanglementAttention.forward monkey-patch 为 seq_only 模式（仅处理序列流）')


class AblationRunner:
	def __init__(self, model_config='./config/models/prorna_ssdn.yml',
				 data_config='./config/datasets/PRI30k.yml',
				 run_config='./config/runs/pretune_struct.yml'):
		self.model_config = model_config
		self.data_config = data_config
		self.run_config = run_config

	def finetune(self, stage='dG', stream_mode='seq_only'):
		"""Run finetune/pretune with EntanglementAttention operating in seq-only mode.

		`stream_mode` kept for API compatibility; when set to 'seq_only' we apply the patch.
		"""
		if stream_mode == 'seq_only':
			patch_entanglement_seq_only()

		runner = LightningRunner(model_config=self.model_config,
								 data_config=self.data_config,
								 run_config=self.run_config)
		runner.finetune(stage=stage)


if __name__ == '__main__':
	fire.Fire(AblationRunner)
