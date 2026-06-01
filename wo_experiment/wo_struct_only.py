"""
Ablation script: Single-stream `struct_only` — EntanglementAttention only processes structure stream.

此脚本会 monkey-patch `models.components.ssdn.EntanglementAttention.forward`，使其
仅执行结构自注意 / 结构更新（pairwise），跳过序列 cross-attention 与序列更新，
并保持返回签名兼容 `(seq_out, struct_out)`。

使用示例：
  mamba activate /root/miniconda3/envs/copra_h
  python wo_experiment/wo_struct_only.py finetune \
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


def patch_entanglement_struct_only():
    try:
        mod = importlib.import_module('models.components.ssdn')
    except Exception as e:
        raise RuntimeError(f'无法导入 models.components.ssdn: {e}')

    if not hasattr(mod, 'EntanglementAttention'):
        raise RuntimeError('模块 models.components.ssdn 中未找到 EntanglementAttention')

    cls = mod.EntanglementAttention

    def struct_only_forward(self, seq_emb, struct_emb, interaction_weights):
        # Minimal struct-only behavior: compute struct self-attention, apply interaction weight and optional gate.
        B = seq_emb.shape[0]
        device = seq_emb.device
        dtype = seq_emb.dtype

        # prepare struct for multihead attention which expects (B*L, L, P)
        try:
            B_s, L, L2, P = struct_emb.shape
            assert B_s == B and L == L2
            struct_reshaped = struct_emb.view(B * L, L, P)
        except Exception:
            # fallback: treat struct_emb as (B, L, P)
            struct_reshaped = struct_emb

        try:
            struct_self, _ = self.struct_self_attn(struct_reshaped, struct_reshaped, struct_reshaped)
        except Exception:
            struct_self = torch.zeros_like(struct_reshaped)

        # reshape back if necessary
        if struct_self.dim() == 3 and struct_self.shape[0] == B * (struct_emb.shape[1] if struct_emb.dim() >= 3 else 1):
            struct_self = struct_self.view(B, struct_emb.shape[1], struct_emb.shape[1], -1)

        # apply normalization and residual
        try:
            struct_out = self.struct_norm(struct_emb + struct_self)
        except Exception:
            struct_out = struct_emb + struct_self

        # optionally apply gating using seq pooled info to keep compatibility
        try:
            struct_pooled = struct_out.mean(dim=(1, 2))
            seq_pooled = seq_emb.mean(dim=1)
            if getattr(self, 'use_gate', False) and hasattr(self, 'struct_gate') and self.struct_gate is not None:
                gate_struct = self.struct_gate(torch.cat([seq_pooled, struct_pooled], dim=-1)).unsqueeze(1).unsqueeze(1)
                struct_out = struct_emb + gate_struct * (struct_out - struct_emb)
        except Exception:
            pass

        # seq_out unchanged
        seq_out = seq_emb

        return seq_out, struct_out

    setattr(cls, 'forward', struct_only_forward)
    print('已将 EntanglementAttention.forward monkey-patch 为 struct_only 模式（仅处理结构流）')


class AblationRunner:
    def __init__(self, model_config='./config/models/prorna_ssdn.yml',
                 data_config='./config/datasets/PRI30k.yml',
                 run_config='./config/runs/pretune_struct.yml'):
        self.model_config = model_config
        self.data_config = data_config
        self.run_config = run_config

    def finetune(self, stage='dG', stream_mode='struct_only'):
        """Run finetune/pretune with EntanglementAttention operating in struct-only mode.

        `stream_mode` kept for API compatibility; when set to 'struct_only' we apply the patch.
        """
        if stream_mode == 'struct_only':
            patch_entanglement_struct_only()

        runner = LightningRunner(model_config=self.model_config,
                                 data_config=self.data_config,
                                 run_config=self.run_config)
        runner.finetune(stage=stage)


if __name__ == '__main__':
    fire.Fire(AblationRunner)
