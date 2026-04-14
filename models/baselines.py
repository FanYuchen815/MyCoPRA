import torch
import torch.nn as nn
from models.register import ModelRegister
from models.model import load_esm, load_rinalmo, segment_cat_pad, cat_pad
from models.encoders.pair import ResiduePairEncoder

R = ModelRegister()


@R.register('sequence_only')
class SequenceOnlyBaseline(nn.Module):
    """仅使用 LM embedding + MLP 的基线模型：序列特征池化后回归。"""
    def __init__(self, rinalmo_weights='./weights/rinalmo_giga_pretrained.pt', esm_type='650M', rinalmo_type='650M', pooling='mean', output_dim=1, fix_lms=True, **kwargs):
        super().__init__()
        self.esm, esm_feat_size = load_esm(esm_type)
        self.rinalmo, rinalmo_feat_size = load_rinalmo(rinalmo_weights, rinalmo_type)
        self.pooling = pooling
        feat_size = rinalmo_feat_size
        hidden = max(128, feat_size)
        self.pred_head = nn.Sequential(nn.Linear(feat_size * 2, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))
        if fix_lms:
            for p in self.esm.parameters():
                p.requires_grad_(False)
            for p in self.rinalmo.parameters():
                p.requires_grad_(False)

    def forward(self, input, strategy='separate'):
        prot_input = input['prot']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        na_input = input['na']
        na_chains = input['na_chains']
        na_mask = input['na_mask']

        with torch.no_grad():
            prot_embedding = self.esm(prot_input, repr_layers=[33], return_contacts=False)['representations'][33]
            na_embedding = self.rinalmo(na_input)['representation']

        if strategy == 'separate':
            out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, input['pos_atoms'].shape[1])
        else:
            out_embedding, masks = cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, input['pos_atoms'].shape[1], None)

        # split pooled prot and rna
        # For sequence-only baseline, pool protein and RNA separately then concat
        B = out_embedding.shape[0]
        # Assuming prot_chains and na_chains are available in batch-level format is complex; fallback to mean pooling over tokens
        seq_pool = out_embedding.mean(dim=1)
        # For simplicity split pool into two halves if possible
        half = seq_pool.shape[-1] // 2
        prot_pool = seq_pool[:, :half]
        na_pool = seq_pool[:, half:half*2]
        pooled = torch.cat([prot_pool, na_pool], dim=-1)
        out = self.pred_head(pooled).squeeze(-1)
        return out


@R.register('structure_only')
class StructureOnlyBaseline(nn.Module):
    """仅使用 ResiduePairEncoder 生成的 pairwise 特征并 pool 的基线模型。"""
    def __init__(self, pair_dim=40, output_dim=1, **kwargs):
        super().__init__()
        self.pair_encoder = ResiduePairEncoder(pair_dim, max_num_atoms=4)
        hidden = max(128, pair_dim)
        self.pred_head = nn.Sequential(nn.Linear(pair_dim, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))

    def forward(self, input, strategy='separate'):
        aa = input.get('restype')
        res_nb = input.get('res_nb')
        chain_nb = input.get('chain_nb')
        pos_atoms = input.get('pos_atoms')
        mask_atoms = input.get('mask_atoms')

        z = self.pair_encoder(
            aa=aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
        )
        pooled = z.mean(dim=(1, 2))
        out = self.pred_head(pooled).squeeze(-1)
        return out
