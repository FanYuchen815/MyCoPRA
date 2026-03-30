import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.attention import MultiHeadSelfAttention
from models.components.interaction_adapter import InteractionAdapter


class SSDNBlock(nn.Module):
    """A lightweight SSDN block: sequence self-attention + struct update + cross interaction + gate

    Minimal, fast-to-run prototype intended as an MVP fusion layer.
    """
    def __init__(self, embed_dim, pair_dim, num_heads=4, cross_heads=4, dropout=0.0):
        super().__init__()
        self.seq_attn = MultiHeadSelfAttention(embed_dim, pair_dim, num_heads)
        self.seq_ln = nn.LayerNorm(embed_dim)
        self.struct_ln = nn.LayerNorm(pair_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.ff_ln = nn.LayerNorm(embed_dim)

        # small MLP to refine struct embeddings
        self.struct_mlp = nn.Sequential(nn.Linear(pair_dim, pair_dim), nn.ReLU(), nn.Linear(pair_dim, pair_dim))

        # cross interaction adapter (pooled cross-attention)
        self.cross_adapter_seq = InteractionAdapter(embed_dim, cross_heads, dropout)
        self.cross_adapter_struct = InteractionAdapter(pair_dim, cross_heads, dropout) if pair_dim == embed_dim else None
        # projections to align dims between seq and struct pooled representations
        self.struct_to_seq = nn.Linear(pair_dim, embed_dim)
        self.seq_to_struct = nn.Linear(embed_dim, pair_dim)

        # gating to dynamically balance interaction
        self.gate = nn.Linear(embed_dim + pair_dim, embed_dim)

    def forward(self, x, struct_embed, key_padding_mask=None, attn_mask=None):
        # x: (B, L, E), struct_embed: either (B, L, P) or (B, L, L, P)
        # Ensure struct_embed is pairwise (B, L, L, P) for compatibility with attention
        if struct_embed.dim() == 3:
            # expand token-level pair features to pairwise by outer product-like interaction
            # result shape: (B, L, L, P)
            struct_embed = torch.einsum('bid,bjd->bijd', struct_embed, struct_embed)

        seq_out, struct_out, attn = self.seq_attn(x, struct_embed, attn_mask, key_padding_mask)
        x = x + seq_out
        struct_embed = struct_embed + struct_out

        # FFN
        residual = x
        x = self.seq_ln(x + self.ff(x))

        # struct MLP
        struct_embed = self.struct_ln(struct_embed + self.struct_mlp(struct_embed))

        # pooled cross interaction (lightweight)
        seq_pool = x.mean(dim=1, keepdim=True)  # (B,1,E)
        # struct_embed is pairwise (B, L, L, P): pool over both sequence axes
        struct_pool = struct_embed.mean(dim=(1, 2)).unsqueeze(1)  # (B,1,P)

        # project pooled struct into seq space for cross-attention
        struct_pool_proj = self.struct_to_seq(struct_pool)
        seq_upd = self.cross_adapter_seq(seq_pool, struct_pool_proj)
        if self.cross_adapter_struct is not None:
            # if struct adapter exists and expects pair_dim==embed_dim
            # project seq pooled into struct dim first
            seq_pool_proj = self.seq_to_struct(seq_pool)
            struct_upd = self.cross_adapter_struct(seq_pool_proj, struct_pool)
        else:
            # project seq_pool into pair space if dims differ
            struct_upd = self.seq_to_struct(seq_pool)

        # broadcast pooled updates back
        # normalize updates along pooled dimension if adapters returned multiple pooled tokens
        if seq_upd.dim() > 2:
            seq_upd = seq_upd.mean(dim=1, keepdim=True)
        if struct_upd.dim() > 2:
            struct_upd = struct_upd.mean(dim=1, keepdim=True)

        # broadcast pooled updates back to token / pairwise shapes safely using expand
        L1 = x.shape[1]
        x = x + seq_upd.expand(-1, L1, -1)
        # struct_embed is (B, L, L, P); expand struct_upd to match
        L = struct_embed.shape[1]
        struct_embed = struct_embed + struct_upd.view(struct_upd.shape[0], 1, 1, struct_upd.shape[-1]).expand(-1, L, L, -1)

        # gating: compute gate from pooled representations and apply to sequence channels
        gate = torch.sigmoid(self.gate(torch.cat([seq_pool.squeeze(1), struct_pool.squeeze(1)], dim=-1))).unsqueeze(1)
        x = (1 - gate) * residual + gate * x

        return x, struct_embed, attn


class SSDN(nn.Module):
    def __init__(self, embed_dim, pair_dim, num_layers=2, num_heads=4, cross_heads=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SSDNBlock(embed_dim, pair_dim, num_heads=num_heads, cross_heads=cross_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.pair_final_layer_norm = nn.LayerNorm(pair_dim)

    def forward(self, x, struct_embed, key_padding_mask=None, need_attn_weights=False, attn_mask=None):
        attn_weights = [] if need_attn_weights else None
        for block in self.blocks:
            x, struct_embed, attn = block(x, struct_embed, key_padding_mask, attn_mask)
            if need_attn_weights:
                attn_weights.append(attn)

        x = self.final_layer_norm(x)
        struct_embed = self.pair_final_layer_norm(struct_embed)
        return x, struct_embed, attn_weights


class InteractionPerceptionLayer(nn.Module):
    """
    互作感知层：跨模态注意力 + 界面锚点 + 门控融合
    """
    def __init__(self, embed_dim, num_heads=8, interface_threshold=5.0):
        super().__init__()
        self.cross_attn_protein = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_rna = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.interface_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.interface_threshold = interface_threshold

    def forward(self, protein_emb, rna_emb, distances):
        # protein_emb: [B, Lp, E], rna_emb: [B, Lr, E], distances: [B, Lp, Lr]
        interface_mask = (distances < self.interface_threshold)  # [B, Lp, Lr]

        # key_padding_mask expects [B, S] where True indicates padding - we invert
        prot_mask = ~interface_mask.any(dim=-1)  # [B, Lp]
        rna_mask = ~interface_mask.any(dim=-2)   # [B, Lr]

        protein_to_rna, _ = self.cross_attn_protein(protein_emb, rna_emb, rna_emb, key_padding_mask=rna_mask)
        rna_to_protein, _ = self.cross_attn_rna(rna_emb, protein_emb, protein_emb, key_padding_mask=prot_mask)

        gate_p = self.interface_gate(torch.cat([protein_emb, protein_to_rna], dim=-1))
        gate_r = self.interface_gate(torch.cat([rna_emb, rna_to_protein], dim=-1))

        protein_out = gate_p * protein_to_rna + (1 - gate_p) * protein_emb
        rna_out = gate_r * rna_to_protein + (1 - gate_r) * rna_emb

        return protein_out, rna_out


class InteractionTypePerceptor(nn.Module):
    """轻量互作类型感知器（ITP）"""
    def __init__(self, embed_dim, num_types=3):
        super().__init__()
        self.num_types = num_types
        # project seq and struct pools to a common embed_dim before gating
        self.seq_proj = nn.Linear(embed_dim, embed_dim)
        self.struct_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_types),
            nn.Softmax(dim=-1)
        )

    def forward(self, seq_emb, struct_emb):
        seq_pool = seq_emb.mean(dim=1)      # [B, E]
        struct_pool = struct_emb.mean(dim=(1, 2))  # [B, P]

        # project pools to embed_dim for stable gating
        # if struct_pool has smaller dim, expand/trim as needed via projection
        # ensure tensors are float
        seq_proj = self.seq_proj(seq_pool)
        # if struct_pool has different last dim, pad/trim before projection by viewing
        if struct_pool.shape[-1] != seq_pool.shape[-1]:
            # project struct_pool via a linear that expects seq_pool-sized input
            # if struct_pool smaller, pad with zeros
            if struct_pool.shape[-1] < seq_pool.shape[-1]:
                pad = torch.zeros(struct_pool.shape[0], seq_pool.shape[-1] - struct_pool.shape[-1], device=struct_pool.device, dtype=struct_pool.dtype)
                struct_adj = torch.cat([struct_pool, pad], dim=-1)
            else:
                struct_adj = struct_pool[:, :seq_pool.shape[-1]]
            struct_proj = self.struct_proj(struct_adj)
        else:
            struct_proj = self.struct_proj(struct_pool)

        combined = torch.cat([seq_proj, struct_proj], dim=-1)
        return self.gate_network(combined)


class MultiScaleGeometricEncoder(nn.Module):
    """多尺度几何编码器（原子/残基/链级）"""
    def __init__(self, embed_dim, num_atoms=4):
        super().__init__()
        self.num_atoms = num_atoms
        self.atom_encoder = nn.Sequential(
            nn.Linear(num_atoms * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.residue_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.chain_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, atom_coords, residue_mask, chain_mask):
        # atom_coords: [B, L, num_atoms, 3]
        atom_features = atom_coords.view(atom_coords.size(0), atom_coords.size(1), -1)
        atom_embed = self.atom_encoder(atom_features)  # [B, L, E]

        residue_embed = self.residue_encoder(atom_embed)  # [B, L, E]

        # chain_mask: [B, C, L]
        chain_embed = torch.einsum('bcl,ble->bce', chain_mask.float(), residue_embed)
        denom = chain_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        chain_embed = chain_embed / denom
        chain_embed = self.chain_encoder(chain_embed)  # [B, C, E]

        chain_embed_expanded = torch.einsum('bce,bcl->ble', chain_embed, chain_mask.float())

        multi_scale = torch.cat([atom_embed, residue_embed, chain_embed_expanded], dim=-1)
        return self.fusion(multi_scale)


def _largest_divisor_leq(n, max_div):
    # find the largest divisor of n that is <= max_div
    for d in range(min(max_div, n), 0, -1):
        if n % d == 0:
            return d
    return 1


class EntanglementAttention(nn.Module):
    """缠结注意力层：序列自注意 + 结构自注意 + 交叉注意 + 动态门控"""
    def __init__(self, embed_dim, pair_dim, num_heads=4, cross_heads=4):
        super().__init__()
        seq_heads = _largest_divisor_leq(embed_dim, num_heads)
        struct_heads = _largest_divisor_leq(pair_dim, num_heads)
        cross_heads_seq = _largest_divisor_leq(embed_dim, cross_heads)
        cross_heads_struct = _largest_divisor_leq(pair_dim, cross_heads)

        self.seq_self_attn = nn.MultiheadAttention(embed_dim, seq_heads, batch_first=True)
        self.struct_self_attn = nn.MultiheadAttention(pair_dim, struct_heads, batch_first=True)
        self.cross_seq2struct = nn.MultiheadAttention(embed_dim, cross_heads_seq, batch_first=True)
        self.cross_struct2seq = nn.MultiheadAttention(pair_dim, cross_heads_struct, batch_first=True)

        self.seq_gate = nn.Sequential(nn.Linear(embed_dim + pair_dim, embed_dim), nn.Sigmoid())
        self.struct_gate = nn.Sequential(nn.Linear(pair_dim + embed_dim, pair_dim), nn.Sigmoid())

        self.struct_to_seq = nn.Linear(pair_dim, embed_dim)
        self.seq_to_struct = nn.Linear(embed_dim, pair_dim)

        self.seq_norm = nn.LayerNorm(embed_dim)
        self.struct_norm = nn.LayerNorm(pair_dim)

    def forward(self, seq_emb, struct_emb, interaction_weights):
        B, L, E = seq_emb.shape
        P = struct_emb.shape[-1]

        seq_self, _ = self.seq_self_attn(seq_emb, seq_emb, seq_emb)
        seq_self = self.seq_norm(seq_emb + seq_self)

        struct_reshaped = struct_emb.view(B * L, L, P)
        struct_self, _ = self.struct_self_attn(struct_reshaped, struct_reshaped, struct_reshaped)
        struct_self = struct_self.view(B, L, L, P)
        struct_self = self.struct_norm(struct_emb + struct_self)

        struct_pooled = struct_emb.mean(dim=2)  # [B, L, P]
        struct_proj = self.struct_to_seq(struct_pooled)  # [B, L, E]
        seq_cross, _ = self.cross_seq2struct(seq_self, struct_proj, struct_proj)

        seq_pooled = seq_emb.mean(dim=1, keepdim=True)  # [B,1,E]
        seq_proj = self.seq_to_struct(seq_pooled)  # [B,1,P]
        # Expand seq_proj to shape [B, L, P], then reshape to [B*L, 1, P]
        # and finally expand to [B*L, L, P] to match struct_reshaped's batch dimension
        seq_proj_expanded = seq_proj.expand(-1, L, -1).reshape(B * L, 1, P).expand(-1, L, -1)
        struct_cross, _ = self.cross_struct2seq(struct_reshaped, seq_proj_expanded, seq_proj_expanded)
        struct_cross = struct_cross.view(B, L, L, P)

        w_seq, w_struct, w_mix = interaction_weights.unbind(dim=-1)

        seq_entangled = w_seq.view(B, 1, 1) * seq_self + w_mix.view(B, 1, 1) * seq_cross
        gate_seq = self.seq_gate(torch.cat([seq_entangled.mean(dim=1), struct_pooled.mean(dim=1)], dim=-1)).unsqueeze(1)
        seq_out = seq_emb + gate_seq * seq_entangled

        struct_entangled = w_struct.view(B, 1, 1, 1) * struct_self + w_mix.view(B, 1, 1, 1) * struct_cross
        gate_struct = self.struct_gate(torch.cat([seq_pooled.squeeze(1), struct_entangled.mean(dim=(1, 2))], dim=-1)).unsqueeze(1).unsqueeze(1)
        struct_out = struct_emb + gate_struct * struct_entangled

        return seq_out, struct_out


class SSDNEnhanced(nn.Module):
    """增强版 SSDN：ITP + EntanglementAttention + 多尺度几何融合"""
    def __init__(self, embed_dim, pair_dim, num_layers=8, num_heads=4, cross_heads_start=4, dropout=0.1):
        super().__init__()
        self.itp = InteractionTypePerceptor(embed_dim)
        self.geo_encoder = MultiScaleGeometricEncoder(embed_dim)

        self.layers = nn.ModuleList([
            EntanglementAttention(embed_dim, pair_dim, num_heads=num_heads, cross_heads=min(cross_heads_start + i, 16))
            for i in range(num_layers)
        ])

        self.seq_ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))
        self.struct_ffn = nn.Sequential(nn.Linear(pair_dim, pair_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(pair_dim * 4, pair_dim))

        self.seq_norm = nn.LayerNorm(embed_dim)
        self.struct_norm = nn.LayerNorm(pair_dim)

        # fusion projects concatenated (seq_pool, struct_pool) into embed_dim
        # use 2 * min(embed_dim, pair_dim) because forward trims each pool to the smaller dim
        min_pool_dim = min(embed_dim, pair_dim)
        self.fusion = nn.Sequential(nn.Linear(min_pool_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, seq_emb, struct_emb, key_padding_mask=None, need_attn_weights=False, attn_mask=None, atom_coords=None, residue_mask=None, chain_mask=None):
        # Accepts legacy SSDN call signature: forward(x, struct, key_padding_mask=..., need_attn_weights=...)
        # If geometric inputs are provided, encode and fuse; otherwise skip geometric encoder.
        if atom_coords is not None and residue_mask is not None and chain_mask is not None:
            try:
                geo_embed = self.geo_encoder(atom_coords, residue_mask, chain_mask)
                seq_emb = seq_emb + geo_embed
            except Exception:
                # if geometric encoder fails or inputs incompatible, skip gracefully
                pass

        interaction_weights = self.itp(seq_emb, struct_emb)

        for layer in self.layers:
            seq_emb, struct_emb = layer(seq_emb, struct_emb, interaction_weights)
            seq_emb = self.seq_norm(seq_emb + self.seq_ffn(seq_emb))
            struct_emb = self.struct_norm(struct_emb + self.struct_ffn(struct_emb))

        seq_pool = seq_emb.mean(dim=1)      # [B, E]
        struct_pool = struct_emb.mean(dim=(1, 2))  # [B, P]

        # ensure pooling dims match by trimming to the smaller of the two
        min_dim = min(seq_pool.shape[-1], struct_pool.shape[-1])
        seq_pool = seq_pool[:, :min_dim]
        struct_pool = struct_pool[:, :min_dim]

        fused = self.fusion(torch.cat([seq_pool, struct_pool], dim=-1))

        # Maintain legacy return shape: (fused [B,E], seq_emb [B,L,E], struct_emb [B,L,L,P])
        return fused, seq_emb, struct_emb


class TaskAdaptiveDecoder(nn.Module):
    """任务自适应解码器：向量化任务提示词 + 自适应聚合"""
    def __init__(self, embed_dim, task_emb_dim=64):
        super().__init__()
        self.task_embeddings = nn.ParameterDict({
            'delta_g': nn.Parameter(torch.randn(task_emb_dim)),
            'delta_delta_g': nn.Parameter(torch.randn(task_emb_dim)),
            'binding_site': nn.Parameter(torch.randn(task_emb_dim))
        })

        self.aggregator = nn.Sequential(nn.Linear(embed_dim + task_emb_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

        self.heads = nn.ModuleDict({
            'delta_g': nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1)),
            'delta_delta_g': nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1)),
            'binding_site': nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 2))
        })

    def forward(self, fused_embedding, task='delta_g', seq_embedding=None):
        task_emb = self.task_embeddings[task]
        task_emb = task_emb.unsqueeze(0).expand(fused_embedding.size(0), -1)

        combined = torch.cat([fused_embedding, task_emb], dim=-1)
        adapted = self.aggregator(combined)

        if task == 'binding_site' and seq_embedding is not None:
            B, L, E = seq_embedding.shape
            seq_adapted = adapted.unsqueeze(1).expand(-1, L, -1)
            seq_combined = seq_embedding + seq_adapted
            return self.heads[task](seq_combined)

        return self.heads[task](adapted)
