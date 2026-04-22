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

        # per-token cross interaction (token <-> pairwise for each position)
        # compute per-token struct pool: for each token i, pool pair features across the other axis
        # struct_pool_tokens: (B, L, P)
        struct_pool_tokens = struct_embed.mean(dim=2)

        # project per-token struct into seq space and run cross-attention per token
        struct_pool_proj_tokens = self.struct_to_seq(struct_pool_tokens)  # (B, L, E)
        kv_mask = (~key_padding_mask) if key_padding_mask is not None else None
        try:
            seq_upd = self.cross_adapter_seq(x, struct_pool_proj_tokens, query_mask=None, kv_mask=kv_mask)
        except Exception:
            # fallback to pooled behavior if adapter fails
            seq_pool = x.mean(dim=1, keepdim=True)
            struct_pool = struct_embed.mean(dim=(1, 2)).unsqueeze(1)
            struct_pool_proj = self.struct_to_seq(struct_pool)
            seq_upd = self.cross_adapter_seq(seq_pool, struct_pool_proj)

        # For struct updates, project token-level seq into pair dim and expand to pairwise
        seq_proj_to_pair = self.seq_to_struct(x)  # (B, L, P)
        # build pairwise update by broadcasting token projections along one axis
        L = struct_embed.shape[1]
        struct_upd = seq_proj_to_pair.unsqueeze(2).expand(-1, L, L, -1)

        # apply sequence update and struct update
        x = x + seq_upd
        struct_embed = struct_embed + struct_upd

        # gating: compute gate from pooled representations (use token-wise pools)
        seq_pool_scalar = x.mean(dim=1)
        struct_pool_scalar = struct_pool_tokens.mean(dim=1)
        gate = torch.sigmoid(self.gate(torch.cat([seq_pool_scalar, struct_pool_scalar], dim=-1))).unsqueeze(1)
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
        # Robust pooling: collapse all non-batch and non-last dims, then average.
        # This supports inputs like [B, L, E], [B, L, L, P], [B, X, L, L, P], etc.
        B = seq_emb.size(0)
        seq_last = seq_emb.size(-1)
        seq_pool = seq_emb.reshape(B, -1, seq_last).mean(dim=1)  # [B, E]

        struct_last = struct_emb.size(-1)
        struct_pool = struct_emb.reshape(struct_emb.size(0), -1, struct_last).mean(dim=1)  # [B, P]

        '''
        # optional debug output when DEBUG_SSDN_ITP=1
        import os
        if os.environ.get('DEBUG_SSDN_ITP', '0') == '1':
            try:
                print('DEBUG_SSDN_ITP: seq_emb.shape=', seq_emb.shape, 'struct_emb.shape=', struct_emb.shape)
                print('DEBUG_SSDN_ITP: seq_pool.shape=', seq_pool.shape, 'struct_pool.shape=', struct_pool.shape)
            except Exception:
                pass
        '''

        # project pools to embed_dim for stable gating
        # if struct_pool has smaller dim, expand/trim as needed via projection
        # ensure tensors are float
        seq_proj = self.seq_proj(seq_pool)
        # if struct_pool has different last dim, pad/trim before projection
        if struct_pool.shape[-1] != seq_pool.shape[-1]:
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


class FixedITP(nn.Module):
    """固定权重的 ITP，用于消融（返回固定归一化权重向量）。"""
    def __init__(self, weights):
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32)
        if w.dim() == 1:
            if w.sum().item() != 0:
                w = w / w.sum()
        else:
            w = w.view(-1)
            if w.sum().item() != 0:
                w = w / w.sum()
        self.register_buffer('w', w)

    def forward(self, seq_emb, struct_emb):
        B = seq_emb.size(0)
        return self.w.unsqueeze(0).expand(B, -1)


# MultiScaleGeometricEncoder removed: geometric fusion was implemented but
# not passed from the higher-level model by default. To avoid unused
# parameters and potential performance overhead, the encoder has been deleted.


def _largest_divisor_leq(n, max_div):
    # find the largest divisor of n that is <= max_div
    for d in range(min(max_div, n), 0, -1):
        if n % d == 0:
            return d
    return 1


class EntanglementAttention(nn.Module):
    """缠结注意力层：序列自注意 + 结构自注意 + 交叉注意 + 动态门控

    支持可选的 cross-attention 与门控开关，以及可配置的跨流方向。
    """
    def __init__(self, embed_dim, pair_dim, num_heads=4, cross_heads=4, use_cross=True, use_gate=True, cross_direction='both'):
        super().__init__()
        self.use_cross = use_cross
        self.use_gate = use_gate
        self.cross_direction = cross_direction

        seq_heads = _largest_divisor_leq(embed_dim, num_heads)
        struct_heads = _largest_divisor_leq(pair_dim, num_heads)
        cross_heads_seq = _largest_divisor_leq(embed_dim, cross_heads)
        cross_heads_struct = _largest_divisor_leq(pair_dim, cross_heads)

        self.seq_self_attn = nn.MultiheadAttention(embed_dim, seq_heads, batch_first=True)
        self.struct_self_attn = nn.MultiheadAttention(pair_dim, struct_heads, batch_first=True)

        # only construct cross attention modules if enabled
        if self.use_cross:
            self.cross_seq2struct = nn.MultiheadAttention(embed_dim, cross_heads_seq, batch_first=True)
            self.cross_struct2seq = nn.MultiheadAttention(pair_dim, cross_heads_struct, batch_first=True)
        else:
            self.cross_seq2struct = None
            self.cross_struct2seq = None

        # gating networks (optional)
        if self.use_gate:
            self.seq_gate = nn.Sequential(nn.Linear(embed_dim + pair_dim, embed_dim), nn.Sigmoid())
            self.struct_gate = nn.Sequential(nn.Linear(pair_dim + embed_dim, pair_dim), nn.Sigmoid())
        else:
            self.seq_gate = None
            self.struct_gate = None

        self.struct_to_seq = nn.Linear(pair_dim, embed_dim)
        self.seq_to_struct = nn.Linear(embed_dim, pair_dim)

        self.seq_norm = nn.LayerNorm(embed_dim)
        self.struct_norm = nn.LayerNorm(pair_dim)

    def forward(self, seq_emb, struct_emb, interaction_weights):
        B, L, E = seq_emb.shape
        P = struct_emb.shape[-1]

        # If struct_emb contains extra leading dimensions (e.g., [B, X, L, L, P]),
        # collapse those pre-dimensions by averaging to obtain [B, L, L, P].
        if struct_emb.dim() > 4:
            # expect last three dims to be [L, L, P]
            if struct_emb.shape[-3] == L and struct_emb.shape[-2] == L:
                pre_shape = struct_emb.shape[1:-3]
                if len(pre_shape) > 0:
                    struct_emb = struct_emb.reshape(B, -1, struct_emb.shape[-3], struct_emb.shape[-2], P).mean(dim=1)
            else:
                # fallback: flatten all leading dims (except batch) and try to average
                struct_emb = struct_emb.reshape(B, -1, struct_emb.shape[-1]).mean(dim=1).reshape(B, L, L, P)

        seq_self, _ = self.seq_self_attn(seq_emb, seq_emb, seq_emb)
        seq_self = self.seq_norm(seq_emb + seq_self)

        struct_reshaped = struct_emb.view(B * L, L, P)
        struct_self, _ = self.struct_self_attn(struct_reshaped, struct_reshaped, struct_reshaped)
        struct_self = struct_self.view(B, L, L, P)
        struct_self = self.struct_norm(struct_emb + struct_self)

        struct_pooled = struct_emb.mean(dim=2)  # [B, L, P]
        struct_proj = self.struct_to_seq(struct_pooled)  # [B, L, E]

        # compute seq_cross if allowed and configured
        if self.use_cross and (self.cross_direction in ('both', 'seq2struct')) and (self.cross_seq2struct is not None):
            seq_cross, _ = self.cross_seq2struct(seq_self, struct_proj, struct_proj)
        else:
            seq_cross = torch.zeros_like(seq_self)

        # build token-wise projected seq for struct cross-attention
        seq_proj_per_token = self.seq_to_struct(seq_self)  # [B, L, P]
        seq_proj_expanded = seq_proj_per_token.reshape(B * L, 1, P).expand(-1, L, -1)

        if self.use_cross and (self.cross_direction in ('both', 'struct2seq')) and (self.cross_struct2seq is not None):
            struct_cross, _ = self.cross_struct2seq(struct_reshaped, seq_proj_expanded, seq_proj_expanded)
            struct_cross = struct_cross.view(B, L, L, P)
        else:
            struct_cross = torch.zeros_like(struct_self)

        seq_pooled = seq_emb.mean(dim=1, keepdim=True)  # [B,1,E]

        w_seq, w_struct, w_mix = interaction_weights.unbind(dim=-1)

        seq_entangled = w_seq.view(B, 1, 1) * seq_self + w_mix.view(B, 1, 1) * seq_cross
        if self.use_gate and (self.seq_gate is not None):
            gate_seq = self.seq_gate(torch.cat([seq_entangled.mean(dim=1), struct_pooled.mean(dim=1)], dim=-1)).unsqueeze(1)
        else:
            gate_seq = torch.ones((B, 1, 1), device=seq_emb.device, dtype=seq_emb.dtype)
        seq_out = seq_emb + gate_seq * seq_entangled

        struct_entangled = w_struct.view(B, 1, 1, 1) * struct_self + w_mix.view(B, 1, 1, 1) * struct_cross
        if self.use_gate and (self.struct_gate is not None):
            gate_struct = self.struct_gate(torch.cat([seq_pooled.squeeze(1), struct_entangled.mean(dim=(1, 2))], dim=-1)).unsqueeze(1).unsqueeze(1)
        else:
            gate_struct = torch.ones((B, 1, 1, 1), device=seq_emb.device, dtype=seq_emb.dtype)
        struct_out = struct_emb + gate_struct * struct_entangled

        return seq_out, struct_out


class SSDNEnhanced(nn.Module):
    """增强版 SSDN：ITP + EntanglementAttention

    新增参数：use_itp / itp_weights / use_cross / use_gate / cross_direction
    用于做消融实验时快速开关各模块。
    """
    def __init__(self, embed_dim, pair_dim, num_layers=8, num_heads=4, cross_heads_start=4, dropout=0.1,
                 use_itp=True, itp_weights=None, use_cross=True, use_gate=True, cross_direction='both'):
        super().__init__()
        # interaction type percepter: either learnable or fixed (for w/o ITP)
        if use_itp:
            self.itp = InteractionTypePerceptor(embed_dim)
        else:
            # default fallback weights if not provided
            default_weights = itp_weights if itp_weights is not None else [0.5, 0.3, 0.2]
            self.itp = FixedITP(default_weights)

        # build layers with configurable cross/gate behavior
        self.layers = nn.ModuleList([
            EntanglementAttention(
                embed_dim, pair_dim,
                num_heads=num_heads,
                cross_heads=min(cross_heads_start + i, 16),
                use_cross=use_cross,
                use_gate=use_gate,
                cross_direction=cross_direction,
            )
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
        # Geometric encoder removed: geometric inputs are no longer used here.
        # The forward signature keeps the atom_coords/residue_mask/chain_mask
        # parameters for compatibility but they are ignored.

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
