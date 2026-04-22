import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry import angstrom_to_nm, pairwise_dihedrals
from models.encoders.layers import AngularEncoding


class GeometricAttention(nn.Module):
    """A lightweight geometric attention producing pairwise residue features.

    Enhancements:
    - keep richer distance distribution via gaussian/RBF pooling instead of a single mean scalar
    - optional k-NN attention sparsification (use_k_nn)
    - incorporate chain identity bias into logits
    - include amino-acid type embeddings to enrich residue descriptors

    Returns tensor of shape (N, L, L, out_dim).
    """
    def __init__(self, res_dim, geo_dim=64, out_dim=320, num_heads=4, use_k_nn=64, aa_vocab=30, aa_emb_dim=8):
        super().__init__()
        self.res_dim = res_dim
        self.geo_dim = geo_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim

        # residue descriptor projection
        self.res_proj = nn.Sequential(
            nn.Linear(res_dim, res_dim), nn.ReLU(), nn.Linear(res_dim, res_dim)
        )

        # geometric feature projector (will take gaussian dists + dihedrals)
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, geo_dim), nn.ReLU(), nn.Linear(geo_dim, geo_dim)
        )

        # angular encoding module (instantiate here so its buffers move with .to(device))
        self.dihedral_embed = AngularEncoding()

        # amino-acid embedding (projected into residue descriptor space)
        self.aa_emb = nn.Embedding(aa_vocab, aa_emb_dim)
        self.aa_proj = nn.Linear(aa_emb_dim, res_dim)

        # q/k/v projections
        self.Wq = nn.Linear(res_dim, out_dim)
        self.Wk = nn.Linear(res_dim + geo_dim, out_dim)
        self.Wv = nn.Linear(res_dim + geo_dim, out_dim)

        # output projection: will project concatenated [out_i, out_j, geo] -> out_dim
        self.out_proj = nn.Linear(out_dim * 2 + geo_dim, out_dim)

        # simple MLP to produce final pair embedding
        self.final_mlp = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

        # parameters
        self.use_k_nn = use_k_nn
        # per-head learnable chain-same bias (added to logits when same chain)
        self.chain_bias = nn.Parameter(torch.zeros(self.num_heads))

        # radial basis for distance distribution (RBF centers + widths)
        # use a small fixed set of centers for gaussian expansion
        self.register_buffer('rbf_centers', torch.linspace(0.0, 2.0, steps=min(8, max(4, geo_dim // 8))))
        self.register_buffer('rbf_widths', torch.full_like(self.rbf_centers, 0.1))

    def _compute_geometric(self, pos_atoms, mask_atoms, aa_pair):
        # pos_atoms: (N, L, A, 3), mask_atoms: (N, L, A)
        N, L, A, _ = pos_atoms.shape
        # pairwise atom distances (N, L, L, A*A)
        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],
            dim=-1, ord=2,
        )).reshape(N, L, L, -1)  # (N, L, L, A*A)

        # mask for valid atom pairs
        mask_atom_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(N, L, L, -1)
        denom = mask_atom_pair.sum(-1, keepdim=True).clamp(min=1.0)

        # RBF expansion of all atom-pair distances, then mean-pool across atom pairs
        # compute gaussian RBFs: (N,L,L,num_rbfs, A*A)
        centers = self.rbf_centers.view(*([1] * 3), -1, 1)  # (1,1,1,C,1)
        widths = self.rbf_widths.view(*([1] * 3), -1, 1)
        d_exp = d.view(N, L, L, 1, -1)
        rbf = torch.exp(-0.5 * ((d_exp - centers) ** 2) / (widths ** 2))  # (N,L,L,C,A*A)
        # mask atom pairs and pool
        rbf = rbf * mask_atom_pair.view(N, L, L, 1, -1)
        rbf_sum = rbf.sum(-1) / denom  # (N, L, L, C)

        # simple dihedral embedding
        dihed = pairwise_dihedrals(pos_atoms)  # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)  # (N, L, L, D_ang)

        # concat rbf distribution + dihedral features
        geo = torch.cat([rbf_sum, feat_dihed], dim=-1)
        # project to geo_dim
        # If dimension mismatch, pad/trim
        if geo.shape[-1] < self.geo_dim:
            pad = geo.new_zeros(list(geo.shape[:-1]) + [self.geo_dim - geo.shape[-1]])
            geo = torch.cat([geo, pad], dim=-1)
        elif geo.shape[-1] > self.geo_dim:
            geo = geo[..., :self.geo_dim]
        geo = self.geo_proj(geo)
        return geo  # (N, L, L, geo_dim)

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms):
        """Compute pair embeddings.

        Returns: (N, L, L, out_dim)
        """
        N, L = aa.size()
        # residue-level descriptor: pool atom coords (e.g., CA/CB or mean)
        mask_atoms_f = mask_atoms.float()
        denom = mask_atoms_f.sum(-1, keepdim=True).clamp(min=1.0)
        res_coords = (pos_atoms * mask_atoms_f.unsqueeze(-1)).sum(dim=2) / denom  # (N, L, 3)

        # expand descriptor by small MLP and add AA embedding
        r = self.res_proj(res_coords)  # (N, L, res_dim)
        if aa is not None:
            try:
                aa_idx = aa.long().to(res_coords.device)
                aa_e = self.aa_emb(aa_idx)
                r = r + self.aa_proj(aa_e)
            except Exception:
                pass

        geo = self._compute_geometric(pos_atoms, mask_atoms, None)  # (N, L, L, geo_dim)

        # build q/k/v
        q = self.Wq(r)  # (N, L, out_dim)
        # prepare k,v by concatenating r_j and geo_ij
        r_j = r.unsqueeze(1).expand(-1, L, -1, -1)  # (N, L, L, res_dim)
        kv_in = torch.cat([r_j, geo], dim=-1)  # (N, L, L, res_dim+geo_dim)
        k = self.Wk(kv_in)  # (N, L, L, out_dim)
        v = self.Wv(kv_in)

        # compute attention: for each i attend over j
        # reshape for matmul: q (N, L, H, Dh), k (N, L, L, H, Dh)
        H = self.num_heads
        Dh = self.head_dim
        qh = q.view(N, L, H, Dh).transpose(1, 2)  # (N, H, L, Dh)
        kh = k.view(N, L, L, H, Dh).permute(0, 3, 1, 2, 4)  # (N, H, L, L, Dh)
        vh = v.view(N, L, L, H, Dh).permute(0, 3, 1, 2, 4)  # (N, H, L, L, Dh)

        # compute logits: (N, H, L, L)
        logits = torch.einsum('nhld,nhmld->nhlm', qh, kh) / (Dh ** 0.5)

        # mask invalid residues: consider a residue valid if any atom exists
        mask_residue = mask_atoms.any(dim=-1).bool()
        mask_pair = mask_residue[:, None, :, None] & mask_residue[:, None, None, :]
        # mask_pair shape (N, 1, L, L) -> expand to (N, H, L, L)
        key_mask = ~mask_pair.expand(-1, H, -1, -1)

        # incorporate chain identity bias: prefer attending to residues in same chain
        if chain_nb is not None:
            chain_nb = chain_nb.to(logits.device)
            same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :]).float()  # (N, L, L)
            logits = logits + same_chain.unsqueeze(1) * self.chain_bias.view(1, H, 1, 1)

        # optionally restrict attention to k nearest neighbors (by residue-centroid distance)
        if self.use_k_nn is not None and self.use_k_nn > 0 and self.use_k_nn < L:
            # compute residue centroid distances (N, L, L)
            res_d = angstrom_to_nm(torch.linalg.norm(res_coords[:, :, None, :] - res_coords[:, None, :, :], dim=-1))
            # mark invalid j with large distance so they won't be selected
            invalid_j = ~mask_residue
            large_val = float(1e6)
            res_d_sel = res_d.clone()
            res_d_sel = res_d_sel.masked_fill(invalid_j[:, None, :], large_val)
            k = min(self.use_k_nn, L)
            # indices: (N, L, k)
            idxs = torch.topk(res_d_sel, k, dim=-1, largest=False).indices
            knn_mask = torch.zeros_like(res_d, dtype=torch.bool)
            Nidx = torch.arange(res_d.shape[0], device=res_d.device)[:, None, None].expand(res_d.shape[0], res_d.shape[1], k)
            Iidx = torch.arange(res_d.shape[1], device=res_d.device)[None, :, None].expand(res_d.shape[0], res_d.shape[1], k)
            knn_mask[Nidx, Iidx, idxs] = True
            key_mask = key_mask | (~knn_mask.unsqueeze(1).expand(-1, H, -1, -1))

        # Fill masked logits with a large (dtype-aware) negative value instead of -inf
        # to avoid producing NaNs when an entire row is masked (softmax of all -inf).
        neg_inf = torch.finfo(logits.dtype).min / 2.0
        fill_val = torch.tensor(neg_inf, dtype=logits.dtype, device=logits.device)
        logits = logits.masked_fill(key_mask, fill_val)

        attn = torch.softmax(logits, dim=-1)  # (N, H, L, L)

        out_heads = torch.einsum('nhlm,nhmld->nhld', attn, vh)  # (N, H, L, Dh)
        out = out_heads.permute(0, 2, 1, 3).contiguous().view(N, L, -1)  # (N, L, out_dim)

        # expand to pairwise by broadcasting out_i and out_j differences
        # produce a symmetric-ish pair feature by outer concatenation
        out_i = out.unsqueeze(2).expand(-1, -1, L, -1)
        out_j = out.unsqueeze(1).expand(-1, L, -1, -1)
        pair_feat = torch.cat([out_i, out_j, geo], dim=-1)
        # project to desired out_dim
        pair_out = self.final_mlp(self.out_proj(pair_feat))
        # explicitly align mask to (N, L, L, 1) before applying to pair_out
        pair_out = pair_out * mask_pair.squeeze(1).unsqueeze(-1)
        # remove the singleton chain dimension to match previous (N, L, L, feat_dim)
        if pair_out.size(1) == 1:
            pair_out = pair_out.squeeze(1)
        return pair_out


class ResiduePairEncoder(nn.Module):
    """Compatibility wrapper that can fuse GeometricAttention with the legacy
    pair encoder via simple concat + projection.

    By default `use_legacy=True` and the wrapper will instantiate the old
    `models.encoders.pair.ResiduePairEncoder` (if available), compute both
    features and apply `Linear(concat(geo, legacy)) -> feat_dim`.
    """
    def __init__(self, feat_dim, max_num_atoms=4, max_aa_types=30, max_relpos=32, use_legacy=True):
        super().__init__()
        self.feat_dim = feat_dim
        # choose geo_dim smaller than feat_dim
        geo_dim = min(64, feat_dim)
        # pass max_aa_types through so AA embedding size matches data
        self.geo_attn = GeometricAttention(res_dim=3, geo_dim=geo_dim, out_dim=feat_dim, num_heads=4, aa_vocab=max_aa_types)

        # optionally instantiate legacy encoder
        self.use_legacy = use_legacy
        self.legacy_enc = None
        if self.use_legacy:
            try:
                from models.encoders.pair import ResiduePairEncoder as LegacyPairEncoder
                # instantiate legacy encoder with same output dim (feat_dim)
                self.legacy_enc = LegacyPairEncoder(feat_dim, max_num_atoms, max_aa_types, max_relpos)
            except Exception:
                self.legacy_enc = None

        in_dim = feat_dim * (2 if self.legacy_enc is not None else 1)
        self.fuse_proj = nn.Sequential(nn.Linear(in_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms):
        # geometric features
        geo = self.geo_attn(aa, res_nb, chain_nb, pos_atoms, mask_atoms)  # (N,L,L,D)

        legacy = None
        if self.legacy_enc is not None:
            try:
                legacy = self.legacy_enc(aa, res_nb, chain_nb, pos_atoms, mask_atoms)
            except Exception:
                legacy = None

        if legacy is not None:
            # assume legacy last-dim == feat_dim; concat and project
            fused = torch.cat([geo, legacy], dim=-1)
        else:
            fused = geo

        out = self.fuse_proj(fused)

        # recompute mask_pair from mask_atoms to zero-out invalid pairs
        mask_residue = mask_atoms.any(dim=-1).bool()
        mask_pair = mask_residue[:, None, :, None] & mask_residue[:, None, None, :]
        out = out * mask_pair.squeeze(1).unsqueeze(-1)

        return out
