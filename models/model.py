import torch.nn as nn
import torch
import esm
from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from models.encoders.geometric_attention import ResiduePairEncoder
from models.register import ModelRegister
from models.components.ssdn import SSDN, SSDNEnhanced, TaskAdaptiveDecoder
import torch.nn.functional as F
import random
from data.complex import SUPER_PROT_IDX, SUPER_RNA_IDX, SUPER_CPLX_IDX, SUPER_CHAIN_IDX

R = ModelRegister()
from models.components.interaction_adapter import InteractionAdapter

def load_esm(esm_type):
    import os
    from pathlib import Path
    # allow overriding via env var `ESM_LOCAL_WEIGHTS`
    local_weights = os.environ.get('ESM_LOCAL_WEIGHTS')
    if local_weights is None:
        repo_root = Path(__file__).resolve().parents[1]
        local_weights = str(repo_root / 'weights' / 'esm2_t33_650M_UR50D.pt')

    if esm_type == '650M':
        # Prepare local torch hub cache if local weights exist, to prevent esm.pretrained from downloading.
        try:
            if os.path.exists(local_weights):
                repo_root = Path(__file__).resolve().parents[1]
                cache_root = repo_root / 'hf_cache'
                checkpoints_dir = cache_root / 'hub' / 'checkpoints'
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                dest = checkpoints_dir / Path(local_weights).name
                if not dest.exists():
                    try:
                        dest.symlink_to(Path(local_weights).resolve())
                    except Exception:
                        import shutil
                        shutil.copy2(local_weights, str(dest))
                os.environ['TORCH_HOME'] = str(cache_root)
                try:
                    torch.hub.set_dir(str(cache_root / 'hub'))
                except Exception:
                    pass
        except Exception:
            pass

        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        try:
            if os.path.exists(local_weights):
                state = torch.load(local_weights, map_location='cpu')
                if isinstance(state, dict) and 'model' in state and not any(k.startswith('embed_tokens') for k in state.keys()):
                    state = state['model']
                try:
                    model.load_state_dict(state)
                except Exception:
                    if isinstance(state, dict) and 'state_dict' in state:
                        model.load_state_dict(state['state_dict'])
        except Exception:
            pass
    elif esm_type == '3B':
        model, _ = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm_type == '15B':
        model, _ = esm.pretrained.esm2_t48_15B_UR50D()
    elif esm_type == '150M':
        model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    elif esm_type == '35M':
        model, _ = esm.pretrained.esm2_t12_35M_UR50D()
    elif esm_type == '8M':
        model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise NotImplementedError
    feat_size = model.embed_dim
    return model, feat_size
    
def load_rinalmo(rinalmo_weights, rinalmo_type):
    if rinalmo_type == '650M':
        size = 'giga'
    elif rinalmo_type == '150M':
        size = 'mega'
    elif rinalmo_type == '35M':
        size = 'micro'
    elif rinalmo_type == '8M':
        size = 'nano'
    config = model_config(size)
    model = RiNALMo(config)
    # alphabet = Alphabet(**config['alphabet'])
    model.load_state_dict(torch.load(rinalmo_weights))
    feat_size = config.globals.embed_dim
    return model, feat_size

def cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx):
    # print("Input shape:", prot_embedding.shape, na_embedding.shape)
    # result = prot_embedding.new_full([len(prot_embedding), seq_len, prot_embedding.shape[-1]], 0) # (N, L, E)
    new_complexes = []
    masks = []
    for i in range(len(prot_embedding)):
        item_prot_embed = prot_embedding[i]
        item_prot_mask = prot_mask[i]
        item_na_embed = na_embedding[i]
        item_na_mask = na_mask[i]
        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)
        if patch_idx is not None:
            selected = torch.index_select(selected, 0, patch_idx[i])
        p1d = (0, 0, 0, max_len-len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)
        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad)
    result = torch.stack(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    return result, masks

def segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx=None):
    cum_prot = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(prot_chains), dim=0)]).int()
    cum_na = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(na_chains), dim=0)]).int()
    new_complexes = []
    masks = []
    for i, (s_prot, e_prot, s_na, e_na) in enumerate(zip(cum_prot[:-1], cum_prot[1:], cum_na[:-1], cum_na[1:])):
        item_prot_embed = prot_embedding[s_prot:e_prot].reshape((-1, prot_embedding.shape[-1]))
        item_prot_mask = prot_mask[s_prot:e_prot].reshape(-1)
        item_na_embed = na_embedding[s_na: e_na].reshape((-1, na_embedding.shape[-1]))
        item_na_mask = na_mask[s_na: e_na].reshape(-1)
        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)
        if patch_idx is not None:
            selected = torch.index_select(selected, 0, patch_idx[i])
        p1d = (0, 0, 0, max_len-len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)
        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        # # selected_pad = torch.cat([selected, torch.zeros((seq_len-len(selected), prot_embedding.shape[-1]), device=selected.device)], dim=0)
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad.unsqueeze(0))
    result = torch.cat(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    # print("Result shape:", result)
    return result, masks

@R.register('PRORNA_SSDN')
class ESM2RiNALMo(nn.Module):
    def __init__(self, 
                 rinalmo_weights='./weights/rinalmo_giga_pretrained.pt',
                 esm_type='650M',
                 rinalmo_type='650M',
                 pooling='mean',
                 output_dim=1,
                 pair_dim=320,
                 fix_lms=True,
                 
                 representation_layer=33,
                 dist_dim=40,
                 **kwargs
                 ):
        super(ESM2RiNALMo, self).__init__()
        self.esm, esm_feat_size = load_esm(esm_type)
        self.rinalmo, rinalmo_feat_size = load_rinalmo(rinalmo_weights, rinalmo_type)
        # Optionally freeze pretrained language/model encoders to save memory and compute
        # LoRA removed: freeze LMs when requested
        if fix_lms:
            for p in self.esm.parameters():
                p.requires_grad = False
            for p in self.rinalmo.parameters():
                p.requires_grad = False
            try:
                self.esm.eval()
            except Exception:
                pass
            try:
                self.rinalmo.eval()
            except Exception:
                pass
        self.pair_encoder = ResiduePairEncoder(pair_dim, max_num_atoms=4)  # N, CA, C, O,
        # Fusion: allow replacing CoFormer with SSDN via config 'fusion'
        fusion_cfg = kwargs.get('fusion', None)
        if fusion_cfg is not None and fusion_cfg.get('type', '').lower() == 'ssdn':
            ssdn_layers = fusion_cfg.get('layers', 2)
            ssdn_heads = fusion_cfg.get('heads', 4)
            ssdn_pair_dim = fusion_cfg.get('pair_dim', pair_dim)
            ssdn_cross_heads = fusion_cfg.get('cross_heads', ssdn_heads)
            ssdn_dropout = fusion_cfg.get('dropout', 0.0)
            # use configured embed dim from coformer section as complex_dim
            embed_dim_cfg = kwargs.get('coformer', {}).get('embed_dim', pair_dim)
            self.c_former = SSDN(embed_dim_cfg, ssdn_pair_dim, num_layers=ssdn_layers, num_heads=ssdn_heads, cross_heads=ssdn_cross_heads, dropout=ssdn_dropout)
        elif fusion_cfg is not None and fusion_cfg.get('type', '').lower() == 'ssdn_enhanced' or (fusion_cfg is not None and fusion_cfg.get('use_enhanced', False)):
            # build SSDNEnhanced using coformer embed dim
            ssdn_layers = fusion_cfg.get('layers', 8)
            ssdn_heads = fusion_cfg.get('heads', 4)
            ssdn_pair_dim = fusion_cfg.get('pair_dim', pair_dim)
            cross_heads_start = fusion_cfg.get('cross_heads_start', 4)
            ssdn_dropout = fusion_cfg.get('dropout', 0.1)
            # embed dim from coformer config
            embed_dim_cfg = kwargs.get('coformer', {}).get('embed_dim', pair_dim)
            self.c_former = SSDNEnhanced(embed_dim_cfg, ssdn_pair_dim, num_layers=ssdn_layers, num_heads=ssdn_heads, cross_heads_start=cross_heads_start, dropout=ssdn_dropout)
        else:
            # Build SSDN from original CoFormer-style config to preserve behavior
            coformer_cfg = kwargs.get('coformer', {})
            embed_dim_cfg = coformer_cfg.get('embed_dim', pair_dim)
            cf_pair_dim = coformer_cfg.get('pair_dim', pair_dim)
            cf_num_blocks = coformer_cfg.get('num_blocks', 6)
            cf_num_heads = coformer_cfg.get('num_heads', 4)
            cf_cross_heads = coformer_cfg.get('cross_heads', cf_num_heads)
            cf_dropout = coformer_cfg.get('attention_dropout', 0.0)
            self.c_former = SSDN(embed_dim_cfg, cf_pair_dim, num_layers=cf_num_blocks, num_heads=cf_num_heads, cross_heads=cf_cross_heads, dropout=cf_dropout)
        self.representation_layer = representation_layer
        self.proj = 0
        if esm_feat_size != rinalmo_feat_size:
            self.proj = 1
            self.project_feat= nn.Linear(esm_feat_size, rinalmo_feat_size)
        # determine complex embedding dimension with safe fallbacks
        self.complex_dim = kwargs.get('coformer', {}).get('embed_dim',
                    kwargs.get('embed_dim', kwargs.get('model', {}).get('embed_dim', pair_dim)))
        self.feat_size = rinalmo_feat_size
        self.proj_cplx= nn.Linear(self.feat_size, self.complex_dim)
        # LoRA support removed. Freeze LMs if requested.
        if fix_lms:
            for p in self.rinalmo.parameters():
                p.requires_grad_(False)
            for p in self.esm.parameters():
                p.requires_grad_(False)
                
        self.pooling = pooling
        print("Pooling Strategy:", self.pooling)
        if self.pooling == 'token':
            self.prot_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.rna_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            nn.init.normal_(self.prot_embedding)
            nn.init.normal_(self.rna_embedding)
            nn.init.normal_(self.complex_embedding)
        if pair_dim != self.complex_dim:
            self.z_proj = nn.Linear(pair_dim, self.complex_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )
        # For mask distance pretraining
        self.mask_token = nn.Parameter(torch.randn(size=(1, pair_dim)))
        self.dist_head = nn.Sequential(
            nn.Linear(pair_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, dist_dim)
        )
        # optional interaction adapter config (for pooled / token special tokens)
        ia_cfg = kwargs.get('interaction_adapter', None)
        if ia_cfg is not None and ia_cfg.get('enable', False):
            ia_embed = ia_cfg.get('embed_dim', self.complex_dim)
            ia_heads = ia_cfg.get('num_heads', 8)
            ia_dropout = ia_cfg.get('dropout', 0.0)
            self.interaction_adapter_prot = InteractionAdapter(ia_embed, ia_heads, ia_dropout)
            self.interaction_adapter_rna = InteractionAdapter(ia_embed, ia_heads, ia_dropout)
            self._interaction_enabled = True
        else:
            self._interaction_enabled = False

        # Optional TaskAdaptiveDecoder
        decoder_cfg = kwargs.get('decoder', None)
        if decoder_cfg is not None:
            task_emb_dim = decoder_cfg.get('task_emb_dim', 64)
            try:
                self.decoder = TaskAdaptiveDecoder(self.complex_dim, task_emb_dim=task_emb_dim)
            except Exception:
                self.decoder = None
    
    def _forward(self, input, strategy='separate', need_mask=False):
        prot_input = input['prot']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        na_input = input['na']
        na_chains = input['na_chains']
        na_mask = input['na_mask']

        with torch.cuda.amp.autocast():
            prot_embedding = self.esm(prot_input, repr_layers=[self.representation_layer], return_contacts=False)['representations'][self.representation_layer]
            na_embedding = self.rinalmo(na_input)['representation']
            if self.proj:
                prot_embedding = self.project_feat(prot_embedding)

        prot_embedding = prot_embedding.float()
        na_embedding = na_embedding.float()
        max_len = input['pos_atoms'].shape[1]
        # Adjust the embeddings from LMs for CoFormer
        if 'patch_idx' in input:
            patch_idx = input['patch_idx']
        else:
            patch_idx = None
        if strategy == 'separate':
            # input shape [N', L], where N' is flexible in every batch
            out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx)
            assert out_embedding.shape[0] == input['size']
        else:
            out_embedding, masks = cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx)
            assert out_embedding.shape[0] == input['size']

        out_embedding = self.proj_cplx(out_embedding)
        key_padding_mask = ~masks
        
        aa=input['restype']
        res_nb=input['res_nb']
        chain_nb=input['chain_nb']
        pos_atoms=input['pos_atoms']
        mask_atoms=input['mask_atoms']
        
        if self.pooling == 'token':
            mask_special = torch.zeros((len(out_embedding), 1), device=out_embedding.device, dtype=key_padding_mask.dtype)
            cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
            prot_embed = self.prot_embedding.repeat(len(out_embedding), 1, 1)
            rna_embed = self.rna_embedding.repeat(len(out_embedding), 1, 1)
            
            out_embedding = torch.cat([cplx_embed, prot_embed, rna_embed, out_embedding], dim=1)
            key_padding_mask = torch.cat([mask_special, mask_special, mask_special, key_padding_mask], dim=1)
            
            cplx_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_CPLX_IDX
            prot_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_PROT_IDX
            rna_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_RNA_IDX
            aa = torch.cat([cplx_type, prot_type, rna_type, aa], dim=1)
            
            res_nb_cplx = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 0
            res_nb_prot = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 1
            res_nb_rna = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 2
            
            res_nb = torch.cat([res_nb_cplx, res_nb_prot, res_nb_rna, res_nb], dim=1)
            super_chain_id = torch.ones_like(mask_special, device=out_embedding.device, dtype=chain_nb.dtype) * SUPER_CHAIN_IDX
            chain_nb = torch.cat([super_chain_id, super_chain_id, super_chain_id, chain_nb], dim=1)
            
            center_cplx = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2], 3), device=out_embedding.device, dtype=pos_atoms.dtype)
            center_prot = ((pos_atoms * (1-input['identifier'])[:, :, None, None] * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((1-input['identifier'][:, :, None]) * mask_atoms + 1e-10).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, 4, 1)
            center_rna = ((pos_atoms * (input['identifier'][:, :, None, None]) * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((input['identifier'][:, :, None]) * mask_atoms + 1e-10).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, 4, 1)
            pos_atoms = torch.cat([center_cplx, center_prot, center_rna, pos_atoms], dim=1)
            # noise = torch.randn_like(pos_atoms, dtype=torch.float32, device=pos_atoms.device)
            # pos_atoms += noise
            mask_atom = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2]), device=out_embedding.device, dtype=mask_atoms.dtype)
            mask_atom[:,:,0] = 1
            mask_atoms = torch.cat([mask_atom, mask_atom, mask_atom, mask_atoms], dim=1)


        z = self.pair_encoder(
            aa=aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
        )
        if need_mask:
            # Randomly mask a small fraction of rows/cols in the pair tensor per-sample.
            # Use a generic boolean-broadcasting approach that locates the axes
            # equal to sequence length and applies mask across them, so this
            # code is robust to extra intermediate dimensions.
            seq_len = aa.size(1)
            for i in range(z.shape[0]):
                to_mask = torch.rand(1).item() > 0.5
                if not to_mask:
                    continue
                # choose indices to mask (avoid first 3 special tokens)
                valid = list(range(3, seq_len))
                k = max(1, int(len(valid) * 0.15))
                mask_indices = random.sample(valid, k)
                if len(mask_indices) == 0:
                    continue

                # boolean vector for residues
                row_mask = torch.zeros(seq_len, dtype=torch.bool, device=z.device)
                col_mask = torch.zeros(seq_len, dtype=torch.bool, device=z.device)
                row_mask[mask_indices] = True
                col_mask[mask_indices] = True

                # mask token vector (feat_dim,)
                mask_val = self.mask_token.to(z.dtype).squeeze(0)

                # operate on z[i] (all axes except batch and feat)
                z_i = z[i]
                feat_dim = z_i.shape[-1]
                spatial_shape = list(z_i.shape[:-1])  # e.g., (L,L) or (X,L,L)

                # find axes among spatial_shape that correspond to sequence (length == seq_len)
                seq_axes = [idx for idx, s in enumerate(spatial_shape) if s == seq_len]
                if len(seq_axes) < 1:
                    # nothing to mask
                    continue

                # pick first two axes for row/col if available, otherwise use first for both
                row_axis = seq_axes[0]
                col_axis = seq_axes[1] if len(seq_axes) > 1 else seq_axes[0]

                # build broadcastable views for row and col masks
                row_view_shape = [1] * len(spatial_shape)
                row_view_shape[row_axis] = seq_len
                row_view = row_mask.view(row_view_shape).expand(*spatial_shape)

                col_view_shape = [1] * len(spatial_shape)
                col_view_shape[col_axis] = seq_len
                col_view = col_mask.view(col_view_shape).expand(*spatial_shape)

                pos_mask = (row_view | col_view)  # boolean mask over spatial positions
                n_selected = int(pos_mask.sum().item())
                if n_selected == 0:
                    continue

                # assign mask_val to selected spatial positions (last dim is feat)
                z_vals = z_i[pos_mask]  # shape (n_selected, feat_dim)
                assign_val = mask_val.unsqueeze(0).expand(n_selected, feat_dim)
                z_i[pos_mask] = assign_val
                # write back
                z[i] = z_i
            
        return out_embedding, z, key_padding_mask
        
    def forward(self, input, strategy='separate', stage='finetune', need_mask=False):
        out_embedding, z, key_padding_mask = self._forward(input, strategy, need_mask=need_mask)
        # run fusion/backbone
        cformer_out = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)

        # normalize outputs for compatibility with legacy SSDN
        # cformer_out may be (output_tokens, pair, attn) OR (fused, seq_emb, struct_emb)
        output_tokens, pair_out, attn = None, None, None
        fused_embedding = None
        seq_embedding = None
        try:
            a0, a1, a2 = cformer_out
            # detect fused (2D) vs token outputs (3D)
            if a0.dim() == 2:
                # SSDNEnhanced: a0=fused [B,E], a1=seq_emb [B,L,E]
                fused_embedding = a0
                seq_embedding = a1
                pair_out = a2
            else:
                # legacy SSDN: a0=token outputs [B,L,E]
                output_tokens = a0
                pair_out = a1
                attn = a2
        except Exception:
            # fallback: try unpacking differently
            output_tokens = cformer_out[0]
            pair_out = cformer_out[1]

        # optional interaction on pooled prot/rna tokens when using token pooling
            if self._interaction_enabled and self.pooling == 'token':
                try:
                    p = output[:, 1, :].unsqueeze(1)
                    r = output[:, 2, :].unsqueeze(1)
                    p_upd = self.interaction_adapter_prot(p, r, query_mask=None, kv_mask=torch.ones((r.shape[0], r.shape[1]), device=r.device).bool())
                    r_upd = self.interaction_adapter_rna(r, p, query_mask=None, kv_mask=torch.ones((p.shape[0], p.shape[1]), device=p.device).bool())
                    # avoid in-place modification of `output` which may be needed for autograd
                    output = output.clone()
                    output[:, 1, :] = p_upd.squeeze(1)
                    output[:, 2, :] = r_upd.squeeze(1)
                except Exception:
                    pass

        # If we have fused_embedding from SSDNEnhanced, use it; otherwise fallback to pooling tokens
        if fused_embedding is None:
            output = output_tokens if output_tokens is not None else cformer_out[0]
            if self.pooling == 'token':
                complex_embedding = output[:, 0, :].squeeze(1)
            else:
                complex_embedding = (output * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
                if self.pooling == 'mean':
                    seq_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
                    complex_embedding = complex_embedding / (seq_mask_sum + 1e-10)
        else:
            complex_embedding = fused_embedding

        # Branch by stage: pretune, mutation, finetune/multitask, or default prediction
        if stage == 'pretune':
            # -------------------------------------------CLIP feature generation ----------------------------------------------
            res_identifier = input['identifier']
            attn_mask = torch.ones((out_embedding.shape[0], out_embedding.shape[1], out_embedding.shape[1]), device=out_embedding.device).bool()
            if self.pooling == 'token':
                prot_token_identifier = torch.zeros(len(out_embedding), 1, dtype=res_identifier.dtype, device=res_identifier.device)
                rna_token_identifier = torch.ones(len(out_embedding), 1, dtype=res_identifier.dtype, device=res_identifier.device)
                res_identifier = torch.cat([prot_token_identifier, rna_token_identifier, res_identifier], dim=1)
                attn_mask[:, 1:, 1:] = (res_identifier[:, :, None] == res_identifier[:, None, :])
            attn_mask = ~attn_mask
            if torch.isnan(z).any():
                print("Found Nan in z!")
            # call c_former and normalize outputs (support SSDNEnhanced legacy and fused outputs)
            cformer_res = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False, attn_mask=attn_mask)
            # cformer_res may be (output_tokens, pair_out, attn) OR (fused, seq_emb, struct_emb)
            try:
                a0, a1, a2 = cformer_res
                if a0.dim() == 2:
                    # SSDNEnhanced: a0=fused [B,E], a1=seq_emb [B,L,E]
                    output = a1
                    pair_out = a2
                    attn = None
                else:
                    # legacy SSDN: a0=token outputs [B,L,E]
                    output = a0
                    pair_out = a1
                    attn = a2
            except Exception:
                # fallback: assume token outputs
                output = cformer_res[0]
                pair_out = cformer_res[1]
                attn = cformer_res[2] if len(cformer_res) > 2 else None

            if self.pooling == 'token':
                # output is expected to be token outputs [B, L_tokens, E]
                complex_embedding = output[:, 0, :].squeeze(1)
                prot_embedding = output[:, 1, :].squeeze(1)
                rna_embedding = output[:, 2, :].squeeze(1)
                if self._interaction_enabled:
                    p = prot_embedding.unsqueeze(1)
                    r = rna_embedding.unsqueeze(1)
                    p_upd = self.interaction_adapter_prot(p, r, query_mask=None, kv_mask=torch.ones((r.shape[0], r.shape[1]), device=r.device).bool())
                    r_upd = self.interaction_adapter_rna(r, p, query_mask=None, kv_mask=torch.ones((p.shape[0], p.shape[1]), device=p.device).bool())
                    prot_embedding = p_upd.squeeze(1)
                    rna_embedding = r_upd.squeeze(1)
            else:
                complex_embedding = (output * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
                prot_embedding = (output * (~key_padding_mask).unsqueeze(-1) * (1-input['identifier']).unsqueeze(-1)).sum(dim=1)
                rna_embedding = (output * (~key_padding_mask).unsqueeze(-1) * (input['identifier'].unsqueeze(-1))).sum(dim=1)
                if self.pooling == 'mean':
                    cplx_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
                    prot_mask_sum = ((~key_padding_mask) * (1-input['identifier'])).sum(dim=1, keepdim=True)
                    rna_mask_sum = ((~key_padding_mask) * (input['identifier'])).sum(dim=1, keepdim=True)
                    complex_embedding = complex_embedding / (cplx_mask_sum + 1e-10)
                    prot_embedding = prot_embedding / (prot_mask_sum + 1e-10)
                    rna_embedding = rna_embedding / (rna_mask_sum + 1e-10)

            similarity = F.cosine_similarity(prot_embedding[:, None, :], rna_embedding[None, :, :], dim=2)
            if torch.isnan(z).any():
                print("Found Nan in z!")
            cformer_res2 = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False, attn_mask=None)
            try:
                b0, b1, b2 = cformer_res2
                if b0.dim() == 2:
                    # SSDNEnhanced
                    output = b1
                    pair_out = b2
                else:
                    output = b0
                    pair_out = b1
            except Exception:
                output = cformer_res2[0]
                pair_out = cformer_res2[1]

            if torch.isnan(pair_out).any():
                print("Found Nan in pair_out!")
            # debug print: when DEBUG_MODEL env set, print pair_out shape and dist_head expected shapes
            import os
            if os.environ.get('DEBUG_MODEL', '0') == '1':
                try:
                    print('DEBUG: pair_out.shape before dist_head =', getattr(pair_out, 'shape', None))
                    first_lin = None
                    for m in self.dist_head.modules():
                        if isinstance(m, nn.Linear):
                            first_lin = m
                            break
                    if first_lin is not None:
                        print('DEBUG: dist_head first Linear in_features, out_features =', first_lin.in_features, first_lin.out_features)
                except Exception:
                    pass
            dist_logits = self.dist_head(pair_out)
            dist_logits = dist_logits[:, 3:, 3:, :]
            return dist_logits, similarity

        elif stage == 'mutation':
            input['prot'] = input['prot_mut']
            input['restype'] = input['mut_restype']
            out_mut, z_mut, _ = self._forward(input, strategy)
            deep = False
            if deep:
                out_forward = out_embedding - out_mut
                z_forward = z - z_mut
                
                out_inv = out_mut - out_embedding
                z_inv = z_mut - z
                
                
                output_forward, z_forward, attn = self.c_former(out_forward, z_forward, key_padding_mask=key_padding_mask, need_attn_weights=False)
                complex_embedding = output_forward + self.z_proj(z_forward).sum(-2) * 0.001
                # Default to be token embeding
                complex_embedding = complex_embedding[:, 0, :].squeeze(1)
                
                output_forward = self.pred_head(complex_embedding)
                output_forward = output_forward.squeeze(1)
                
                output_inv, z_inv, attn = self.c_former(out_inv, z_inv, key_padding_mask=key_padding_mask, need_attn_weights=False)
                complex_embedding_inv = output_inv + self.z_proj(z_inv).sum(-2) * 0.001
                # Default to be token embeding
                complex_embedding_inv = complex_embedding_inv[:, 0, :].squeeze(1)
                
                output_inv = self.pred_head(complex_embedding_inv)
                output_inv = output_inv.squeeze(1)
                
                return output_forward, output_inv
            else:
                output_wild, z_wild, attn = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)
                output_mut, z_mut, attn = self.c_former(out_mut, z_mut, key_padding_mask=key_padding_mask, need_attn_weights=False)
                wild_embedding = output_wild + self.z_proj(z_wild).sum(-2) * 0.001
                # Default to be token embeding
                wild_embedding = wild_embedding[:, 0, :].squeeze(1)
                mut_embedding = output_mut + self.z_proj(z_mut).sum(-2) * 0.001
                mut_embedding = mut_embedding[:, 0, :].squeeze(1)
                
                
                forward_embedding = wild_embedding - mut_embedding
                inv_embedding = mut_embedding - wild_embedding
                
                output_forward = self.pred_head(forward_embedding).squeeze(1)
                output_inv = self.pred_head(inv_embedding).squeeze(1)
                
                return output_forward, output_inv

        elif stage in ('finetune', 'multitask'):
            # Default prediction head for finetune (legacy single-task)
            try:
                out = self.pred_head(complex_embedding).squeeze(1)
            except Exception:
                out = self.pred_head(complex_embedding)

            # If multitask decoder exists and user requested multitask, try to use it
            if stage == 'multitask' and hasattr(self, 'decoder') and self.decoder is not None:
                try:
                    preds = {}
                    preds['delta_g'] = self.decoder(complex_embedding, task='delta_g')
                    preds['delta_delta_g'] = self.decoder(complex_embedding, task='delta_delta_g')
                    try:
                        preds['binding_site'] = self.decoder(complex_embedding, task='binding_site', seq_embedding=seq_embedding)
                    except Exception:
                        preds['binding_site'] = None
                    return preds
                except Exception:
                    # fallback to single output
                    return out

            return out

        else:
            raise NotImplementedError
            raise NotImplementedError


# Register alias name so users can reference the same implementation
# under the new name 'PRORNA_SSDN' without changing the class.
try:
    R.register('PRORNA_SSDN')(ESM2RiNALMo)
except Exception:
    # Fallback: directly assign into the register dict
    R['PRORNA_SSDN'] = ESM2RiNALMo
try:
    R.register('prorna_ssdn')(ESM2RiNALMo)
except Exception:
    R['prorna_ssdn'] = ESM2RiNALMo

        
        
            
            

