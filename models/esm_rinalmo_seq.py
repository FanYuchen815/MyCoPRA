import torch.nn as nn
import torch
import esm
from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from models.register import ModelRegister
from peft import (
    LoraConfig,
    get_peft_model,
)
from models.lora_tune import LoRAESM, LoRARiNALMo, ESMConfig, RiNALMoConfig
from models.components.valina_transformer import Transformer
from models.model import cat_pad, segment_cat_pad
R = ModelRegister()
from models.components.interaction_adapter import InteractionAdapter
from models.components.ssdn import SSDN, SSDNEnhanced

def load_esm(esm_type):
    import os
    from pathlib import Path
    # allow overriding via env var `ESM_LOCAL_WEIGHTS`
    local_weights = os.environ.get('ESM_LOCAL_WEIGHTS')
    # default fallback to repository weights folder
    if local_weights is None:
        repo_root = Path(__file__).resolve().parents[1]
        local_weights = str(repo_root / 'weights' / 'esm2_t33_650M_UR50D.pt')

    if esm_type == '650M':
        # If a local checkpoint exists, place it into a local torch hub cache
        # and point TORCH_HOME there so esm.pretrained will reuse it instead of downloading.
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
                        # fallback to copy if symlink not allowed
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
        # try to load local checkpoint if present (defensive attempt)
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
    else:
        raise NotImplementedError
    feat_size = model.embed_dim
    return model, feat_size
    
def load_rinalmo(rinalmo_weights):
    config = model_config('giga')
    model = RiNALMo(config)
    # alphabet = Alphabet(**config['alphabet'])
    model.load_state_dict(torch.load(rinalmo_weights))
    feat_size = config.globals.embed_dim
    return model, feat_size

def segment_pool(input, chains, mask, pooling):
    # input shape [N', L, E], mask_shape [N', L]
    result = input.new_full([len(chains), input.shape[-1]], 0) # (N, E)
    mask_result = mask.new_full([len(chains), 1], 0) #(N, 1)
    input_flattened = input.reshape((-1, input.shape[-1])) #(N'*L, E)
    mask_flattened = mask.reshape((-1, 1)) #(N'*L, 1)
    # print("Shapes:", result.shape, mask_result.shape, input_flattened.shape, mask_flattened.shape)
    # segment_id shape (N', )
    segment_id = torch.tensor(sum([[i] * chain for i, chain in enumerate(chains)], start=[]), device=result.device, dtype=torch.int64)
    segment_id = segment_id.repeat_interleave(input.shape[1]) #(N'*L)
    result.scatter_add_(0, segment_id.unsqueeze(1).expand_as(input_flattened), input_flattened*mask_flattened)
    mask_result.scatter_add_(0, segment_id.unsqueeze(1), mask_flattened)
    mask_result.reshape((-1, ))
    
    if pooling == 'mean':
        result = result / (mask_result + 1e-10)
    
    return result

@R.register('esm2_rinalmo_seq')
class ESM2RiNALMo(nn.Module):
    def __init__(self, 
                 rinalmo_weights='./weights/rinalmo_giga_pretrained.pt',
                 esm_type='650M',
                 pooling='token',
                 output_dim=1,
                 fix_lms=True,
                 lora_tune=False,
                 lora_rank=16,
                 lora_alpha=32,
                 representation_layer=33,
                 vallina=True,
                 **kwargs
                 ):
        super(ESM2RiNALMo, self).__init__()
        self.esm, esm_feat_size = load_esm(esm_type)
        self.rinalmo, rinalmo_feat_size = load_rinalmo(rinalmo_weights)
        self.vallina=vallina
        # if esm_feat_size != rinalmo_feat_size:
        #     self.project_layer = nn.Linear(esm_feat_size, rinalmo_feat_size)
        self.cat_size = esm_feat_size + rinalmo_feat_size
        self.feat_size = rinalmo_feat_size
        self.representation_layer = representation_layer
        # Fusion / backbone: support legacy Transformer or new SSDN fusion module
        fusion_cfg = kwargs.get('fusion', None)
        if fusion_cfg is not None and fusion_cfg.get('type', '').lower() == 'ssdn':
            ssdn_layers = fusion_cfg.get('layers', 2)
            ssdn_heads = fusion_cfg.get('heads', 4)
            ssdn_pair_dim = fusion_cfg.get('pair_dim', kwargs.get('coformer', {}).get('pair_dim', 40))
            ssdn_cross_heads = fusion_cfg.get('cross_heads', ssdn_heads)
            ssdn_dropout = fusion_cfg.get('dropout', 0.0)
            self.transformer = SSDN(self.complex_dim, ssdn_pair_dim, num_layers=ssdn_layers, num_heads=ssdn_heads, cross_heads=ssdn_cross_heads, dropout=ssdn_dropout)
        elif fusion_cfg is not None and fusion_cfg.get('type', '').lower() == 'ssdn_enhanced' or (fusion_cfg is not None and fusion_cfg.get('use_enhanced', False)):
            ssdn_layers = fusion_cfg.get('layers', 8)
            ssdn_heads = fusion_cfg.get('heads', 4)
            ssdn_pair_dim = fusion_cfg.get('pair_dim', kwargs.get('coformer', {}).get('pair_dim', 40))
            cross_heads_start = fusion_cfg.get('cross_heads_start', ssdn_heads)
            ssdn_dropout = fusion_cfg.get('dropout', 0.1)
            embed_dim_cfg = kwargs.get('coformer', {}).get('embed_dim', pair_dim if 'pair_dim' in locals() else 320)
            self.transformer = SSDNEnhanced(embed_dim_cfg, ssdn_pair_dim, num_layers=ssdn_layers, num_heads=ssdn_heads, cross_heads_start=cross_heads_start, dropout=ssdn_dropout)
        else:
            self.transformer = Transformer(**kwargs['transformer'])
        self.complex_dim = kwargs['transformer']['embed_dim']
        self.proj_cplx= nn.Linear(self.feat_size, self.complex_dim)
        self.pooling = pooling
        if self.pooling == 'token':
            self.prot_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.rna_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            nn.init.normal_(self.prot_embedding)
            nn.init.normal_(self.rna_embedding)
            nn.init.normal_(self.complex_embedding)
        if lora_tune:
            # copied from LongLoRA
            rinalmo_lora_config = LoraConfig(
                r=lora_rank,
                bias="none",
                lora_alpha=lora_alpha
            )
            esm_lora_config = LoraConfig(
                r=lora_rank,
                bias="none",
                lora_alpha=lora_alpha
            )
            rinalmo_config = RiNALMoConfig()
            esm_config = ESMConfig()
            self.rinalmo = LoRARiNALMo(self.rinalmo, rinalmo_config)
            # print(esm_config)
            self.esm = LoRAESM(self.esm, esm_config)
            # print("ESM:", self.esm)
            self.rinalmo = get_peft_model(self.rinalmo, rinalmo_lora_config)
            # print("Get RINALMO DONE!!!!!")
            self.esm = get_peft_model(self.esm, esm_lora_config)
            # print("Get ESM DONE!!!!!")

        elif fix_lms:
            for p in self.rinalmo.parameters():
                p.requires_grad_(False)
            for p in self.esm.parameters():
                p.requires_grad_(False)
        self.pooling = pooling
        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )
        if self.vallina:
            print("Using vallina version!")
            self.cat_pred_head = nn.Sequential(
            nn.Linear(self.cat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim) 
            )
        # optional interaction adapter config
        ia_cfg = kwargs.get('interaction_adapter', None)
        if ia_cfg is not None and ia_cfg.get('enable', False):
            ia_embed = ia_cfg.get('embed_dim', self.feat_size)
            ia_heads = ia_cfg.get('num_heads', 8)
            ia_dropout = ia_cfg.get('dropout', 0.0)
            # adapters for prot->rna and rna->prot when using pooled (vallina) embeddings
            self.interaction_adapter_prot = InteractionAdapter(ia_embed, ia_heads, ia_dropout)
            self.interaction_adapter_rna = InteractionAdapter(ia_embed, ia_heads, ia_dropout)
            self._interaction_enabled = True
        else:
            self._interaction_enabled = False

    
    def forward(self, input, strategy='separate'):
        prot_input = input['prot']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        na_input = input['na']
        na_chains = input['na_chains']
        na_mask = input['na_mask']
        # print("Input Shape:", prot_input.shape, na_input.shape, prot_chains, na_chains)
        with torch.cuda.amp.autocast():
            prot_embedding = self.esm(prot_input, repr_layers=[self.representation_layer], return_contacts=False)['representations'][self.representation_layer]
            na_embedding = self.rinalmo(na_input)['representation']
        
        # Vallina implementation with mean pooling
        # print("Original Embedding:", prot_embedding.shape, na_embedding.shape)
        if self.vallina:
            if strategy == 'separate':
                # input shape [N', L], where N' is flexible in every batch
                prot_embedding = segment_pool(prot_embedding, prot_chains, prot_mask, pooling=self.pooling)
                na_embedding = segment_pool(na_embedding, na_chains, na_mask, pooling=self.pooling)
            else:
                if self.pooling == 'max':
                    prot_embedding = (prot_embedding * prot_mask.unsqueeze(-1)).max(dim=1)[0]
                    na_embedding = (na_embedding * na_mask.unsqueeze(-1)).max(dim=1)[0]
                else:
                    prot_embedding = (prot_embedding * prot_mask.unsqueeze(-1)).sum(dim=1)
                    na_embedding = (na_embedding * na_mask.unsqueeze(-1)).sum(dim=1)
                    if self.pooling == 'mean':
                        # Prot_mask: [N, L]
                        prot_mask_sum = prot_mask.sum(dim=1, keepdim=True)
                        na_mask_sum = na_mask.sum(dim=1, keepdim=True)
                        prot_embedding = prot_embedding / (prot_mask_sum + 1e-10)
                        na_embedding = na_embedding / (na_mask_sum + 1e-10)
            complex_embedding = torch.cat([prot_embedding, na_embedding], dim=1)
            # apply lightweight cross-attention between pooled prot and rna embeddings if enabled
            if self._interaction_enabled:
                # ensure shape (B, N, E); here prot_embedding/na_embedding are (B, E)
                p = prot_embedding.unsqueeze(1)
                r = na_embedding.unsqueeze(1)
                # use other side as KV
                p_upd = self.interaction_adapter_prot(p, r, query_mask=None, kv_mask=torch.ones((r.shape[0], r.shape[1]), device=r.device).bool())
                r_upd = self.interaction_adapter_rna(r, p, query_mask=None, kv_mask=torch.ones((p.shape[0], p.shape[1]), device=p.device).bool())
                prot_embedding = p_upd.squeeze(1)
                na_embedding = r_upd.squeeze(1)
                complex_embedding = torch.cat([prot_embedding, na_embedding], dim=1)
            output = self.cat_pred_head(complex_embedding)
            output = output.squeeze(1)
        else:   
            prot_embedding = prot_embedding.float()
            na_embedding = na_embedding.float()
            # print("Original Embedding:", prot_embedding, na_embedding)
            max_len = input['pos_atoms'].shape[1]
            # Adjust the embeddings from LMs for CFormer
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
            
            if self.pooling == 'token':
                mask_special = torch.zeros((len(out_embedding), 1), device=out_embedding.device, dtype=key_padding_mask.dtype)
                cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
                prot_embed = self.prot_embedding.repeat(len(out_embedding), 1, 1)
                rna_embed = self.rna_embedding.repeat(len(out_embedding), 1, 1)
                out_embedding = torch.cat([cplx_embed, prot_embed, rna_embed, out_embedding], dim=1)
                key_padding_mask = torch.cat([mask_special, mask_special, mask_special, key_padding_mask], dim=1)
                
            output, _ = self.transformer(out_embedding, key_padding_mask=key_padding_mask, need_attn_weights=False)

            if self.pooling == 'token':
                complex_embedding = output[:, 0, :].squeeze(1)
            else:
                complex_embedding = (output * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
                if self.pooling == 'mean':
                    # Prot_mask: [N, L]
                    seq_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
                    complex_embedding = complex_embedding / (seq_mask_sum + 1e-10)

            output = self.pred_head(complex_embedding)
            output = output.squeeze(1)
        
        return output

        
        
            
            
