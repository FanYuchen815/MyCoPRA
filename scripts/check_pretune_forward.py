import torch
from models.components.ssdn import SSDNEnhanced
import torch.nn.functional as F

# Simulate minimal environment and verify pretune token-pooling handling

def run_check():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    L = 6
    embed_dim = 32
    pair_dim = 16

    # build out_embedding that simulates token pooling: prepend 3 special tokens + L tokens
    # total tokens = 3 + L
    total_L = 3 + L
    out_embedding = torch.randn(B, total_L, embed_dim, device=device)
    # make seq tokens correspond to last L positions
    # z: pairwise features shape [B, total_L, total_L, pair_dim]
    z = torch.randn(B, total_L, total_L, pair_dim, device=device)
    key_padding_mask = torch.zeros(B, total_L, dtype=torch.bool, device=device)

    model = SSDNEnhanced(embed_dim=embed_dim, pair_dim=pair_dim, num_layers=2, num_heads=4, cross_heads_start=2, dropout=0.0).to(device)
    model.eval()

    # call as in pretune branch
    cformer_res = model(out_embedding, z, key_padding_mask=~key_padding_mask, need_attn_weights=False, attn_mask=None)
    print('cformer_res types and shapes:')
    for i, r in enumerate(cformer_res):
        print(i, type(r), getattr(r, 'shape', None))

    # normalize like in model.forward pretune branch
    try:
        a0, a1, a2 = cformer_res
        if a0.dim() == 2:
            output = a1
            pair_out = a2
        else:
            output = a0
            pair_out = a1
    except Exception:
        output = cformer_res[0]
        pair_out = cformer_res[1]

    print('Normalized output shape:', output.shape)
    # token pooling expectations
    if output.dim() == 3 and output.shape[1] >= 3:
        complex_embedding = output[:, 0, :].squeeze(1)
        prot_embedding = output[:, 1, :].squeeze(1)
        rna_embedding = output[:, 2, :].squeeze(1)
        print('complex_embedding shape:', complex_embedding.shape)
        print('prot_embedding shape:', prot_embedding.shape)
        print('rna_embedding shape:', rna_embedding.shape)
    else:
        print('Output not in token format; skipping token pooling checks')

if __name__ == '__main__':
    run_check()
