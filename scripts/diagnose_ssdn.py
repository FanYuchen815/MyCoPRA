import traceback
import torch
from models.components.ssdn import SSDNEnhanced

def try_run():
    try:
        print('torch version:', torch.__version__)
        print('cuda available:', torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        embed_dim = 32
        pair_dim = 16
        B = 2
        L = 8

        model = SSDNEnhanced(embed_dim=embed_dim, pair_dim=pair_dim, num_layers=2, num_heads=4, cross_heads_start=2, dropout=0.0)
        model = model.to(device)
        model.eval()

        seq_emb = torch.randn(B, L, embed_dim, device=device)
        struct_emb = torch.randn(B, L, L, pair_dim, device=device)

        print('Input shapes -> seq_emb:', seq_emb.shape, 'struct_emb:', struct_emb.shape)

        with torch.no_grad():
            out = model(seq_emb, struct_emb)

        print('Output type:', type(out))
        if isinstance(out, tuple) or isinstance(out, list):
            for i, o in enumerate(out):
                try:
                    print(f'out[{i}] shape:', o.shape)
                except Exception:
                    print(f'out[{i}] repr:', repr(o))
        else:
            try:
                print('out shape:', out.shape)
            except Exception:
                print('out repr:', repr(out))

        print('SSDNEnhanced forward succeeded.')
    except Exception as e:
        print('Exception during SSDNEnhanced diagnostic:')
        traceback.print_exc()

if __name__ == '__main__':
    try_run()
