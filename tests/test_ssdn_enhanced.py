import torch
from models.components.ssdn import SSDNEnhanced, InteractionTypePerceptor


def test_ssdn_enhanced_forward():
    B, L, E, P = 2, 32, 64, 16
    num_layers = 2
    model = SSDNEnhanced(E, P, num_layers=num_layers)

    seq_emb = torch.randn(B, L, E)
    struct_emb = torch.randn(B, L, L, P)
    atom_coords = torch.randn(B, L, 4, 3)
    residue_mask = torch.ones(B, L).bool()
    chain_mask = torch.ones(B, 2, L).bool()

    fused, seq_out, struct_out = model(seq_emb, struct_emb, atom_coords, residue_mask, chain_mask)

    assert fused.shape == (B, E)
    assert seq_out.shape == (B, L, E)
    assert struct_out.shape == (B, L, L, P)


def test_itp():
    itp = InteractionTypePerceptor(64)
    seq_emb = torch.randn(2, 32, 64)
    struct_emb = torch.randn(2, 32, 32, 16)
    weights = itp(seq_emb, struct_emb)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
