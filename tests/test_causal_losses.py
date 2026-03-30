import torch
from models.components import causal_loss


def test_modality_alignment_loss():
    B, Lp, Lr, E = 2, 8, 6, 16
    protein = torch.randn(B, Lp, E)
    rna = torch.randn(B, Lr, E)
    dmat = torch.rand(B, Lp, Lr) * 10.0
    loss = causal_loss.modality_alignment_loss(protein, rna, dmat)
    assert loss.requires_grad


def test_global_and_local():
    original = torch.randn(2, 1)
    perturbed = original + 0.1
    affinity = torch.randn(2, 1)
    loss_g = causal_loss.global_causal_loss(original, perturbed, affinity)
    assert loss_g.requires_grad

    seq_emb = torch.randn(2, 32, 16)
    interface_mask = torch.ones(2, 32).bool()
    masked = causal_loss.mask_interface_sites(seq_emb, interface_mask, 0.2)
    assert masked.shape == seq_emb.shape
