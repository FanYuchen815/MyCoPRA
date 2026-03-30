import torch
import torch.nn.functional as F


def modality_alignment_loss(protein_interface, rna_interface, interface_distances, temperature=1.0):
    """通过界面距离监督实现模态对齐（无对比学习）"""
    similarity = F.cosine_similarity(
        protein_interface.unsqueeze(2),  # [B, Lp, 1, E]
        rna_interface.unsqueeze(1),      # [B, 1, Lr, E]
        dim=-1
    )  # [B, Lp, Lr]

    target_similarity = torch.exp(-interface_distances / temperature)

    loss = F.mse_loss(similarity, target_similarity)
    loss.requires_grad_(True)
    return loss


def apply_global_perturbation(atom_coords, perturbation_scale=0.1):
    noise = torch.randn_like(atom_coords) * perturbation_scale
    return atom_coords + noise


def global_causal_loss(original_pred, perturbed_pred, affinity):
    delta_pred = perturbed_pred - original_pred
    # 用 sigmoid(binary)目标来学习扰动方向，保持可导
    direction_pred = torch.tanh(delta_pred)
    # 简化目标：相对亲和力变化期望方向（示例实现）
    direction_target = -torch.sign(affinity)
    loss = F.mse_loss(direction_pred, direction_target)
    loss.requires_grad_(True)
    return loss


def mask_interface_sites(seq_emb, interface_mask, mask_ratio=0.15):
    masked_emb = seq_emb.clone()
    mask = (torch.rand_like(interface_mask.float()) < mask_ratio) & interface_mask
    # mask shape [B, L]; expand to embed dim
    mask_exp = mask.unsqueeze(-1).expand_as(seq_emb)
    masked_emb[mask_exp] = 0.0
    return masked_emb


def local_causal_loss(original_pred, masked_pred, site_importance):
    delta = torch.abs(original_pred - masked_pred)
    # site_importance expected shape compatible with delta (simplified to mean)
    if site_importance.dim() > delta.dim():
        site_importance = site_importance.mean(dim=-1, keepdim=True)
    loss = F.mse_loss(delta, site_importance)
    loss.requires_grad_(True)
    return loss


def atomic_causal_loss(distance_matrix, affinity, temperature=1.0):
    bin_edges = torch.tensor([0, 2, 4, 6, 8, 10, 15, 20], device=distance_matrix.device, dtype=distance_matrix.dtype)
    bins = torch.bucketize(distance_matrix, bin_edges)

    # compute influence per sample by summing exp(-d/temperature) per bin
    influence = torch.zeros_like(affinity)
    # For safety handle batch dimension
    if affinity.dim() == 1:
        affinity = affinity.unsqueeze(-1)
    for b in range(1, len(bin_edges)):
        mask = (bins == b)
        if mask.any():
            # sum over masked distances per batch
            # mask: [B, Lp, Lr] -> compute per-batch sums
            bin_vals = torch.where(mask, torch.exp(-distance_matrix / temperature), torch.zeros_like(distance_matrix))
            per_sample = bin_vals.view(distance_matrix.size(0), -1).sum(dim=-1, keepdim=True)
            influence = influence + per_sample

    loss = F.mse_loss(affinity, influence)
    loss.requires_grad_(True)
    return loss
