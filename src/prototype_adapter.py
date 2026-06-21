"""Region-prototype adapter losses for SAM3 pseudo supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTextPrototypeAdapter(nn.Module):
    """Small residual adapter on top of a frozen text projection."""

    def __init__(self, clip_dim=512, dino_dim=768, hidden_dim=768, beta=0.02, zero_init=True):
        super().__init__()
        self.clip_dim = int(clip_dim)
        self.dino_dim = int(dino_dim)
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.adapter = nn.Sequential(
            nn.Linear(self.clip_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dino_dim),
        )
        if zero_init:
            # 初始输出等价于冻结 projector，避免一开始破坏全局语义空间
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, clip_feature, base_text_emb):
        delta = self.adapter(clip_feature.float())
        z = base_text_emb.float() + self.beta * delta
        return F.normalize(z, dim=-1)


def masked_region_prototypes(patch_tokens, mask_flat, eps=1e-6):
    """从 SAM3 mask 内外分别提取 DINO patch prototype"""
    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    mask_flat = mask_flat.float()
    if mask_flat.dim() != 2:
        raise ValueError(f"mask_flat should be [B, N], got {tuple(mask_flat.shape)}")
    if patch_tokens.shape[:2] != mask_flat.shape:
        raise ValueError(
            f"patch/mask shape mismatch: {tuple(patch_tokens.shape[:2])} vs {tuple(mask_flat.shape)}"
        )

    pos_count = mask_flat.sum(dim=1, keepdim=True)
    neg_mask = 1.0 - mask_flat
    neg_count = neg_mask.sum(dim=1, keepdim=True)
    valid = (pos_count[:, 0] > 0) & (neg_count[:, 0] > 0)

    pos = (patch_tokens * mask_flat.unsqueeze(-1)).sum(dim=1) / pos_count.clamp_min(eps)
    neg = (patch_tokens * neg_mask.unsqueeze(-1)).sum(dim=1) / neg_count.clamp_min(eps)
    return F.normalize(pos, dim=-1), F.normalize(neg, dim=-1), valid


def prototype_contrastive_loss(
    text_emb,
    base_text_emb,
    pos_proto,
    neg_proto,
    sample_weight=None,
    temperature=0.07,
    neg_margin=0.2,
    neg_weight=0.5,
    anchor_weight=10.0,
):
    """InfoNCE + negative margin + keep-close anchor loss."""
    text_emb = F.normalize(text_emb.float(), dim=-1)
    base_text_emb = F.normalize(base_text_emb.float(), dim=-1)
    pos_proto = F.normalize(pos_proto.float(), dim=-1)
    neg_proto = F.normalize(neg_proto.float(), dim=-1)

    logits = text_emb @ pos_proto.t()
    labels = torch.arange(text_emb.shape[0], device=text_emb.device)
    if sample_weight is None:
        weight = torch.ones(text_emb.shape[0], device=text_emb.device)
    else:
        weight = sample_weight.float().to(text_emb.device)
        # 只改变样本相对贡献，不隐式改变整体 loss 尺度
        weight = weight / weight.mean().detach().clamp_min(1e-6)

    def weighted_mean(values):
        return (values * weight).sum() / weight.sum().clamp_min(1e-6)

    nce_per_sample = F.cross_entropy(logits / temperature, labels, reduction="none")
    nce = weighted_mean(nce_per_sample)

    neg_sim = (text_emb * neg_proto).sum(dim=-1)
    neg_penalty = weighted_mean(F.relu(neg_sim - neg_margin).pow(2))
    anchor = weighted_mean(1.0 - F.cosine_similarity(text_emb, base_text_emb, dim=-1))

    loss = nce + neg_weight * neg_penalty + anchor_weight * anchor
    return loss, {
        "loss": loss.detach(),
        "nce": nce.detach(),
        "neg_margin": neg_penalty.detach(),
        "anchor": anchor.detach(),
        "sample_weight": weight.mean().detach(),
    }
