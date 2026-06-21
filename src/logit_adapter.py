"""Logit-level residual adapter for SAM3 pseudo-label supervision."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLogitResidualAdapter(nn.Module):
    """Low-rank residual that calibrates text-patch matching logits.

    It does not modify the frozen text embedding or DINO patch token space.
    Instead, it predicts a bounded residual added to the base cosine simmap.
    """

    def __init__(self, dino_dim=768, rank_dim=128, gamma=0.1, zero_init=True):
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.rank_dim = int(rank_dim)
        self.gamma = float(gamma)

        self.text_proj = nn.Linear(self.dino_dim, self.rank_dim)
        self.patch_proj = nn.Linear(self.dino_dim, self.rank_dim)
        init_scale = 0.0 if zero_init else 1.0
        self.raw_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def _project_text(self, text_emb):
        text_emb = F.normalize(text_emb.float(), dim=-1)
        return F.normalize(self.text_proj(text_emb), dim=-1)

    def _project_patch_tokens(self, patch_tokens):
        patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
        return F.normalize(self.patch_proj(patch_tokens), dim=-1)

    def _bounded_scale(self):
        return self.gamma * torch.tanh(self.raw_scale)

    def forward_pair(self, text_emb, patch_tokens):
        """Phrase-level residual.

        Args:
            text_emb: [B, D]
            patch_tokens: [B, N, D]

        Returns:
            residual logits: [B, N]
        """
        text_q = self._project_text(text_emb)
        patch_k = self._project_patch_tokens(patch_tokens)
        residual = torch.einsum("br,bnr->bn", text_q, patch_k)
        return self._bounded_scale() * residual

    def forward_dense(self, text_emb, image_feat):
        """Dense segmentation residual.

        Args:
            text_emb: [C, D]
            image_feat: [B, D, H, W]

        Returns:
            residual logits: [B, C, H, W]
        """
        b, d, h, w = image_feat.shape
        if d != self.dino_dim:
            raise ValueError(f"image_feat dim mismatch: {d} vs {self.dino_dim}")
        patch_tokens = image_feat.permute(0, 2, 3, 1).reshape(b, h * w, d)
        text_q = self._project_text(text_emb)
        patch_k = self._project_patch_tokens(patch_tokens)
        residual = torch.einsum("cr,bnr->bcn", text_q, patch_k)
        residual = residual.reshape(b, text_emb.shape[0], h, w)
        return self._bounded_scale() * residual


def dice_loss_with_logits(logits, targets, eps=1e-6):
    """Binary Dice loss averaged over phrase samples."""
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def logit_residual_loss(
    base_logits,
    residual_logits,
    targets,
    dice_weight=0.5,
    distill_weight=1.0,
    residual_l2_weight=0.01,
):
    """Train final logits while keeping them close to frozen base logits."""
    targets = targets.float()
    final_logits = base_logits.detach() + residual_logits
    bce = F.binary_cross_entropy_with_logits(final_logits, targets)
    dice = dice_loss_with_logits(final_logits, targets)
    distill = F.mse_loss(final_logits, base_logits.detach())
    residual_l2 = residual_logits.pow(2).mean()
    loss = (
        bce
        + dice_weight * dice
        + distill_weight * distill
        + residual_l2_weight * residual_l2
    )
    return loss, {
        "loss": loss.detach(),
        "bce": bce.detach(),
        "dice": dice.detach(),
        "distill": distill.detach(),
        "residual_l2": residual_l2.detach(),
    }


def hard_iou_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    """Diagnostic hard IoU for pseudo masks."""
    pred = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()
    tp = (pred * targets).sum(dim=1)
    fp = (pred * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - pred) * targets).sum(dim=1)
    return (tp / (tp + fp + fn + eps)).mean()
