"""SAM3 pseudo label 局部对齐训练组件"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class Sam3PseudoPhraseDataset(Dataset):
    """phrase-level pseudo label 数据集"""

    def __init__(self, pseudo_pth, max_samples=None):
        data = torch.load(pseudo_pth, map_location="cpu")
        self.meta = data.get("meta", {})
        self.images = {image["id"]: image for image in data["images"]}
        self.samples = data["phrase_samples"]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.images[sample["image_id"]]

        return {
            "patch_tokens": image["patch_tokens"].float(),
            "phrase_clip_feature": sample["phrase_clip_feature"].float(),
            "mask_grid": sample["mask_grid"].float(),
            "metadata": {
                "sample_id": sample["id"],
                "image_id": sample["image_id"],
                "annotation_id": sample["annotation_id"],
                "phrase": sample["phrase"],
            },
        }


class LocalAdapterProjector(nn.Module):
    """冻结旧 projector，只训练 residual adapter"""

    def __init__(
        self,
        frozen_projector,
        clip_embed_dim=512,
        dino_embed_dim=768,
        hidden_dim=None,
        alpha=0.1,
        zero_init=True,
    ):
        super().__init__()
        self.frozen_projector = frozen_projector
        for param in self.frozen_projector.parameters():
            param.requires_grad = False

        hidden_dim = hidden_dim or dino_embed_dim
        self.adapter = nn.Sequential(
            nn.Linear(clip_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dino_embed_dim),
        )
        self.alpha = alpha

        if zero_init:
            # 初始时 z_local 近似等于旧 projector 输出
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, phrase_clip_feature):
        with torch.no_grad():
            z_base = self.frozen_projector.project_clip_txt(phrase_clip_feature)

        z_delta = self.adapter(phrase_clip_feature.float())
        z_local = z_base + self.alpha * z_delta
        return z_local, z_base


def dice_loss_with_logits(logits, targets, eps=1e-6):
    """二值 Dice loss，适合稀疏 mask"""
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def local_mask_loss(
    z_local,
    z_base,
    patch_tokens,
    mask_grid,
    temperature=10.0,
    dice_weight=0.5,
    keep_weight=0.05,
):
    """计算局部 mask loss 和 keep-close 约束"""
    batch_size = patch_tokens.shape[0]
    patch_tokens = F.normalize(patch_tokens.float(), p=2, dim=-1)
    z_local = F.normalize(z_local.float(), p=2, dim=-1)
    z_base = F.normalize(z_base.float(), p=2, dim=-1)

    logits = temperature * torch.einsum("bd,bnd->bn", z_local, patch_tokens)
    targets = mask_grid.view(batch_size, -1).float()

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_with_logits(logits, targets)
    keep = 1.0 - F.cosine_similarity(z_local, z_base, dim=-1).mean()
    loss = bce + dice_weight * dice + keep_weight * keep

    return loss, {
        "loss": loss.detach(),
        "bce": bce.detach(),
        "dice": dice.detach(),
        "keep": keep.detach(),
    }
