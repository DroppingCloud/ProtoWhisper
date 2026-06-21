"""项目级常量配置和模型加载工具"""

import os
from pathlib import Path

import clip
import torch


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 显式指定本地缓存路径
# -------------------------------

# DINOv2 本地 torch hub 仓库
DINO_V2_LOCAL_REPO = Path(
    os.environ.get(
        "DINO_V2_LOCAL_REPO",
        "/root/.cache/torch/hub/facebookresearch_dinov2_main",
    )
)

# DINOv2 权重所在的 torch hub 目录
# DINOv2 的 .pth 权重会从 torch.hub.get_dir()/checkpoints 下查找
DINO_TORCH_HUB_DIR = Path(
    os.environ.get(
        "DINO_TORCH_HUB_DIR",
        "/root/.cache/torch/hub",
    )
)

DINO_V2_REMOTE_REPO = os.environ.get(
    "DINO_V2_REMOTE_REPO",
    "facebookresearch/dinov2",
)

# CLIP 权重目录
# 你的 ViT-B-16.pt 在 /root/.cache/clip/ 下，所以这里不要写成 /root/.cache/torch/clip
CLIP_CACHE_DIR = Path(
    os.environ.get(
        "CLIP_CACHE_DIR",
        "/root/.cache/clip",
    )
)


def load_dinov2_model(model_name, device=DEFAULT_DEVICE):
    """优先从本地 DINOv2 repo 和本地 checkpoint 加载模型"""

    # 关键：让 torch hub 从 /root/.cache/torch/hub/checkpoints 找 DINO 权重
    torch.hub.set_dir(str(DINO_TORCH_HUB_DIR))

    model = None

    if DINO_V2_LOCAL_REPO.exists():
        print(f"Loading DINOv2 repo from local cache: {DINO_V2_LOCAL_REPO}")
        print(f"Using torch hub dir: {torch.hub.get_dir()}")
        try:
            model = torch.hub.load(
                str(DINO_V2_LOCAL_REPO),
                model_name,
                source="local",
            )
        except Exception as exc:
            print(f"Local DINOv2 load failed: {exc}")

    if model is None:
        print(f"Downloading DINOv2 from {DINO_V2_REMOTE_REPO}")
        model = torch.hub.load(DINO_V2_REMOTE_REPO, model_name)

    model.eval()
    model.to(device)
    return model


def load_clip_model(model_name, device=DEFAULT_DEVICE, jit=False):
    """优先从本地 CLIP_CACHE_DIR 加载 CLIP 模型权重"""

    CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP from cache dir: {CLIP_CACHE_DIR}")

    model, preprocess = clip.load(
        model_name,
        device=device,
        jit=jit,
        download_root=str(CLIP_CACHE_DIR),
    )

    model.eval()
    return model, preprocess