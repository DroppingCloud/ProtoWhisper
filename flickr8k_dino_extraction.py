"""
从 Flickr8k 图像数据中提取 DINOv2 视觉特征的命令行入口

python flickr8k_dino_extraction.py \\
    --data_dir path/to/Flickr8k \\
    --images_dir path/to/Flickr8k/Flicker8k_Dataset \\
    --split train \\
    --model dinov2_vitb14_reg \\
    --extract_cls
"""

import argparse
import os
import sys
import warnings

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.hooks import feats, get_self_attention, process_self_attention

# -------------------------------
# 终端输出与 warning 控制
# -------------------------------
warnings.filterwarnings("ignore", message=r".*xFormers is not available.*", category=UserWarning)

_COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_RESET  = "\033[0m"  if _COLOR_ENABLED else ""
_BOLD   = "\033[1m"  if _COLOR_ENABLED else ""
_DIM    = "\033[2m"  if _COLOR_ENABLED else ""
_CYAN   = "\033[36m" if _COLOR_ENABLED else ""
_GREEN  = "\033[32m" if _COLOR_ENABLED else ""
_YELLOW = "\033[33m" if _COLOR_ENABLED else ""
_RED    = "\033[31m" if _COLOR_ENABLED else ""
_BLUE   = "\033[34m" if _COLOR_ENABLED else ""


def _fmt(color, text, bold=False):
    prefix = (_BOLD if bold else "") + color
    return f"{prefix}{text}{_RESET}"

def log_step(msg):    print(_fmt(_CYAN,   f"\n▶ {msg}", bold=True), flush=True)
def log_info(msg):    print(_fmt(_BLUE,   f"  • {msg}"),            flush=True)
def log_success(msg): print(_fmt(_GREEN,  f"  ✓ {msg}"),            flush=True)
def log_warning(msg): print(_fmt(_YELLOW, f"  ⚠ {msg}"),            flush=True)
def log_error(msg):   print(_fmt(_RED,    f"  ✗ {msg}"),            flush=True)
def log_kv(k, v):     print(f"    {_DIM}{k:<18}{_RESET} {v}",      flush=True)


DINO_HUB_PATH = "/root/.cache/torch/hub/facebookresearch_dinov2_main"


# ---------------------------------------------------------------------------
# Flickr8k 数据加载
# ---------------------------------------------------------------------------

def load_flickr8k_image_names(data_dir, split):
    """
    读取划分文件，返回去重且顺序确定的图像文件名列表

    Flickr8k 划分文件：
        Flickr_8k.trainImages.txt / Flickr_8k.devImages.txt / Flickr_8k.testImages.txt
    每行一个文件名，如 1000268201_693b08cb0e.jpg

    参数:
        data_dir: Flickr8k 根目录（含划分 .txt 文件）
        split:    'train' | 'dev' | 'test' | 'all'

    返回:
        image_names: 有序不重复的图像文件名列表
    """
    split_map = {
        "train": "Flickr_8k.trainImages.txt",
        "dev":   "Flickr_8k.devImages.txt",
        "test":  "Flickr_8k.testImages.txt",
    }

    def _read(fname):
        with open(os.path.join(data_dir, fname), "r") as f:
            return [l.strip() for l in f if l.strip()]

    if split == "all":
        seen, names = set(), []
        for fname in split_map.values():
            for n in _read(fname):
                if n not in seen:
                    seen.add(n)
                    names.append(n)
        return names

    if split not in split_map:
        raise ValueError(f"split 必须是 'train'/'dev'/'test'/'all'，收到: {split!r}")
    return _read(split_map[split])


# ---------------------------------------------------------------------------
# Dataset & transforms
# ---------------------------------------------------------------------------

class Flickr8kImageDataset(Dataset):
    """用于 DINOv2 特征提取的 Flickr8k 图像数据集"""

    def __init__(self, images_meta, images_dir, transform):
        self.images_meta = images_meta
        self.images_dir  = images_dir
        self.transform   = transform

    def __len__(self):
        return len(self.images_meta)

    def __getitem__(self, idx):
        file_name = self.images_meta[idx]["file_name"]
        img_path  = os.path.join(self.images_dir, file_name)
        try:
            with Image.open(img_path) as pil_img:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                return self.transform(pil_img), idx, True
        except Exception:
            return self.transform(Image.new("RGB", (224, 224))), idx, False


def build_image_transforms(resize_dim, crop_dim):
    return T.Compose([
        T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_dim),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_dinov2_model(model_name, device):
    log_step("Loading DINOv2 model")
    log_kv("model", model_name)
    log_kv("repo",  DINO_HUB_PATH)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*xFormers is not available.*", category=UserWarning)
        model = torch.hub.load(DINO_HUB_PATH, model_name, source="local")
    model.eval()
    model.to(device)
    log_success(f"DINOv2 is ready on {device}")
    return model


def needs_attention_features(avg, maps, disentangled):
    return avg or maps or disentangled


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_batch_features(
    model,
    batch_imgs,
    batch_size,
    num_tokens,
    num_attn_heads,
    embed_dim,       # 从 model.embed_dim 动态获取，非硬编码
    scale,
    num_global_tokens,
    extract_avg_self_attn,
    extract_patch_tokens,
    extract_self_attn_maps,
    extract_disentangled_self_attn,
):
    outputs = model(batch_imgs, is_training=True)
    batch_features = {"dino_features": outputs["x_norm_clstoken"]}

    if needs_attention_features(extract_avg_self_attn, extract_self_attn_maps, extract_disentangled_self_attn):
        self_attn, self_attn_maps = process_self_attention(
            feats["self_attn"],
            batch_size,
            num_tokens,
            num_attn_heads,
            embed_dim,
            scale,
            num_global_tokens,
            ret_self_attn_maps=True,
        )
        batch_features["self_attn_maps"] = self_attn_maps

        if extract_avg_self_attn:
            batch_features["avg_self_attn_out"] = (
                self_attn.unsqueeze(-1) * outputs["x_norm_patchtokens"]
            ).mean(dim=1)

        if extract_disentangled_self_attn:
            sa_maps = self_attn_maps.softmax(dim=-1)
            batch_features["disentangled_self_attn"] = (
                outputs["x_norm_patchtokens"].unsqueeze(1) * sa_maps.unsqueeze(-1)
            ).mean(dim=2)

    if extract_patch_tokens:
        batch_features["patch_tokens"] = outputs["x_norm_patchtokens"]

    return batch_features


def write_batch_features(
    data,
    batch_features,
    indices,
    valid_masks,
    extract_cls,
    extract_avg_self_attn,
    extract_patch_tokens,
    extract_self_attn_maps,
    extract_disentangled_self_attn,
):
    n_errors = 0
    for batch_index, (image_index, valid) in enumerate(zip(indices, valid_masks)):
        image_index = image_index.item()
        if not valid:
            n_errors += 1
            continue
        img = data["images"][image_index]
        if extract_cls or (not extract_avg_self_attn):
            img["dino_features"] = batch_features["dino_features"][batch_index].cpu()
        if extract_avg_self_attn:
            img["avg_self_attn_out"] = batch_features["avg_self_attn_out"][batch_index].cpu()
        if extract_patch_tokens:
            img["patch_tokens"] = batch_features["patch_tokens"][batch_index].cpu()
        if extract_self_attn_maps:
            img["self_attn_maps"] = batch_features["self_attn_maps"][batch_index].cpu()
        if extract_disentangled_self_attn:
            img["disentangled_self_attn"] = batch_features["disentangled_self_attn"][batch_index].cpu()
    return n_errors


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------

def run_flickr8k_dino_extraction(
    data_dir,
    images_dir,
    split,
    model_name,
    batch_size,
    resize_dim=518,
    crop_dim=518,
    out_path=None,
    extract_cls=False,
    extract_avg_self_attn=False,
    extract_patch_tokens=False,
    extract_self_attn_maps=False,
    extract_disentangled_self_attn=False,
    num_workers=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型
    model = load_dinov2_model(model_name, device)

    # 动态获取 embed_dim，避免针对不同 ViT 变体硬编码 768
    embed_dim = model.embed_dim
    num_attn_heads = model.num_heads

    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens  = (crop_dim // 14) * (crop_dim // 14)
    num_tokens        = num_global_tokens + num_patch_tokens
    head_dim          = embed_dim // num_attn_heads
    scale             = head_dim ** -0.5     # 按 head dim 缩放，ViT-B: 64**-0.5 = 0.125

    if needs_attention_features(extract_avg_self_attn, extract_self_attn_maps, extract_disentangled_self_attn):
        model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)

    # 2. 加载图像列表（按 split 文件，保证顺序确定）
    log_step(f"Loading Flickr8k image list: split={split}")
    image_names = load_flickr8k_image_names(data_dir, split)
    log_success(f"Found {len(image_names)} images")

    # 构建 images 元信息，同时保留 file_name → id 映射供后续合并
    images_meta = [{"id": i, "file_name": name} for i, name in enumerate(image_names)]
    image_name_to_id = {name: i for i, name in enumerate(image_names)}

    # data 结构与 DinoClipDataset 兼容：必须同时含 images 和 annotations
    # 此处 annotations 为空列表；合并脚本（或文本提取后）会填充
    data = {
        "split":            split,
        "images":           images_meta,
        "annotations":      [],          # 由文本提取脚本/合并脚本填充
        "image_name_to_id": image_name_to_id,  # 用于 file_name 对齐
    }

    # 3. DataLoader
    transforms = build_image_transforms(resize_dim, crop_dim)
    dataset    = Flickr8kImageDataset(images_meta, images_dir, transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )

    log_step("Starting DINOv2 feature extraction")
    log_kv("images",      len(images_meta))
    log_kv("batch size",  batch_size)
    log_kv("num workers", num_workers)
    log_kv("images_dir",  images_dir)
    log_kv("resize/crop", f"{resize_dim}/{crop_dim}")
    log_kv("embed_dim",   embed_dim)
    log_kv("features", ", ".join(
        name for name, enabled in [
            ("cls",                  extract_cls),
            ("avg_self_attn",        extract_avg_self_attn),
            ("patch_tokens",         extract_patch_tokens),
            ("self_attn_maps",       extract_self_attn_maps),
            ("disentangled_self_attn", extract_disentangled_self_attn),
        ] if enabled
    ) or "dino_features (default)")

    # 4. 提取
    n_errors = 0
    for batch_imgs, indices, valid_masks in tqdm(
        dataloader,
        desc="Extracting DINO features",
        dynamic_ncols=True,
        colour="cyan" if _COLOR_ENABLED else None,
    ):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = extract_batch_features(
                model,
                batch_imgs,
                batch_imgs.shape[0],
                num_tokens,
                num_attn_heads,
                embed_dim,
                scale,
                num_global_tokens,
                extract_avg_self_attn,
                extract_patch_tokens,
                extract_self_attn_maps,
                extract_disentangled_self_attn,
            )
        n_errors += write_batch_features(
            data,
            batch_features,
            indices,
            valid_masks,
            extract_cls,
            extract_avg_self_attn,
            extract_patch_tokens,
            extract_self_attn_maps,
            extract_disentangled_self_attn,
        )

    log_success("Feature extraction finished")
    if n_errors:
        log_warning(f"Failed to extract {n_errors} of {len(images_meta)} images")
    else:
        log_success(f"All {len(images_meta)} images processed successfully")

    # 5. 保存
    if out_path is None:
        out_path = os.path.join(data_dir, f"flickr8k_dino_{split}.pth")
    torch.save(data, out_path)
    log_success(f"Features saved at {out_path}")
    log_info("Run flickr8k_text_extraction.py with --dino_pth pointing to this file to merge text features.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    parser = argparse.ArgumentParser(
        description="从 Flickr8k 提取 DINOv2 视觉特征",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入输出
    parser.add_argument("--data_dir",   type=str, required=True,
        help="Flickr8k 根目录，须含 Flickr_8k.{train/dev/test}Images.txt")
    parser.add_argument("--images_dir", type=str, default=None,
        help="图像文件目录（含 .jpg 文件）默认为 data_dir/Flicker8k_Dataset")
    parser.add_argument("--split",      type=str, default="train",
        choices=["train", "dev", "test", "all"],
        help="数据划分")
    parser.add_argument("--out_path",   type=str, default=None,
        help="输出 .pth 路径默认 data_dir/flickr8k_dino_<split>.pth")

    # 模型
    parser.add_argument("--model",      type=str, default="dinov2_vitb14_reg",
        help="DINOv2 模型名称reg 版本含额外注册词元")
    parser.add_argument("--resize_dim", type=int, default=518)
    parser.add_argument("--crop_dim",   type=int, default=518)

    # 批处理
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # 特征开关
    parser.add_argument("--extract_cls",                   default=False, action="store_true")
    parser.add_argument("--extract_avg_self_attn",         default=False, action="store_true")
    parser.add_argument("--extract_patch_tokens",          default=False, action="store_true")
    parser.add_argument("--extract_self_attn_maps",        default=False, action="store_true")
    parser.add_argument("--extract_disentangled_self_attn",default=False, action="store_true")

    return parser


def main():
    args = build_argparser().parse_args()

    images_dir = args.images_dir or os.path.join(args.data_dir, "Flicker8k_Dataset")

    run_flickr8k_dino_extraction(
        data_dir=args.data_dir,
        images_dir=images_dir,
        split=args.split,
        model_name=args.model,
        batch_size=args.batch_size,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim,
        out_path=args.out_path,
        extract_cls=args.extract_cls,
        extract_avg_self_attn=args.extract_avg_self_attn,
        extract_patch_tokens=args.extract_patch_tokens,
        extract_self_attn_maps=args.extract_self_attn_maps,
        extract_disentangled_self_attn=args.extract_disentangled_self_attn,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
