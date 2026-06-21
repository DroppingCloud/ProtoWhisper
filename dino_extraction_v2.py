"""从 COCO 风格图像数据中提取 DINO 视觉特征"""

import argparse
import json
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
# DINOv2 在未安装 xFormers 时会打印性能提示；不影响正确性，这里默认隐藏
warnings.filterwarnings("ignore", message=r".*xFormers is not available.*", category=UserWarning)

_COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_RESET = "\033[0m" if _COLOR_ENABLED else ""
_BOLD = "\033[1m" if _COLOR_ENABLED else ""
_DIM = "\033[2m" if _COLOR_ENABLED else ""
_CYAN = "\033[36m" if _COLOR_ENABLED else ""
_GREEN = "\033[32m" if _COLOR_ENABLED else ""
_YELLOW = "\033[33m" if _COLOR_ENABLED else ""
_RED = "\033[31m" if _COLOR_ENABLED else ""
_BLUE = "\033[34m" if _COLOR_ENABLED else ""


def _fmt(color, text, bold=False):
    prefix = (_BOLD if bold else "") + color
    return f"{prefix}{text}{_RESET}"


def log_step(message):
    print(_fmt(_CYAN, f"\n▶ {message}", bold=True), flush=True)


def log_info(message):
    print(_fmt(_BLUE, f"  • {message}"), flush=True)


def log_success(message):
    print(_fmt(_GREEN, f"  ✓ {message}"), flush=True)


def log_warning(message):
    print(_fmt(_YELLOW, f"  ⚠ {message}"), flush=True)


def log_error(message):
    print(_fmt(_RED, f"  ✗ {message}"), flush=True)


def log_kv(key, value):
    print(f"    {_DIM}{key:<18}{_RESET} {value}", flush=True)


DINO_HUB_PATH = "/root/.cache/torch/hub/facebookresearch_dinov2_main"


class ImageDataset(Dataset):
    """用于 DINOv2 特征提取的图像数据集"""

    def __init__(self, images_meta, data_dir, transform):
        self.images_meta = images_meta
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_meta)

    def __getitem__(self, idx):
        """返回变换后的图像、原始索引以及该图像是否成功读取"""
        file_name = self.images_meta[idx]["file_name"]

        try:
            pil_img = self._open_coco_image(file_name)
            if pil_img is None:
                return self._invalid_sample(idx)

            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return self.transform(pil_img), idx, True
        except Exception:
            # 异常图片不终止整体提取流程，只累计错误数量
            return self._invalid_sample(idx)

    def _open_coco_image(self, file_name):
        """根据 COCO 文件名中的数据划分信息打开对应图片"""
        if "train" in file_name:
            return Image.open(os.path.join(self.data_dir, f"train2014/{file_name}"))
        if "val" in file_name:
            return Image.open(os.path.join(self.data_dir, f"val2014/{file_name}"))
        if "test" in file_name:
            return Image.open(os.path.join(self.data_dir, f"test2014/{file_name}"))
        return None

    def _invalid_sample(self, idx):
        """用占位图保持 batch 形状稳定，并标记该样本不写回特征"""
        pil_img = Image.new("RGB", (224, 224))
        return self.transform(pil_img), idx, False


def build_image_transforms(resize_dim, crop_dim):
    """构建 DINOv2 输入所需的缩放、裁剪和归一化变换"""
    return T.Compose(
        [
            T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop_dim),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_dinov2_model(model_name, device):
    """从本地 torch hub 缓存加载 DINOv2，避免运行时联网下载"""
    log_step("Loading DINOv2 model")
    log_kv("model", model_name)
    log_kv("repo", DINO_HUB_PATH)
    log_kv("torch hub dir", torch.hub.get_dir())

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*xFormers is not available.*", category=UserWarning)
        model = torch.hub.load(DINO_HUB_PATH, model_name, source="local")

    model.eval()
    model.to(device)
    log_success(f"DINOv2 is ready on {device}")
    return model


def load_annotation_data(ann_path):
    """读取 JSON 或 PTH 格式的 COCO 风格数据"""
    log_step("Loading annotations")
    log_kv("source", ann_path)

    if ann_path.endswith(".json"):
        log_info("Detected COCO JSON annotations")
        with open(ann_path, "r") as f:
            return json.load(f)

    log_info("Detected PTH annotations/features")
    return torch.load(ann_path)


def needs_attention_features(extract_avg_self_attn, extract_self_attn_maps, extract_disentangled_self_attn):
    """判断是否需要钩取注意力层 qkv 输出"""
    return extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn


def extract_batch_features(
    model,
    batch_imgs,
    batch_size,
    num_tokens,
    num_attn_heads,
    embed_dim,
    scale,
    num_global_tokens,
    extract_avg_self_attn,
    extract_patch_tokens,
    extract_self_attn_maps,
    extract_disentangled_self_attn,
):
    """执行一次 DINOv2 前向传播，并按开关返回当前 batch 需要的特征"""
    outputs = model(batch_imgs, is_training=True)
    batch_features = {"dino_features": outputs["x_norm_clstoken"]}

    if needs_attention_features(
        extract_avg_self_attn,
        extract_self_attn_maps,
        extract_disentangled_self_attn,
    ):
        # 将 qkv 钩子输出还原成自注意力图，用于得到局部化图像表征
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
    """根据命令行开关把不同粒度的视觉特征写回 data['images']"""
    n_errors = 0
    for batch_index, (image_index, valid) in enumerate(zip(indices, valid_masks)):
        image_index = image_index.item()
        if not valid:
            n_errors += 1
            continue

        image_data = data["images"][image_index]
        if extract_cls or (not extract_avg_self_attn):
            image_data["dino_features"] = batch_features["dino_features"][batch_index].cpu()
        if extract_avg_self_attn:
            image_data["avg_self_attn_out"] = batch_features["avg_self_attn_out"][batch_index].cpu()
        if extract_patch_tokens:
            image_data["patch_tokens"] = batch_features["patch_tokens"][batch_index].cpu()
        if extract_self_attn_maps:
            image_data["self_attn_maps"] = batch_features["self_attn_maps"][batch_index].cpu()
        if extract_disentangled_self_attn:
            image_data["disentangled_self_attn"] = batch_features["disentangled_self_attn"][batch_index].cpu()

    return n_errors


def save_extraction_output(data, ann_path, out_path):
    """输出单个 .pth 文件，供后续训练读取"""
    if out_path is None:
        out_path = os.path.splitext(ann_path)[0] + ".pth"
    torch.save(data, out_path)
    log_success(f"Features saved at {out_path}")


def run_dinov2_extraction(
    model_name,
    data_dir,
    ann_path,
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
    """批量提取 DINOv2 CLS、图像块词元或自注意力加权特征并写回 COCO 风格数据"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # reg 版本 DINOv2 带额外注册词元，需要从图像块词元数量中区分出来
    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens = crop_dim // 14 * crop_dim // 14
    num_tokens = num_global_tokens + num_patch_tokens
    embed_dim = 768  # ViT-B
    scale = 0.125

    model = load_dinov2_model(model_name, device)
    image_transforms = build_image_transforms(resize_dim, crop_dim)
    num_attn_heads = model.num_heads
    data = load_annotation_data(ann_path)

    if needs_attention_features(
        extract_avg_self_attn,
        extract_self_attn_maps,
        extract_disentangled_self_attn,
    ):
        model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)

    log_step("Starting DINOv2 feature extraction")
    log_kv("images", len(data["images"]))
    log_kv("batch size", batch_size)
    log_kv("num workers", num_workers)
    log_kv("resize/crop", f"{resize_dim}/{crop_dim}")
    log_kv("features", ", ".join(
        name for name, enabled in [
            ("cls", extract_cls),
            ("avg_self_attn", extract_avg_self_attn),
            ("patch_tokens", extract_patch_tokens),
            ("self_attn_maps", extract_self_attn_maps),
            ("disentangled_self_attn", extract_disentangled_self_attn),
        ] if enabled
    ) or "dino_features")

    dataset = ImageDataset(data["images"], data_dir, image_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )

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
    if n_errors > 0:
        log_warning(f"Failed to extract {n_errors} of {len(data['images'])} images")
    else:
        log_success(f"All {len(data['images'])} images processed successfully")

    save_extraction_output(data, ann_path, out_path)


def build_argparser():
    """定义 DINOv2 特征提取脚本的命令行参数"""
    parser = argparse.ArgumentParser()

    # -------------------------------
    # 输入输出相关参数
    # -------------------------------
    parser.add_argument(
        "--ann_path", type=str, default="coco/test1k.json",
        help="标注文件路径，可为 JSON, PTH 或 WebDataset tar用于读取 COCO 风格的图像元信息"
    )
    parser.add_argument(
        "--data_dir", type=str, default="../coco/",
        help="图像根目录，用于根据 ann_path 的 file_name 读取实际图片文件"
    )
    parser.add_argument(
        "--out_path", type=str, default=None,
        help="保存特征输出的文件路径如果为空，则根据 ann_path 自动生成 .pth 文件"
    )

    # -------------------------------
    # 模型相关参数
    # -------------------------------
    parser.add_argument(
        "--model", type=str, default="dinov2_vitb14_reg",
        help="DINOv2 模型名称，如 dinov2_vitb14_reg、dinov2_vitl16 等reg 版本带额外注册词元"
    )
    parser.add_argument(
        "--resize_dim", type=int, default=518,
        help="图像缩放的边长尺寸，先 resize 再 center crop，用于保持 DINO 输入固定大小"
    )
    parser.add_argument(
        "--crop_dim", type=int, default=518,
        help="中心裁剪尺寸，保证输入给 ViT 的 patch token 数量固定"
    )

    # -------------------------------
    # 批量处理参数
    # -------------------------------
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="每次前向传播的 batch 大小大 batch 利用 GPU，但占用更多显存"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="PyTorch DataLoader 的子进程数量，用于并行加载图像数据，提高 I/O 和预处理速度"
    )

    # -------------------------------
    # 特征提取开关
    # -------------------------------
    parser.add_argument(
        "--extract_cls", default=False, action="store_true",
        help="是否提取 CLS token 向量作为图像全局特征"
    )
    parser.add_argument(
        "--extract_avg_self_attn", default=False, action="store_true",
        help="是否提取平均自注意力加权特征（Avg Self-Attention），可用于局部感知的全局特征"
    )
    parser.add_argument(
        "--extract_patch_tokens", default=False, action="store_true",
        help="是否提取所有 patch token，得到完整的局部图像表示（每个 patch 一个向量）"
    )
    parser.add_argument(
        "--extract_self_attn_maps", default=False, action="store_true",
        help="是否提取每层自注意力图，可用于可视化注意力分布或生成掩码"
    )
    parser.add_argument(
        "--extract_disentangled_self_attn", default=False, action="store_true",
        help="是否提取解耦的自注意力特征（disentangled self-attention），按注意力权重聚合 patch token"
    )

    return parser


def main():
    args = build_argparser().parse_args()

    run_dinov2_extraction(
            args.model,
            args.data_dir,
            args.ann_path,
            args.batch_size,
            args.resize_dim,
            args.crop_dim,
            args.out_path,
            args.extract_cls,
            args.extract_avg_self_attn,
            args.extract_patch_tokens,
            args.extract_self_attn_maps,
            args.extract_disentangled_self_attn,
            args.num_workers,
        )


if __name__ == "__main__":
    main()
