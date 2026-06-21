"""
从 Flickr8k 标注中提取 CLIP 文本特征，并与 DINO 视觉特征合并为训练可用的 .pth 文件

python flickr8k_text_extraction.py \\
    --data_dir path/to/Flickr8k \\
    --dino_pth path/to/flickr8k_dino_train.pth \\
    --split train \\
    --clip_model ViT-B/32
"""

import argparse
import math
import os
import sys
import warnings

import clip
import torch
from tqdm import tqdm

from src.hooks import feats, get_all_out_tokens, get_clip_second_last_dense_out

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
def log_kv(k, v):     print(f"    {_DIM}{k:<22}{_RESET} {v}",      flush=True)


# ---------------------------------------------------------------------------
# Flickr8k caption 加载
# ---------------------------------------------------------------------------

def load_flickr8k_captions(data_dir, split, image_name_to_id):
    """
    读取 Flickr8k.token.txt，过滤出 split 内的图像，
    并用 image_name_to_id 为每条 annotation 分配与 DINO 脚本完全一致的 image_id

    Flickr8k caption 格式（Flickr8k.token.txt）：
        image_file_name#caption_index\\tcaption_text
    每张图片有 5 条 caption，索引为 0~4

    参数:
        data_dir:          Flickr8k 根目录
        split:             'train' | 'dev' | 'test' | 'all'
        image_name_to_id:  由 DINO 脚本生成的 file_name → image_id 映射
                           （从 dino_pth["image_name_to_id"] 读取）

    返回:
        annotations: list of dict，每条形如：
            {
                "id":            int,   # annotation 顺序编号
                "image_id":      int,   # 与 images[i]["id"] 对应，与 DINO 脚本对齐
                "image_name":    str,   # 图像文件名（供调试）
                "caption_index": int,   # 0~4
                "caption":       str,   # 原始 caption 文本
            }
    """
    split_map = {
        "train": "Flickr_8k.trainImages.txt",
        "dev":   "Flickr_8k.devImages.txt",
        "test":  "Flickr_8k.testImages.txt",
    }

    # 确定该 split 下的图像集合
    if split == "all":
        split_images = set()
        for fname in split_map.values():
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        split_images.add(line)
    else:
        if split not in split_map:
            raise ValueError(f"split 必须是 'train'/'dev'/'test'/'all'，收到: {split!r}")
        fpath = os.path.join(data_dir, split_map[split])
        with open(fpath, "r") as f:
            split_images = {line.strip() for line in f if line.strip()}

    token_file = os.path.join(data_dir, "Flickr8k.token.txt")
    annotations = []
    ann_id = 0
    skipped_no_dino = 0

    with open(token_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式：image_name#cap_idx\tcaption
            try:
                key, caption = line.split("\t", 1)
                img_name, cap_idx = key.rsplit("#", 1)
                cap_idx = int(cap_idx)
            except ValueError:
                log_warning(f"跳过无法解析的行：{line[:80]}")
                continue

            if img_name not in split_images:
                continue

            # 用 DINO 脚本的映射获取 image_id，保证两者完全对齐
            if img_name not in image_name_to_id:
                skipped_no_dino += 1
                continue

            annotations.append({
                "id":            ann_id,
                "image_id":      image_name_to_id[img_name],
                "image_name":    img_name,
                "caption_index": cap_idx,
                "caption":       caption,
            })
            ann_id += 1

    if skipped_no_dino:
        log_warning(
            f"{skipped_no_dino} captions skipped：对应图像不在 dino_pth image_name_to_id 中"
        )

    return annotations


# ---------------------------------------------------------------------------
# CLIP 文本特征提取辅助函数
# ---------------------------------------------------------------------------

def register_text_hooks(model, extract_dense_out, extract_second_last_dense_out):
    """
    按需注册 CLIP 文本编码器的 forward hook

    参数:
        model:                         CLIP 模型
        extract_dense_out:             是否提取最后一层 token-level 特征
        extract_second_last_dense_out: 是否提取倒数第二层 token-level 特征
    """
    if extract_dense_out:
        model.ln_final.register_forward_hook(get_all_out_tokens)
    if extract_second_last_dense_out:
        model.transformer.resblocks[-2].register_forward_hook(get_clip_second_last_dense_out)


def get_batch_bounds(batch_index, batch_size, total_items, total_batches):
    """计算当前 batch 在 annotations 列表中的起止下标 [start, end)"""
    start = batch_index * batch_size
    end = start + batch_size if batch_index < total_batches - 1 else total_items
    return start, end


def extract_text_batch(model, texts, extract_dense_out, device):
    """
    对一个 batch 的 caption 文本提取 CLIP 文本特征

    返回:
        inputs:         tokenized 文本，shape [B, 77]
        batch_features: 特征字典，包含 ann_feats 及可选的 token-level 特征
    """
    # CLIP 最大文本长度 77，truncate=True 表示超长文本会被截断
    inputs = clip.tokenize(texts, truncate=True).to(device)

    with torch.no_grad():
        outputs = model.encode_text(inputs)
        batch_features = {"ann_feats": outputs}

        if extract_dense_out:
            # hook 捕获 ln_final 的 token-level 输出，再投影到 CLIP 特征空间
            batch_features["clip_txt_out_tokens"] = (
                feats["clip_txt_out_tokens"] @ model.text_projection
            )
            # inputs > 0 标记有效 token（padding token 通常为 0）
            batch_features["text_input_mask"] = inputs > 0

    return inputs, batch_features


def write_text_features(
    annotations,
    start,
    end,
    inputs,
    batch_features,
    extract_dense_out,
    extract_second_last_dense_out,
):
    """
    将当前 batch 提取到的文本特征写回 annotations 列表对应条目

    写回字段:
        annotation["ann_feats"]            — 句子级 CLIP 文本特征
        annotation["clip_txt_out_tokens"]  — 最后一层 token-level 特征（可选）
        annotation["text_input_mask"]      — 有效 token mask（可选）
        annotation["clip_second_last_out"] — 倒数第二层 token-level 特征（可选）
        annotation["text_argmax"]          — token id 最大位置（EOT 定位）
    """
    for annotation_index in range(start, end):
        batch_index = annotation_index - start
        annotation = annotations[annotation_index]

        annotation["ann_feats"] = batch_features["ann_feats"][batch_index].to("cpu")

        if extract_dense_out:
            annotation["clip_txt_out_tokens"] = (
                batch_features["clip_txt_out_tokens"][batch_index].to("cpu")
            )
            annotation["text_input_mask"] = (
                batch_features["text_input_mask"][batch_index].to("cpu")
            )

        if extract_second_last_dense_out:
            annotation["clip_second_last_out"] = (
                feats["clip_second_last_out"][batch_index].to("cpu")
            )

        # text_argmax：EOT token 通常具有最大 token id，用于定位句子级特征
        annotation["text_argmax"] = inputs[batch_index].argmax().item()


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------

def run_flickr8k_text_extraction(
    data_dir,
    dino_pth,
    split,
    clip_model_name="ViT-B/32",
    batch_size=256,
    out_path=None,
    extract_dense_out=False,
    extract_second_last_dense_out=False,
):
    """
    提取 CLIP 文本特征并写入 DINO .pth，生成 DinoClipDataset 可直接读取的合并文件

    参数:
        data_dir:                      Flickr8k 根目录（含 split txt 和 token.txt）
        dino_pth:                      flickr8k_dino_extraction.py 的输出路径
        split:                         数据划分，须与 dino_pth 一致
        clip_model_name:               CLIP 模型名称（如 ViT-B/32）
        batch_size:                    文本提取的 batch 大小
        out_path:                      输出路径，默认覆盖 dino_pth（原地合并）
        extract_dense_out:             是否提取最后一层 token-level 文本特征
        extract_second_last_dense_out: 是否提取倒数第二层 token-level 文本特征
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 DINO .pth（含 images 和 image_name_to_id）
    log_step("Loading DINO features pth")
    log_kv("dino_pth", dino_pth)
    data = torch.load(dino_pth, map_location="cpu")

    if "image_name_to_id" not in data:
        raise KeyError(
            "'image_name_to_id' 不在 dino_pth 中"
            "请用最新版 flickr8k_dino_extraction.py 重新生成"
        )

    # 校验 split 一致性，防止 dino_pth 和 caption split 错配导致大量 annotation 被跳过
    pth_split = data.get("split")
    if pth_split is not None and pth_split != split:
        raise ValueError(
            f"split 不一致：dino_pth 的 split='{pth_split}'，"
            f"但命令行传入 --split='{split}'"
            "请确保两个脚本使用相同的 split"
        )
    if pth_split is None:
        log_warning("dino_pth 中不含 'split' 字段，无法自动校验 split 一致性")

    image_name_to_id = data["image_name_to_id"]
    log_success(f"Loaded {len(data['images'])} images with DINO features")

    # 2. 加载 CLIP 模型
    log_step("Loading CLIP model")
    log_kv("clip_model", clip_model_name)
    model, _ = clip.load(clip_model_name, device=device)
    model.eval()
    log_success(f"CLIP is ready on {device}")

    register_text_hooks(model, extract_dense_out, extract_second_last_dense_out)

    # 3. 加载 caption 数据，用 image_name_to_id 对齐 image_id
    log_step(f"Loading Flickr8k captions: split={split}")
    annotations = load_flickr8k_captions(data_dir, split, image_name_to_id)
    log_success(f"Loaded {len(annotations)} captions")

    # 4. 提取文本特征
    log_step("Starting CLIP text feature extraction")
    log_kv("annotations",  len(annotations))
    log_kv("batch_size",   batch_size)
    log_kv("features", ", ".join(
        name for name, enabled in [
            ("ann_feats (sentence)",    True),
            ("clip_txt_out_tokens",     extract_dense_out),
            ("clip_second_last_out",    extract_second_last_dense_out),
        ] if enabled
    ))

    total_batches = math.ceil(len(annotations) / batch_size)

    for batch_index in tqdm(
        range(total_batches),
        desc="Extracting CLIP text features",
        dynamic_ncols=True,
        colour="green" if _COLOR_ENABLED else None,
    ):
        start, end = get_batch_bounds(batch_index, batch_size, len(annotations), total_batches)
        texts = [ann["caption"] for ann in annotations[start:end]]

        inputs, batch_features = extract_text_batch(model, texts, extract_dense_out, device)

        write_text_features(
            annotations,
            start,
            end,
            inputs,
            batch_features,
            extract_dense_out,
            extract_second_last_dense_out,
        )

    log_success("Text feature extraction finished")

    # 5. 将 annotations 写入 data 并保存（合并输出）
    data["annotations"] = annotations

    if out_path is None:
        out_path = dino_pth  # 默认原地合并覆盖 DINO pth

    torch.save(data, out_path)
    log_success(f"Combined features saved at {out_path}")
    log_info(
        "data['images'] contains DINO features; "
        "data['annotations'] contains CLIP text features. "
        "Ready for DinoClipDataset."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    parser = argparse.ArgumentParser(
        description="从 Flickr8k 提取 CLIP 文本特征，并与 DINO 视觉特征合并",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入输出
    parser.add_argument("--data_dir",  type=str, required=True,
        help="Flickr8k 根目录，须含 Flickr_8k.{train/dev/test}Images.txt 和 Flickr8k.token.txt")
    parser.add_argument("--dino_pth",  type=str, required=True,
        help="flickr8k_dino_extraction.py 的输出 .pth，将在其基础上写入文本特征")
    parser.add_argument("--split",     type=str, default="train",
        choices=["train", "dev", "test", "all"],
        help="数据划分，须与 dino_pth 生成时使用的 split 一致")
    parser.add_argument("--out_path",  type=str, default=None,
        help="输出 .pth 路径默认覆盖 dino_pth（原地合并）"
             "如需保留原始 DINO pth，请指定不同路径")

    # 模型
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
        help="CLIP 模型名称，如 ViT-B/32、ViT-L/14")

    # 批处理
    parser.add_argument("--batch_size", type=int, default=256,
        help="文本特征提取的 batch 大小")

    # 特征开关
    parser.add_argument("--extract_dense_out", default=False, action="store_true",
        help="是否提取 CLIP 最后一层 token-level 文本特征（clip_txt_out_tokens）")
    parser.add_argument("--extract_second_last_dense_out", default=False, action="store_true",
        help="是否提取 CLIP 倒数第二层 Transformer block 的 token-level 输出（clip_second_last_out）")

    return parser


def main():
    args = build_argparser().parse_args()

    run_flickr8k_text_extraction(
        data_dir=args.data_dir,
        dino_pth=args.dino_pth,
        split=args.split,
        clip_model_name=args.clip_model,
        batch_size=args.batch_size,
        out_path=args.out_path,
        extract_dense_out=args.extract_dense_out,
        extract_second_last_dense_out=args.extract_second_last_dense_out,
    )


if __name__ == "__main__":
    main()
