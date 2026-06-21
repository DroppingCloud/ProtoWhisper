"""
为 Flickr8k 构建 SAM3 phrase-level pseudo label 数据集

python build_flickr8k_sam3_pseudo_labels.py \
    --data_dir Flickr8k \
    --dino_pth Flickr8k/flickr8k_dino_train.pth \
    --split train \
    --sam3_checkpoint /path/to/sam3.pt \
    --out_path Flickr8k/flickr8k_sam3_pseudo_train.pth
"""

import argparse
import os
import sys
import warnings
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)
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


GENERIC_PHRASES = {
    "background",
    "foreground",
    "image",
    "picture",
    "photo",
    "scene",
    "thing",
    "things",
    "object",
    "objects",
    "someone",
    "something",
    "people",
}
LEADING_WORDS = {"a", "an", "the", "this", "that", "these", "those", "his", "her", "its", "their"}


def _fmt(color, text, bold=False):
    prefix = (_BOLD if bold else "") + color
    return f"{prefix}{text}{_RESET}"


def log_step(msg): print(_fmt(_CYAN, f"\n▶ {msg}", bold=True), flush=True)
def log_info(msg): print(_fmt(_BLUE, f"  • {msg}"), flush=True)
def log_success(msg): print(_fmt(_GREEN, f"  ✓ {msg}"), flush=True)
def log_warning(msg): print(_fmt(_YELLOW, f"  ⚠ {msg}"), flush=True)
def log_error(msg): print(_fmt(_RED, f"  ✗ {msg}"), flush=True)
def log_kv(k, v): print(f"    {_DIM}{k:<22}{_RESET} {v}", flush=True)


def get_autocast(device):
    """SAM3 CUDA 推理使用 bf16，降低 dtype 不匹配风险"""
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def to_numpy(value):
    """tensor/ndarray 转 numpy，便于后处理"""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype in (torch.bfloat16, torch.float16):
            value = value.float()
        return value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {key: to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_numpy(item) for item in value]
    return value


def get_first(output, keys):
    for key in keys:
        if key in output:
            return output[key]
    return None


def extract_masks(output):
    """统一 SAM3 mask 输出为 [N, H, W] bool"""
    masks = get_first(output, ("masks", "pred_masks", "low_res_masks"))
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool)
    masks = np.asarray(to_numpy(masks))
    if masks.size == 0:
        return np.zeros((0, 0, 0), dtype=bool)
    masks = np.squeeze(masks)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim > 3:
        masks = masks.reshape((-1,) + masks.shape[-2:])
    return masks > 0


def extract_scores(output):
    scores = get_first(output, ("scores", "pred_scores", "iou_predictions"))
    if scores is None:
        return []
    return np.asarray(to_numpy(scores), dtype=float).reshape(-1).tolist()


def resize_bool_mask(mask, size):
    """最近邻 resize，避免 mask 边界产生软标签"""
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask_img) > 0


def make_pil_dino_transform(resize_dim, crop_dim):
    """和 DINO 提取脚本保持一致的 PIL 几何变换"""
    import torchvision.transforms as T

    return T.Compose([
        T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_dim),
    ])


def load_spacy_model(model_name):
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit("缺少 spaCy：请先安装 spacy，并下载 en_core_web_sm") from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"找不到 spaCy 模型 {model_name!r}请运行：python -m spacy download {model_name}"
        ) from exc


def clean_phrase(text):
    """清洗 noun chunk，减少无效 prompt"""
    phrase = " ".join(text.lower().strip().split())
    words = phrase.split()
    while words and words[0] in LEADING_WORDS:
        words = words[1:]
    phrase = " ".join(words)
    phrase = phrase.strip(".,;:!?\"'()[]{}")
    if not phrase or phrase in GENERIC_PHRASES:
        return None
    if len(phrase) < 2:
        return None
    return phrase


def extract_noun_phrases(nlp, caption, max_phrases):
    """使用 spaCy noun_chunks 提取短语名词"""
    doc = nlp(caption)
    phrases = []
    seen = set()
    for chunk in doc.noun_chunks:
        phrase = clean_phrase(chunk.text)
        if phrase is None or phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


def load_flickr8k_captions(data_dir, split, image_name_to_id):
    """读取 Flickr8k token 标注，并用 DINO image_id 对齐"""
    split_map = {
        "train": "Flickr_8k.trainImages.txt",
        "dev": "Flickr_8k.devImages.txt",
        "test": "Flickr_8k.testImages.txt",
    }

    if split == "all":
        split_images = set()
        for fname in split_map.values():
            with open(os.path.join(data_dir, fname), "r") as f:
                split_images.update(line.strip() for line in f if line.strip())
    else:
        if split not in split_map:
            raise ValueError("split 必须是 train/dev/test/all")
        with open(os.path.join(data_dir, split_map[split]), "r") as f:
            split_images = {line.strip() for line in f if line.strip()}

    annotations = []
    token_file = os.path.join(data_dir, "Flickr8k.token.txt")
    with open(token_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                key, caption = line.split("\t", 1)
                image_name, cap_idx = key.rsplit("#", 1)
            except ValueError:
                continue
            if image_name not in split_images or image_name not in image_name_to_id:
                continue
            annotations.append({
                "id": len(annotations),
                "image_id": image_name_to_id[image_name],
                "image_name": image_name,
                "caption_index": int(cap_idx),
                "caption": caption,
            })
    return annotations


def load_sam3(checkpoint, device, confidence_threshold):
    """加载 SAM3 image model 和 processor"""
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise SystemExit("缺少 SAM3：请先安装 facebookresearch/sam3") from exc

    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        device=device,
        enable_inst_interactivity=False,
    )
    model = model.to(device)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    return model, processor


def load_clip_model(model_name, device):
    """加载 CLIP，用于离线编码 phrase"""
    try:
        import clip
    except ImportError as exc:
        raise SystemExit("缺少 CLIP：请先安装 openai/CLIP") from exc

    model, _ = clip.load(model_name, device=device)
    model.eval()
    return model


def encode_phrases(clip_model, phrases, device, batch_size):
    """批量提取 phrase 的 CLIP 句向量"""
    try:
        import clip
    except ImportError as exc:
        raise SystemExit("缺少 CLIP：请先安装 openai/CLIP") from exc

    features = {}
    for start in tqdm(range(0, len(phrases), batch_size), desc="Encoding phrases", dynamic_ncols=True):
        batch = phrases[start:start + batch_size]
        inputs = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            feats = clip_model.encode_text(inputs).float().cpu()
        for phrase, feat in zip(batch, feats):
            features[phrase] = feat
    return features


def select_union_mask(masks, scores, image_size, score_threshold):
    """按置信度筛选 instance mask，并合并为 concept mask"""
    if len(masks) == 0:
        return None, 0, None

    kept = []
    kept_scores = []
    for index, mask in enumerate(masks):
        score = scores[index] if index < len(scores) else None
        if score is not None and score < score_threshold:
            continue
        if mask.shape != (image_size[1], image_size[0]):
            mask = resize_bool_mask(mask, image_size)
        kept.append(mask)
        if score is not None:
            kept_scores.append(float(score))

    if not kept:
        return None, 0, None

    union = np.logical_or.reduce(kept)
    avg_score = float(np.mean(kept_scores)) if kept_scores else None
    return union, len(kept), avg_score


def mask_to_patch_grid(mask, patch_grid):
    """把 crop 图上的 mask 压到 DINO patch grid"""
    grid_w, grid_h = patch_grid[1], patch_grid[0]
    return resize_bool_mask(mask, (grid_w, grid_h))


def save_debug_overlay(image, mask, path, phrase):
    """保存少量可视化样例，便于检查伪标签质量"""
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 180, 255, 95))
    alpha = Image.fromarray(mask.astype(np.uint8) * 95, mode="L")
    layer.putalpha(alpha)
    base = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(base)
    draw.text((8, 8), phrase, fill=(255, 255, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path, quality=95)


def build_pseudo_labels(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = args.images_dir or os.path.join(args.data_dir, "Flicker8k_Dataset")

    log_step("Loading DINO features")
    log_kv("dino_pth", args.dino_pth)
    data = torch.load(args.dino_pth, map_location="cpu")
    if "image_name_to_id" not in data:
        raise KeyError("dino_pth 缺少 image_name_to_id，请重新运行 flickr8k_dino_extraction.py")
    if data.get("split") is not None and data["split"] != args.split:
        raise ValueError(f"dino_pth split={data['split']!r}，命令行 split={args.split!r}，两者不一致")
    for image in data["images"]:
        if "patch_tokens" not in image:
            raise KeyError("dino_pth 中缺少 patch_tokens，请用 --extract_patch_tokens 重新提取")

    patch_grid_size = args.crop_dim // args.patch_size
    patch_grid = (patch_grid_size, patch_grid_size)
    pil_transform = make_pil_dino_transform(args.resize_dim, args.crop_dim)

    log_success(f"Loaded {len(data['images'])} images")
    log_kv("patch_grid", f"{patch_grid[0]} x {patch_grid[1]}")

    log_step("Extracting noun chunks")
    nlp = load_spacy_model(args.spacy_model)
    annotations = load_flickr8k_captions(args.data_dir, args.split, data["image_name_to_id"])
    annotation_phrases = []
    phrase_counter = Counter()
    for ann in tqdm(annotations, desc="spaCy noun_chunks", dynamic_ncols=True):
        phrases = extract_noun_phrases(nlp, ann["caption"], args.max_phrases_per_caption)
        annotation_phrases.append((ann, phrases))
        phrase_counter.update(phrases)

    unique_phrases = sorted(phrase_counter)
    log_success(f"Found {len(unique_phrases)} unique phrases from {len(annotations)} captions")

    log_step("Encoding phrase features with CLIP")
    clip_model = load_clip_model(args.clip_model, device)
    phrase_features = encode_phrases(clip_model, unique_phrases, device, args.clip_batch_size)
    del clip_model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    log_step("Loading SAM3")
    _sam_model, processor = load_sam3(args.sam3_checkpoint, device, args.confidence_threshold)
    log_success(f"SAM3 is ready on {device}")

    images_by_id = {image["id"]: image for image in data["images"]}
    anns_by_image = {}
    for ann, phrases in annotation_phrases:
        if phrases:
            anns_by_image.setdefault(ann["image_id"], []).append((ann, phrases))

    out_images = []
    phrase_samples = []
    stats = Counter()
    debug_count = 0

    log_step("Generating SAM3 pseudo labels")
    for image_id, image_meta in tqdm(images_by_id.items(), desc="SAM3 pseudo labels", dynamic_ncols=True):
        image_anns = anns_by_image.get(image_id, [])
        if not image_anns:
            continue

        image_path = os.path.join(images_dir, image_meta["file_name"])
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                crop_image = pil_transform(image)
        except Exception as exc:
            log_warning(f"跳过无法读取的图像 {image_path}: {exc}")
            stats["bad_image"] += 1
            continue

        with torch.inference_mode(), get_autocast(device):
            state = processor.set_image(crop_image)

        out_images.append({
            "id": image_id,
            "file_name": image_meta["file_name"],
            "patch_tokens": image_meta["patch_tokens"].cpu(),
            "patch_grid": patch_grid,
        })

        # 同一张图中重复 phrase 只跑一次 SAM3
        phrase_to_anns = {}
        for ann, phrases in image_anns:
            for phrase in phrases:
                phrase_to_anns.setdefault(phrase, []).append(ann)

        for phrase, phrase_anns in phrase_to_anns.items():
            with torch.inference_mode(), get_autocast(device):
                output = processor.set_text_prompt(state=state, prompt=phrase)

            masks = extract_masks(output)
            scores = extract_scores(output)
            mask, instance_count, avg_score = select_union_mask(
                masks=masks,
                scores=scores,
                image_size=crop_image.size,
                score_threshold=args.sam_score_threshold,
            )
            if mask is None:
                stats["no_mask"] += len(phrase_anns)
                continue

            area_ratio = float(mask.mean())
            if area_ratio < args.min_area_ratio:
                stats["too_small"] += len(phrase_anns)
                continue
            if area_ratio > args.max_area_ratio:
                stats["too_large"] += len(phrase_anns)
                continue

            mask_grid = mask_to_patch_grid(mask, patch_grid)
            for ann in phrase_anns:
                phrase_samples.append({
                    "id": len(phrase_samples),
                    "image_id": image_id,
                    "annotation_id": ann["id"],
                    "caption_index": ann["caption_index"],
                    "caption": ann["caption"],
                    "phrase": phrase,
                    "phrase_clip_feature": phrase_features[phrase],
                    "mask_grid": torch.from_numpy(mask_grid.copy()).bool(),
                    "sam_score": avg_score,
                    "sam_instance_count": instance_count,
                    "area_ratio": area_ratio,
                })

            stats["kept"] += len(phrase_anns)
            if args.debug_vis_dir and debug_count < args.max_debug_vis:
                debug_path = Path(args.debug_vis_dir) / f"{image_id:06d}_{debug_count:04d}_{safe_name(phrase)}.jpg"
                save_debug_overlay(crop_image, mask, debug_path, phrase)
                debug_count += 1

    out = {
        "meta": {
            "split": args.split,
            "resize_dim": args.resize_dim,
            "crop_dim": args.crop_dim,
            "patch_size": args.patch_size,
            "patch_grid": patch_grid,
            "clip_model": args.clip_model,
            "spacy_model": args.spacy_model,
            "source_dino_pth": args.dino_pth,
        },
        "images": out_images,
        "phrase_samples": phrase_samples,
        "stats": dict(stats),
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    log_success(f"Saved pseudo labels at {out_path}")
    log_kv("images", len(out_images))
    log_kv("phrase_samples", len(phrase_samples))
    for key, value in sorted(stats.items()):
        log_kv(key, value)


def safe_name(value):
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "phrase"


def build_argparser():
    parser = argparse.ArgumentParser(
        description="使用 spaCy noun chunks 和 SAM3 为 Flickr8k 构建 phrase-level pseudo labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", required=True, help="Flickr8k 根目录")
    parser.add_argument("--images_dir", default=None, help="图像目录，默认 data_dir/Flicker8k_Dataset")
    parser.add_argument("--dino_pth", required=True, help="包含 patch_tokens 的 DINO .pth")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test", "all"])
    parser.add_argument("--out_path", required=True, help="输出 pseudo label .pth")

    parser.add_argument("--sam3_checkpoint", required=True, help="SAM3 checkpoint 路径")
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--sam_score_threshold", type=float, default=0.5)

    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--clip_batch_size", type=int, default=256)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--max_phrases_per_caption", type=int, default=6)

    parser.add_argument("--resize_dim", type=int, default=518)
    parser.add_argument("--crop_dim", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--min_area_ratio", type=float, default=0.005)
    parser.add_argument("--max_area_ratio", type=float, default=0.7)

    parser.add_argument("--debug_vis_dir", default=None, help="可选：保存少量 mask 可视化")
    parser.add_argument("--max_debug_vis", type=int, default=64)
    return parser


def main():
    args = build_argparser().parse_args()
    build_pseudo_labels(args)


if __name__ == "__main__":
    main()
