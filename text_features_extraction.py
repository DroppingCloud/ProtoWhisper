"""从 COCO 风格标注中提取 CLIP 文本特征"""

import argparse
import math
import os
import sys
import warnings

import clip
import torch
from tqdm import tqdm

# 项目内部工具：
# - hooks：用于注册 forward hook，额外保存 CLIP 中间层或 token 级输出
from src.hooks import feats, get_all_out_tokens, get_clip_second_last_dense_out

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


def register_text_hooks(model, extract_dense_out, extract_second_last_dense_out):
    """
    按需注册 CLIP 文本编码器的 forward hook 

    参数:
        model:
            CLIP 模型 

        extract_dense_out:
            是否提取 CLIP 最后一层的 token-level 文本特征 
            若开启，会在 model.ln_final 上注册 hook 

        extract_second_last_dense_out:
            是否提取 CLIP 倒数第二层 Transformer block 的输出 
            若开启，会在 transformer.resblocks[-2] 上注册 hook 

    说明:
        CLIP 默认 encode_text 只返回句子级文本向量 
        如果需要每个 token 的 dense 表示，就必须通过 hook 捕获中间特征 
    """
    if extract_dense_out:
        # 捕获 CLIP 文本编码器最后 ln_final 的所有 token 输出
        model.ln_final.register_forward_hook(get_all_out_tokens)

    if extract_second_last_dense_out:
        # 捕获 CLIP 文本 Transformer 倒数第二层的 token 输出
        model.transformer.resblocks[-2].register_forward_hook(get_clip_second_last_dense_out)


def load_caption_data(ann_path):
    """
    读取 COCO 风格 caption 数据

    支持两种输入:
        1. ann_path 是 .json: COCO 标注 JSON 文件
        2. ann_path 是其他文件: 默认 torch.load (.pth)
    """
    log_step("Loading caption data")
    log_kv("source", ann_path)

    if ann_path.endswith(".json"):
        log_info("Detected COCO JSON annotations")
        import json
        with open(ann_path, "r") as f:
            return json.load(f)

    log_info("Detected PTH annotations/features")
    return torch.load(ann_path)


def get_batch_bounds(batch_index, batch_size, total_items, total_batches):
    """
    计算当前 batch 在 annotations 列表中的起止下标 

    参数:
        batch_index:
            当前 batch 编号 

        batch_size:
            每个 batch 的样本数量 

        total_items:
            annotation 总数 

        total_batches:
            batch 总数 

    返回:
        start, end:
            当前 batch 对应的左闭右开区间 [start, end) 

    说明:
        最后一个 batch 可能不足 batch_size，因此需要单独处理 end 
    """
    start = batch_index * batch_size
    end = start + batch_size if batch_index < total_batches - 1 else total_items
    return start, end


def extract_text_batch(model, texts, extract_dense_out, device):
    """
    对一个 batch 的 caption 文本提取 CLIP 文本特征 

    参数:
        model:
            CLIP 模型 

        texts:
            当前 batch 的 caption 文本列表 

        extract_dense_out:
            是否额外提取 token-level dense 文本特征 

        device:
            运行设备，cuda 或 cpu 

    返回:
        inputs:
            tokenized 后的文本输入，shape 通常为 [B, 77] 

        batch_features:
            当前 batch 的文本特征字典，可能包含:
            - ann_feats:
                CLIP encode_text 输出的句子级文本特征 
            - clip_txt_out_tokens:
                CLIP 最后一层 token-level 文本特征 
            - text_input_mask:
                文本 token mask，用于区分有效 token 和 padding 
    """
    # CLIP 默认最大文本长度为 77，truncate=True 表示超长文本会被截断
    inputs = clip.tokenize(texts, truncate=True).to(device)

    with torch.no_grad():
        # 句子级文本特征，通常取 EOT token 对应位置并经过 text_projection
        outputs = model.encode_text(inputs)

        batch_features = {
            "ann_feats": outputs
        }

        if extract_dense_out:
            # hook 会把 ln_final 的 token-level 输出保存到 feats["clip_txt_out_tokens"]
            #
            # 这里再乘 model.text_projection，是为了让 token-level 特征
            # 和 encode_text 输出的句子级特征处于相同的 CLIP 投影空间 
            batch_features["clip_txt_out_tokens"] = (
                feats["clip_txt_out_tokens"] @ model.text_projection
            )

            # inputs > 0 用于标记有效 token 
            # padding token 通常为 0 
            batch_features["text_input_mask"] = inputs > 0

    return inputs, batch_features


def write_text_features(
    data,
    start,
    end,
    inputs,
    batch_features,
    extract_dense_out,
    extract_second_last_dense_out,
):
    """
    将当前 batch 提取到的文本特征写回 data["annotations"] 

    参数:
        data:
            COCO 风格数据字典 

        start, end:
            当前 batch 在 annotations 中的下标范围 

        inputs:
            CLIP tokenize 后的文本输入 

        batch_features:
            当前 batch 的特征字典 

        extract_dense_out:
            是否写入最后一层 token-level 特征 

        extract_second_last_dense_out:
            是否写入倒数第二层 token-level 特征 

    写回字段:
        annotation["ann_feats"]:
            句子级 CLIP 文本特征 

        annotation["clip_txt_out_tokens"]:
            最后一层 token-level 文本特征 

        annotation["text_input_mask"]:
            有效文本 token mask 

        annotation["clip_second_last_out"]:
            倒数第二层 token-level 文本特征 

        annotation["text_argmax"]:
            文本中 token id 最大的位置 
            在 CLIP 原实现里，EOT token 通常具有较大的 token id，
            因此 argmax 常用于定位句子级特征对应的 EOT 位置 
    """
    for annotation_index in range(start, end):
        batch_index = annotation_index - start
        annotation = data["annotations"][annotation_index]

        # 写入句子级文本特征
        annotation["ann_feats"] = batch_features["ann_feats"][batch_index].to("cpu")

        if extract_dense_out:
            # 写入 CLIP 最后一层所有 token 的特征
            annotation["clip_txt_out_tokens"] = (
                batch_features["clip_txt_out_tokens"][batch_index].to("cpu")
            )

            # 写入有效 token mask
            annotation["text_input_mask"] = (
                batch_features["text_input_mask"][batch_index].to("cpu")
            )

        if extract_second_last_dense_out:
            # 写入倒数第二层 Transformer block 的 token-level 输出
            annotation["clip_second_last_out"] = (
                feats["clip_second_last_out"][batch_index].to("cpu")
            )

            # 记录 EOT token 位置，供后续模型使用
            annotation["text_argmax"] = (
                inputs.argmax(dim=-1)[batch_index].to("cpu")
            )


def save_text_features(data, ann_path, out_path):
    """
    保存提取后的文本特征（.pth 文件）

    参数:
        data:
            已经写入文本特征的 COCO 风格数据

        ann_path:
            输入标注路径，用于在 out_path 为空时生成默认输出名

        out_path:
            输出路径
    """
    if out_path is None:
        # 若未指定输出路径，则默认把输入文件后缀替换为 .pth
        out_path = os.path.splitext(ann_path)[0] + ".pth"

    torch.save(data, out_path)
    log_success(f"Features saved at {out_path}")


def run_clip_text_extraction(
    model_name,
    ann_path,
    batch_size,
    out_path,
    extract_dense_out=False,
    extract_second_last_dense_out=False,
):
    """
    CLIP 文本特征提取主流程

    流程:
        1. 加载 CLIP 模型
        2. 按需注册 hook
        3. 读取 COCO 风格 caption 数据
        4. 遍历 annotations 中的 caption
        5. 批量 tokenize 并 encode_text
        6. 将特征写回 annotation
        7. 保存为 .pth 文件

    参数:
        model_name:
            CLIP 模型名称，例如 "ViT-B/16"

        ann_path:
            输入标注或特征文件路径

        batch_size:
            文本特征提取 batch size

        out_path:
            输出路径

        extract_dense_out:
            是否提取最后一层 token-level 文本特征

        extract_second_last_dense_out:
            是否提取倒数第二层 token-level 文本特征
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_step("Loading CLIP text encoder")
    log_kv("model", model_name)
    log_kv("device", device)

    # 加载 CLIP 模型第二个返回值是预处理 transform，这里只提取文本特征，所以不使用
    with warnings.catch_warnings():
        model, _ = clip.load(model_name, device=device)

    # 如果需要 token-level 特征，注册对应 hook
    register_text_hooks(model, extract_dense_out, extract_second_last_dense_out)

    model.eval()

    # 读取 COCO 风格数据
    data = load_caption_data(ann_path)

    log_step("Starting CLIP text feature extraction")

    # caption 总数
    n_capts = len(data["annotations"])

    # 根据 batch_size 计算总 batch 数
    n_batch = math.ceil(n_capts / batch_size)
    log_kv("captions", n_capts)
    log_kv("batch size", batch_size)
    log_kv("batches", n_batch)
    log_kv("features", ", ".join(
        name for name, enabled in [
            ("ann_feats", True),
            ("clip_txt_out_tokens", extract_dense_out),
            ("clip_second_last_out", extract_second_last_dense_out),
        ] if enabled
    ))

    for batch_index in tqdm(
        range(n_batch),
        desc="Extracting CLIP text features",
        dynamic_ncols=True,
        colour="cyan" if _COLOR_ENABLED else None,
    ):
        # 当前 batch 在 annotations 中的范围
        start, end = get_batch_bounds(batch_index, batch_size, n_capts, n_batch)

        # 只取 caption 文本；图像信息和图像特征会保留在原始 data 结构中
        texts = [data["annotations"][j]["caption"] for j in range(start, end)]

        # 提取当前 batch 的文本特征
        inputs, batch_features = extract_text_batch(
            model,
            texts,
            extract_dense_out,
            device,
        )

        # 写回 data["annotations"]
        write_text_features(
            data,
            start,
            end,
            inputs,
            batch_features,
            extract_dense_out,
            extract_second_last_dense_out,
        )

    log_success("Feature extraction finished")

    # 保存最终结果
    save_text_features(data, ann_path, out_path)


def build_argparser():
    """
    定义 CLIP 文本特征提取脚本的命令行参数 

    示例:
        python clip_text_extraction.py \\
            --ann_path /root/autodl-tmp/datasets/coco2014_features/train.pth \\
            --model ViT-B/16 \\
            --batch_size 256 \\
            --extract_dense_out \\
            --out_path /root/autodl-tmp/datasets/coco2014_features/train_clip_text.pth
    """
    parser = argparse.ArgumentParser()

    # -------------------------------
    # 输入输出参数
    # -------------------------------

    parser.add_argument(
        "--ann_path",
        type=str,
        default="coco/test1k.json",
        help=(
            "输入标注或特征文件路径 通常是 COCO 风格 .pth 文件，"
            "其中 data['annotations'] 里包含 caption 字段 "
        ),
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help=(
            "输出文件路径 若不指定，则默认使用 ann_path 的文件名，"
            "并替换为 .pth 后缀 "
        ),
    )

    # -------------------------------
    # 模型与批处理参数
    # -------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help=(
            "文本特征提取的 batch size 越大速度通常越快，"
            "但会占用更多显存 "
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/16",
        help=(
            "CLIP 模型名称，例如 ViT-B/16、ViT-B/32、RN50 等 "
            "需要与后续训练使用的 CLIP 文本空间保持一致 "
        ),
    )

    # -------------------------------
    # 文本特征类型参数
    # -------------------------------

    parser.add_argument(
        "--extract_dense_out",
        action="store_true",
        default=False,
        help=(
            "是否提取 CLIP 最后一层所有 token 的 dense 文本特征 "
            "默认只提取句子级 ann_feats "
        ),
    )

    parser.add_argument(
        "--extract_second_last_dense_out",
        action="store_true",
        default=False,
        help=(
            "是否提取 CLIP 倒数第二层 Transformer block 的 token-level 输出 "
            "常用于需要更细粒度文本表示的对齐方法 "
        ),
    )

    return parser


def main():
    """
    命令行入口:
        1. 解析参数 
        2. 调用 run_clip_text_extraction 执行特征提取 
    """
    args = build_argparser().parse_args()

    run_clip_text_extraction(
        args.model,
        args.ann_path,
        args.batch_size,
        args.out_path,
        args.extract_dense_out,
        args.extract_second_last_dense_out,
    )


if __name__ == "__main__":
    main()
