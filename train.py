"""训练 Talk2DINO projector"""


import argparse
import importlib
import os
import sys
import warnings

import numpy as np
import torch
import torchvision.transforms as T
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Talk2DINO 项目内部模块
from src import dataset as dataset_module
from src.dataset import DinoClipDataset
from src.metrics import get_image_and_text_tensor, i2t, t2i
from src.train_util import (
    do_train, set_seed,
    setup_distributed, cleanup_distributed, is_main_process,
)

# 屏蔽 DINOv2 / xFormers 相关 warning，让训练日志更干净
warnings.filterwarnings("ignore", message=r".*xFormers is not available.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*TypedStorage is deprecated.*", category=UserWarning)


# -------------------------------
# 友好型终端输出工具
# -------------------------------
class Console:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


def _color(text, color):
    return f"{color}{text}{Console.RESET}"


def log_header(title):
    print(_color(f"\n{'=' * 18} {title} {'=' * 18}", Console.BOLD + Console.CYAN))


def log_step(title):
    print(_color(f"\n▶ {title}", Console.BOLD + Console.CYAN))


def log_info(message):
    print(_color("  • ", Console.BLUE) + str(message))


def log_success(message):
    print(_color("  ✓ ", Console.GREEN) + str(message))


def log_warning(message):
    print(_color("  ! ", Console.YELLOW) + str(message))


def log_kv(key, value):
    print(f"    {_color(str(key).ljust(18), Console.DIM)} {value}")


# 使用可用设备进行训练
device = "cuda" if torch.cuda.is_available() else "cpu"


def _format_model_summary(model):
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params >= 1_000_000:
        n_params_text = f"{n_params / 1_000_000:.2f}M"
    elif n_params >= 1_000:
        n_params_text = f"{n_params / 1_000:.2f}K"
    else:
        n_params_text = str(n_params)
    return n_params_text, trainable


def train_and_eval(
    config_file,
    train_dataset,
    val_dataset,
    texts=None,
    images=None,
    model_type="cls",
    test_set=None,
    optimizer="adam",
    weight_decay=0.05,
    scheduler="linear",
    warmup=0,
    name_pedix="",
    save_head_activations=None,
):
    """核心训练与评估函数（支持单卡/多卡 DDP）"""
    set_seed(123)

    out_dir = "weights"
    os.makedirs(out_dir, exist_ok=True)

    model_name = os.path.basename(config_file).split(".")[0]
    if name_pedix != "":
        model_name += f"_{name_pedix}"

    if model_type == "":
        out_path = os.path.join(out_dir, f"{model_name}")
    else:
        out_path = os.path.join(out_dir, f"{model_name}_{model_type}")

    if is_main_process():
        log_step("Loading training config")
        log_kv("config", config_file)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    if is_main_process():
        log_kv("model class", model_class_name)

    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)

    if is_main_process():
        log_step("Building model")
    model = ModelClass.from_config(config["model"])
    model.to(device)
    if is_main_process():
        n_params_text, trainable = _format_model_summary(model)
        log_kv("device", device)
        log_kv("parameters", n_params_text)
        log_kv("trainable", trainable)
        log_kv("save path", f"{out_path}.pth")

    if is_main_process():
        log_step("Training projector")
    model, train_losses, val_losses = do_train(
        model,
        train_dataset,
        val_dataset,
        config["train"],
        optimizer_name=optimizer,
        weight_decay=weight_decay,
        scheduler_name=scheduler,
        warmup=warmup,
        save_head_attivations=save_head_activations,
    )

    # 只在主进程保存模型和评估
    if is_main_process():
        torch.save(model.state_dict(), f"{out_path}.pth")
        log_success(f"Model saved: {out_path}.pth")

        # patch-token 特征在部分对齐策略下需要训练后重新构造测试张量
        if model_type == "patch_tokens":
            log_info("Rebuilding image/text tensors for patch-token evaluation")
            images, texts = get_image_and_text_tensor(args.test_dataset, args.feature_name, model=model)

        if texts is not None:
            log_step("Retrieval evaluation")
            texts_proj = model.project_clip_txt(texts.to(device).float()).detach().cpu()

            t2i_rk = t2i(images.numpy(), texts_proj.numpy())
            i2t_rk = i2t(images.numpy(), texts_proj.numpy())

            data = [
                ["t2i"] + list(t2i_rk),
                ["i2t"] + list(i2t_rk),
            ]
            columns = ["type", "r1", "r5", "r10", "median_rank", "mean_rank"]
            import pandas as pd

            df = pd.DataFrame(data, columns=columns)
            print(df.to_string(index=False))

    return train_losses, val_losses


def plot_losses(train_losses, val_losses, labels=("Training Loss", "Validation Loss")):
    """绘制训练和验证损失曲线"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=labels[0], color="blue", marker="o")
    plt.plot(val_losses, label=labels[1], color="red", marker="o")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def build_argparser():
    """定义训练脚本命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_dim", type=int, default=518, help="中心裁剪尺寸，预处理图像大小")
    parser.add_argument("--data_dir", type=str, default="../coco/", help="原始图像目录")
    parser.add_argument("--feature_name", type=str, default="disentangled_self_attn", help="选择哪种 DINO 特征")
    parser.add_argument("--text_features", type=str, default="ann_feats", help="文本特征字段名")
    parser.add_argument("--model_config", type=str, default="dinov2_vitl14_reg", help="模型配置文件路径或名称")
    parser.add_argument("--resize_dim", type=int, default=518, help="缩放尺寸，预处理用")
    parser.add_argument("--test_dataset", type=str, default="../coco2014_features/test.pth", help="测试集特征文件")
    parser.add_argument("--train_dataset", type=str, default="../coco2014_features/train.pth", help="训练集特征文件")
    parser.add_argument("--val_dataset", type=str, default="../coco2014_features/val.pth", help="验证集特征文件")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="是否使用 wandb 记录训练日志")
    parser.add_argument("--optimizer", type=str, default="Adam", help="优化器名称")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减")
    parser.add_argument("--scheduler", type=str, default="linear", help="学习率调度器")
    parser.add_argument("--name_pedix", type=str, default="", help="附加在权重文件名上的前缀")
    parser.add_argument("--save_head_activations", type=str, default=None, help="是否保存最后一层 head 激活")
    parser.add_argument("--warmup", type=int, default=0, help="预热 epoch 数")
    return parser


def main():
    args = build_argparser().parse_args()

    # 初始化分布式训练（如果通过 torchrun 启动）
    distributed = setup_distributed()

    # 设置当前进程使用的 GPU
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        global device
        device = f"cuda:{local_rank}"

    if is_main_process():
        log_header("Training The Projector")
        log_kv("device", device)
        log_kv("distributed", distributed)
        if distributed:
            log_kv("world size", int(os.environ.get("WORLD_SIZE", 1)))
        log_kv("model config", args.model_config)
        log_kv("train dataset", args.train_dataset)
        log_kv("val dataset", args.val_dataset)
        log_kv("feature", args.feature_name)
        log_kv("text feature", args.text_features)

    if args.use_wandb and is_main_process():
        log_info("Initializing wandb project: dino-clip")
        import wandb
        wandb.init(project="dino-clip")

    if is_main_process():
        log_step("Loading datasets")
    if not ("dino" in args.model_config):
        train_feature_name = args.feature_name
        val_feature_name = "disentangled_self_attn" if args.feature_name == "disentangled_self_attn" else args.feature_name

        if is_main_process():
            log_kv("dataset mode", "pre-extracted features")
            log_kv("train image field", train_feature_name)
            log_kv("val image field", val_feature_name)
            log_kv("text field", args.text_features)

        # 位置参数兼容旧版 DinoClipDataset(feature_name=...) 接口
        val_dataset = DinoClipDataset(
            args.val_dataset,
            val_feature_name,
            args.text_features,
            args.feature_name == "patch_tokens",
        )
        train_dataset = DinoClipDataset(
            args.train_dataset,
            train_feature_name,
            args.text_features,
            args.feature_name == "patch_tokens",
        )
    else:
        if is_main_process():
            log_kv("dataset mode", "online image-caption loading")
            log_kv("resize / crop", f"{args.resize_dim} / {args.crop_dim}")

        image_transforms = T.Compose(
            [
                T.Resize(args.resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(args.crop_dim),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        import clip

        COCOCaptions = getattr(dataset_module, "COCOCaptions", None)
        if COCOCaptions is None:
            raise ImportError(
                "当前 src.dataset 中没有 COCOCaptions"
                "请使用预提取特征模式，或恢复 COCOCaptions 数据集类"
            )
        train_dataset = COCOCaptions(args.train_dataset, "coco/train2014", "train", image_transforms, clip.tokenize)
        val_dataset = COCOCaptions(args.val_dataset, "coco/val2014", "val", image_transforms, clip.tokenize)

    if is_main_process():
        log_success("Datasets are ready")
        log_kv("train samples", len(train_dataset))
        log_kv("val samples", len(val_dataset))

    if args.feature_name == "patch_tokens":
        if is_main_process():
            log_step("Preparing retrieval tensors")
        if args.text_features == "clip_second_last_out":
            images, texts, text_argmax = get_image_and_text_tensor(args.test_dataset, args.feature_name, args.text_features)
        else:
            images, texts = get_image_and_text_tensor(args.test_dataset, args.feature_name, args.text_features)
        if is_main_process():
            log_success("Retrieval tensors are ready")
    else:
        images = None
        texts = None

    train_and_eval(
        args.model_config,
        train_dataset,
        val_dataset,
        texts,
        images,
        test_set=args.test_dataset,
        model_type="",
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup=args.warmup,
        name_pedix=args.name_pedix,
        save_head_activations=args.save_head_activations,
    )

    # 清理分布式进程组
    cleanup_distributed()


if __name__ == "__main__":
    main()
