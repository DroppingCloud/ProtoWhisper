# ------------------------------------------------------------------------------
# Talk2DINO
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
"""开放词汇语义分割评测入口：友好型终端输出版本"""

import os
import warnings
import logging

# ------------------------------
# Suppress noisy third-party warnings before importing mmcv/torchvision/transformers.
# ------------------------------
warnings.filterwarnings(
    "ignore",
    message="On January 1, 2023, MMCV will release v2.0.0.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The default value of the antialias parameter.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated.*",
    category=UserWarning,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

import argparse
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import collect_env, get_git_hash
from torch.utils.data import Subset

from src.open_vocabulary_segmentation.models import build_model
from omegaconf import OmegaConf

from src.open_vocabulary_segmentation.segmentation.evaluation import (
    build_dinotext_seg_inference,
    build_seg_dataloader,
    build_seg_dataset,
)
import src.open_vocabulary_segmentation.us as us
from src.open_vocabulary_segmentation.utils import (
    get_logger,
    load_config,
)

try:
    from termcolor import colored
except ImportError:  # 与 logger.py 保持一致：没有 termcolor 时自动退化为普通文本
    def colored(text, *args, **kwargs):
        return text


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def silence_external_loggers():
    """降低第三方库日志噪声，只保留 warning/error，避免干扰评测摘要"""
    noisy_loggers = [
        "mmseg",
        "mmcv",
        "mmdet",
        "transformers",
        "PIL",
    ]
    for name in noisy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        # mmseg 有时会提前绑定 handler；保守处理为不向上重复传播
        logger.propagate = False


from mmseg.datasets import PIPELINES, PascalVOCDataset, PascalContextDataset, ADE20KDataset, CityscapesDataset, \
    COCOStuffDataset, PascalContextDataset59


@PIPELINES.register_module()
class FloatImage:
    def __call__(self, results):
        results['img'] = results['img'].astype(np.float32)
        return results


def is_main_process():
    """只让主进程打印关键摘要，避免多卡评测时日志刷屏"""
    if DEVICE == "cpu":
        return True
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def c(text, color=None, attrs=None):
    """轻量颜色包装，termcolor 不可用时会自动返回原文本"""
    return colored(str(text), color, attrs=attrs)


def fmt_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def log_section(logger, title, color="cyan"):
    if not is_main_process():
        return
    line = "=" * 78
    logger.info("\n%s\n%s\n%s", c(line, color), c(f" {title}", color, attrs=["bold"]), c(line, color))


def log_subsection(logger, title, color="blue"):
    if not is_main_process():
        return
    logger.info(c(f"\n▶ {title}", color, attrs=["bold"]))


def log_kv_table(logger, title, items, key_width=22, color="cyan"):
    """输出对齐的 key-value 表items: list[tuple[str, Any]]"""
    if not is_main_process():
        return
    log_subsection(logger, title, color=color)
    for key, value in items:
        logger.info("  %s : %s", c(f"{key:<{key_width}}", "yellow"), fmt_value(value))
    logger.info("")


def log_metric_table(logger, title, rows):
    """输出评测指标表rows: list[tuple[dataset, num_images, miou]]"""
    if not is_main_process():
        return
    log_subsection(logger, title, color="green")
    header = f"{'Dataset':<18} {'Images':>10} {'mIoU (%)':>12}"
    logger.info("  %s", c(header, "yellow", attrs=["bold"]))
    logger.info("  %s", c("-" * len(header), "yellow"))
    for dataset, num_images, miou in rows:
        logger.info("  %-18s %10s %12.2f", dataset, num_images, miou)
    logger.info("")


def get_argparser():
    parser = argparse.ArgumentParser("DINOText segmentation evaluation script")
    parser.add_argument(
        "--opts", help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs="+"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="evaluation output folder",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--wandb", action="store_true", help="Use W&B to log experiments")
    parser.add_argument("--wandb_name", type=str, help="W&B run name", default="default")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval_cfg', type=str, default="configs/dinotext.yml")
    parser.add_argument("--eval_base_cfg", type=str, default="configs/eval.yml")

    parser.add_argument("--pred_qual_path", type=str, default=None)
    parser.add_argument("--gt_qual_path", type=str, default=None)

    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--num_jobs", type=int, default=1)

    return parser


def log_results(miou, proj_name, bench, result_dir, logger):
    os.makedirs(result_dir, exist_ok=True)

    json_path = os.path.join(result_dir, f"{proj_name}.json")

    # 如果已有结果文件，则在原结果上更新当前 benchmark
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[bench] = miou

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    log_kv_table(
        logger,
        "Result Saved",
        [
            ("benchmark", bench),
            ("projector", proj_name),
            ("mIoU", f"{miou:.2f}%"),
            ("json path", json_path),
        ],
        key_width=14,
        color="green",
    )


def build_eval_loaders(cfg, args):
    """根据评测配置构建 MMSeg 数据集和 dataloader"""
    logger = get_logger()
    val_loaders = {}
    loader_rows = []

    for key in cfg.evaluate.task:
        if key == "cls":
            continue

        dataset = build_seg_dataset(cfg.evaluate.get(key))
        len_dataset = len(dataset)

        first_sample = args.job_id * len_dataset // args.num_jobs
        last_sample = ((args.job_id + 1) * len_dataset // args.num_jobs)
        if args.job_id == args.num_jobs - 1:
            last_sample = len_dataset

        subset_size = last_sample - first_sample
        dataset = Subset(dataset, range(first_sample, last_sample))
        loader = build_seg_dataloader(dataset)
        val_loaders[key] = loader

        loader_rows.append((key, len_dataset, f"[{first_sample}, {last_sample})", subset_size))

    if is_main_process():
        log_subsection(logger, "Evaluation Datasets", color="cyan")
        header = f"{'Task':<18} {'Total':>10} {'Shard Range':>20} {'Current':>10}"
        logger.info("  %s", c(header, "yellow", attrs=["bold"]))
        logger.info("  %s", c("-" * len(header), "yellow"))
        for task, total, shard_range, current in loader_rows:
            logger.info("  %-18s %10d %20s %10d", task, total, shard_range, current)
        logger.info("")

    return val_loaders


def build_eval_model(cfg):
    """构建 DINOText 模型，并在 CUDA 环境下包装为分布式评测模型"""
    logger = get_logger()

    log_section(logger, "Build Evaluation Model")
    log_kv_table(
        logger,
        "Model Config",
        [
            ("model type", cfg.model.type),
            ("model name", cfg.model_name),
            ("projector", cfg.model.get("proj_name", "N/A") if hasattr(cfg.model, "get") else "N/A"),
            ("device", DEVICE),
        ],
    )

    model = build_model(cfg.model)
    if DEVICE == "cuda":
        model.cuda()
        model = MMDistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_kv_table(
        logger,
        "Model Statistics",
        [
            ("trainable params", f"{n_parameters:,}"),
            ("params (M)", f"{n_parameters / 1000 / 1000:.2f}M"),
        ],
    )
    return model


def run_evaluation(cfg, args):
    """执行开放词汇分割评测，并记录 mIoU 结果"""
    if DEVICE == "cuda":
        dist.barrier()

    logger = get_logger()
    log_section(logger, "Open-Vocabulary Segmentation Evaluation")

    val_loaders = build_eval_loaders(cfg, args)
    model = build_eval_model(cfg)

    res = evaluate(cfg, model, val_loaders)
    metrics = res.pop("metrics", None)

    metric_rows = []
    for key, value in res.items():
        if key.startswith("val/") and key.endswith("_miou") and key != "val/avg_miou":
            dataset_name = key.replace("val/", "").replace("_miou", "")
            num_images = len(val_loaders[dataset_name].dataset) if dataset_name in val_loaders else "-"
            metric_rows.append((dataset_name, num_images, value))

    if metric_rows:
        log_metric_table(logger, "Evaluation Summary", metric_rows)

    log_kv_table(
        logger,
        "Final Result",
        [
            ("average mIoU", f"{res['val/avg_miou']:.2f}%"),
            ("experiment dir", cfg.output),
        ],
        color="green",
    )

    if cfg.wandb and metrics:
        import wandb
        wandb.init(
            project="open-vocab-metrics",
            name=args.wandb_name,
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=False,
        )
        wandb.log(metrics[0])

    log_results(
        res["val/avg_miou"],
        cfg["model"]["proj_name"],
        cfg["evaluate"]["task"][0],
        "segmentation_results",
        logger,
    )


@torch.no_grad()
def evaluate(cfg, model, val_loaders):
    logger = get_logger()
    ret = {}
    model.eval()

    for key, loader in val_loaders.items():
        if key == "cls":
            continue

        dataset_class = loader.dataset.__class__.__name__
        log_subsection(logger, f"Validate: {key}", color="magenta")
        log_kv_table(
            logger,
            "Dataset Runtime Info",
            [
                ("dataset wrapper", dataset_class),
                ("num images", len(loader.dataset)),
            ],
            key_width=16,
            color="magenta",
        )

        miou, metrics = validate_seg(cfg, cfg.evaluate.get(key), loader, model)

        log_kv_table(
            logger,
            f"{key} Metrics",
            [("mIoU", f"{miou:.2f}%")],
            key_width=16,
            color="green",
        )
        ret[f"val/{key}_miou"] = miou
        ret[f"metrics"] = metrics

    ret["val/avg_miou"] = np.mean([v for k, v in ret.items() if "miou" in k])

    return ret


@torch.no_grad()
def validate_seg(config, seg_config, data_loader, model):
    logger = get_logger()
    if DEVICE == "cuda":
        dist.barrier()

    model.eval()

    if hasattr(model, "module"):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    log_subsection(logger, "Build segmentation inference wrapper", color="blue")
    seg_model = build_dinotext_seg_inference(
        model_without_ddp,
        data_loader.dataset,
        config,
        seg_config,
    )

    if DEVICE == "cuda":
        mmddp_model = MMDistributedDataParallel(
            seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
    else:
        mmddp_model = seg_model
    mmddp_model.eval()

    log_subsection(logger, "Run multi-gpu test", color="blue")
    # TODO: Use multi-gpu-test from mmseg instead of ours
    results, pred_qualitatives, gt_qualitatives, num_classes = us.multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=DEVICE == "cuda",
        efficient_test=False,
        pre_eval=True,
        format_only=False,
    )

    # mmcv.ProgressBar 结束后有时不会额外空行，这里补一个换行，避免后续日志粘在进度条后面
    if is_main_process():
        print("", flush=True)

    if DEVICE == "cpu" or dist.get_rank() == 0:
        # logger="silent" 可隐藏 mmseg 的 per-class results / Summary 表格，仍正常返回 mIoU
        metric = [data_loader.dataset.dataset.evaluate(results, metric="mIoU", logger="silent")]
    else:
        metric = [None]

    if DEVICE == "cuda":
        dist.broadcast_object_list(metric)
    miou_result = metric[0]["mIoU"] * 100

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        dist.barrier()
    return miou_result, metric


def main():
    silence_external_loggers()

    parser = get_argparser()
    args = parser.parse_args()

    if not args.eval:
        parser.error(
            "src/open_vocabulary_segmentation/main.py 当前只支持 --eval 分割评测；"
            "投影层训练请使用仓库根目录的 train.py"
        )

    # 评测配置合并顺序：模型默认配置 -> 空的历史实验配置占位 -> 数据集评测配置
    default_cfg = load_config(args.eval_cfg)
    org_cfg = OmegaConf.create()
    eval_cfg = OmegaConf.load(args.eval_base_cfg)
    cfg = OmegaConf.merge(default_cfg, org_cfg, eval_cfg)
    if args.opts is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))

    cfg.wandb = args.wandb
    cfg.evaluate.eval_only = True
    args.output = args.output if args.output is not None else "output/eval"
    cfg.output = args.output

    Path(cfg.output).mkdir(parents=True, exist_ok=True)

    if DEVICE == "cuda":
        # start faster ref: https://github.com/open-mmlab/mmdetection/pull/7036
        mp.set_start_method("fork", force=True)
        init_dist("pytorch")
        rank, world_size = get_dist_info()
        dist.barrier()
    else:
        rank = 0
        world_size = 1

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    log_section(logger, "Talk2DINO Evaluation Launch", color="green")
    log_kv_table(
        logger,
        "Runtime",
        [
            ("device", DEVICE),
            ("rank/world size", f"{rank}/{world_size}"),
            ("seed", cfg.seed),
            ("cudnn benchmark", cudnn.benchmark),
            ("output", cfg.output),
        ],
        color="green",
    )

    if DEVICE == "cuda" and dist.get_rank() == 0:
        path = os.path.join(cfg.output, "config.json")
        OmegaConf.save(cfg, path)
        log_kv_table(logger, "Config Saved", [("config path", path)], key_width=14, color="green")

    # 环境信息仍然写入日志，但终端只展示关键信息，避免刷屏
    env_info_dict = collect_env()
    logger.debug("Full environment info:\n%s", "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()]))

    log_kv_table(
        logger,
        "Environment Snapshot",
        [
            ("python", env_info_dict.get("Python", "N/A")),
            ("pytorch", env_info_dict.get("PyTorch", "N/A")),
            ("cuda available", env_info_dict.get("CUDA available", "N/A")),
            ("cuda runtime", env_info_dict.get("CUDA_HOME", "N/A")),
            ("git hash", get_git_hash(digits=7)),
        ],
        color="cyan",
    )

    logger.debug("Full config:\n%s", OmegaConf.to_yaml(cfg))

    run_evaluation(cfg, args)
    if DEVICE == "cuda":
        dist.barrier()

    log_kv_table(logger, "Finished", [("experiment dir", cfg.output)], key_width=16, color="green")


if __name__ == "__main__":
    main()
