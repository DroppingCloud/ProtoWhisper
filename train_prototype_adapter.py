"""使用 SAM3 区域原型训练文本侧 prototype residual adapter"""

import argparse
import importlib
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prototype_adapter import (
    ResidualTextPrototypeAdapter,
    masked_region_prototypes,
    prototype_contrastive_loss,
)
from src.train_util import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)

warnings.filterwarnings("ignore", message=r".*TypedStorage is deprecated.*", category=UserWarning)

GENERIC_HEADS = {
    "thing", "things", "stuff", "area", "side", "background", "foreground",
    "group", "lot", "lots", "part", "parts", "piece", "pieces", "place",
    "places", "scene", "view", "photo", "picture", "image", "one", "two",
    "three", "some", "many", "several",
}


def log(message):
    if is_main_process():
        print(message, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phrase_head(phrase):
    tokens = [token.strip(".,;:!?()[]{}\"'").lower() for token in phrase.split()]
    tokens = [token for token in tokens if token]
    return tokens[-1] if tokens else ""


def compute_soft_weight(
    sample,
    min_weight=0.20,
    score_floor=0.60,
    score_power=1.0,
    area_center=0.16,
    area_sigma=1.0,
    instance_power=0.5,
):
    """根据 SAM 质量、mask 面积和实例数给 pseudo label 软权重"""
    score = float(sample.get("sam_score", 0.0))
    area = float(sample.get("area_ratio", sample["mask_grid"].float().mean().item()))
    instances = max(1, int(sample.get("sam_instance_count", 1)))

    score_denom = max(1e-6, 1.0 - float(score_floor))
    score_w = min(1.0, max(0.0, (score - float(score_floor)) / score_denom))
    score_w = score_w ** float(score_power)

    area = max(area, 1e-6)
    area_center = max(float(area_center), 1e-6)
    area_sigma = max(float(area_sigma), 1e-6)
    area_w = float(np.exp(-0.5 * (np.log(area / area_center) / area_sigma) ** 2))

    instance_w = instances ** (-float(instance_power))
    raw_weight = score_w * area_w * instance_w
    return float(max(float(min_weight), min(1.0, raw_weight)))


class FilteredSam3PrototypeDataset(Dataset):
    """按 pseudo label 质量过滤 phrase sample"""

    def __init__(
        self,
        pseudo_pth,
        min_area=0.01,
        max_area=0.55,
        min_sam_score=0.70,
        max_instances=3,
        drop_generic_heads=True,
        filter_index_json=None,
        filter_split=None,
        bridge_feature_cache=None,
        bridge_original_prob=1.0,
        use_sample_weight=False,
        weight_min=0.20,
        weight_score_floor=0.60,
        weight_score_power=1.0,
        weight_area_center=0.16,
        weight_area_sigma=1.0,
        weight_instance_power=0.5,
        max_samples=None,
    ):
        data = torch.load(pseudo_pth, map_location="cpu")
        self.meta = data.get("meta", {})
        self.images = {image["id"]: image for image in data["images"]}
        samples = data["phrase_samples"]
        allowed_ids = None
        if filter_index_json is not None:
            with open(filter_index_json, "r") as f:
                index = json.load(f)
            split_key = filter_split or os.path.basename(pseudo_pth).split(".")[0]
            allowed_ids = set(index[split_key]["kept_sample_ids"])

        filtered = []
        for sample in samples:
            if allowed_ids is not None and int(sample["id"]) not in allowed_ids:
                continue
            area = float(sample.get("area_ratio", sample["mask_grid"].float().mean().item()))
            score = float(sample.get("sam_score", 0.0))
            instances = int(sample.get("sam_instance_count", 0))
            head = phrase_head(sample["phrase"])
            if area < min_area or area > max_area:
                continue
            if score < min_sam_score:
                continue
            if instances > max_instances:
                continue
            if drop_generic_heads and head in GENERIC_HEADS:
                continue
            filtered.append(sample)

        if max_samples is not None:
            filtered = filtered[:max_samples]
        self.samples = filtered
        self.use_sample_weight = bool(use_sample_weight)
        self.weight_kwargs = {
            "min_weight": weight_min,
            "score_floor": weight_score_floor,
            "score_power": weight_score_power,
            "area_center": weight_area_center,
            "area_sigma": weight_area_sigma,
            "instance_power": weight_instance_power,
        }
        self.bridge_original_prob = float(bridge_original_prob)
        self.bridge_features = {}
        if bridge_feature_cache is not None:
            cache = torch.load(bridge_feature_cache, map_location="cpu")
            self.bridge_features = cache.get("features", cache)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.images[sample["image_id"]]
        phrase_feature = sample["phrase_clip_feature"].float()
        bridge = self.bridge_features.get(sample["phrase"])
        if bridge is not None and len(bridge) > 0 and random.random() > self.bridge_original_prob:
            # 随机替换为 head noun / prompt variant 的 CLIP 特征，复用同一个 SAM3 mask
            phrase_feature = bridge[random.randrange(len(bridge))].float()
        return {
            "patch_tokens": image["patch_tokens"].float(),
            "phrase_clip_feature": phrase_feature,
            "mask_grid": sample["mask_grid"].float(),
            "sample_weight": torch.tensor(
                compute_soft_weight(sample, **self.weight_kwargs) if self.use_sample_weight else 1.0,
                dtype=torch.float32,
            ),
        }


def build_base_projector(config_path, checkpoint_path, device):
    """加载并冻结旧 projector"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    model_class_name = model_cfg.get("model_class", "ProjectionLayer")
    model_cls = getattr(importlib.import_module("src.model"), model_class_name)
    projector = model_cls.from_config(model_cfg)
    projector.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    projector.to(device).eval()
    for param in projector.parameters():
        param.requires_grad = False
    return projector, model_cfg


def move_batch_to_device(batch, device):
    return {
        "patch_tokens": batch["patch_tokens"].to(device, non_blocking=True),
        "phrase_clip_feature": batch["phrase_clip_feature"].to(device, non_blocking=True),
        "mask_grid": batch["mask_grid"].to(device, non_blocking=True),
        "sample_weight": batch["sample_weight"].to(device, non_blocking=True),
    }


class FrozenProjectorPrototypeAdapter(torch.nn.Module):
    """冻结旧 projector，只训练文本 residual adapter"""

    def __init__(self, projector, adapter):
        super().__init__()
        self.projector = projector
        self.adapter = adapter

    def forward(self, phrase_clip_feature, patch_tokens, mask_grid):
        with torch.no_grad():
            z_base = self.projector.project_clip_txt(phrase_clip_feature)
            z_base = F.normalize(z_base.float(), dim=-1)
            pos_proto, neg_proto, valid = masked_region_prototypes(
                patch_tokens,
                mask_grid.view(mask_grid.shape[0], -1),
            )
        z_local = self.adapter(phrase_clip_feature, z_base)
        return z_local, z_base, pos_proto, neg_proto, valid


def distributed_mean(value, device):
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return float(tensor.item())


def run_epoch(model, dataloader, optimizer, device, args, train):
    model.train(train)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.projector.eval()

    totals = {
        "loss": 0.0,
        "nce": 0.0,
        "neg_margin": 0.0,
        "anchor": 0.0,
        "logit_shift": 0.0,
        "inside_shift": 0.0,
        "outside_shift": 0.0,
        "proto_acc": 0.0,
        "base_proto_acc": 0.0,
        "valid_ratio": 0.0,
        "delta_norm": 0.0,
        "sample_weight": 0.0,
    }
    n_batches = 0
    desc = "Train" if train else "Validate"
    iterator = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=False) if is_main_process() else dataloader

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            z_local, z_base, pos_proto, neg_proto, valid = model(
                batch["phrase_clip_feature"],
                batch["patch_tokens"],
                batch["mask_grid"],
            )
            valid_count = int(valid.sum().item())
            if valid_count < 2:
                continue
            z_local = z_local[valid]
            z_base = z_base[valid]
            pos_proto = pos_proto[valid]
            neg_proto = neg_proto[valid]
            sample_weight = batch["sample_weight"][valid]
            loss, parts = prototype_contrastive_loss(
                z_local,
                z_base,
                pos_proto,
                neg_proto,
                sample_weight=sample_weight,
                temperature=args.temperature,
                neg_margin=args.neg_margin,
                neg_weight=args.neg_weight,
                anchor_weight=args.anchor_weight,
            )
            if args.logit_shift_weight > 0:
                patch_tokens = F.normalize(batch["patch_tokens"][valid].float(), dim=-1)
                mask_flat = batch["mask_grid"].view(batch["mask_grid"].shape[0], -1)[valid].float()
                sim_local = torch.einsum("bd,bnd->bn", z_local, patch_tokens)
                sim_base = torch.einsum("bd,bnd->bn", z_base, patch_tokens)
                shift = sim_local - sim_base
                pos_count = mask_flat.sum(dim=1).clamp_min(1.0)
                neg_mask = 1.0 - mask_flat
                neg_count = neg_mask.sum(dim=1).clamp_min(1.0)
                inside_shift = (shift * mask_flat).sum(dim=1) / pos_count
                outside_shift = (shift * neg_mask).sum(dim=1) / neg_count
                # mask 内需要有正向增益，mask 外的平均增益不能一起被抬高
                inside_loss = F.relu(args.inside_shift_margin - inside_shift).mean()
                outside_loss = F.relu(outside_shift - args.outside_shift_margin).pow(2).mean()
                logit_shift_loss = inside_loss + args.outside_shift_weight * outside_loss
                loss = loss + args.logit_shift_weight * logit_shift_loss
                parts["loss"] = loss.detach()
                parts["logit_shift"] = logit_shift_loss.detach()
                parts["inside_shift"] = inside_shift.mean().detach()
                parts["outside_shift"] = outside_shift.mean().detach()
            else:
                parts["logit_shift"] = torch.zeros((), device=device)
                parts["inside_shift"] = torch.zeros((), device=device)
                parts["outside_shift"] = torch.zeros((), device=device)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.adapter.parameters(), args.clip_grad)
            optimizer.step()

        with torch.no_grad():
            pos_sim = (z_local * pos_proto).sum(dim=-1)
            neg_sim = (z_local * neg_proto).sum(dim=-1)
            base_pos_sim = (z_base * pos_proto).sum(dim=-1)
            base_neg_sim = (z_base * neg_proto).sum(dim=-1)
            proto_acc = (pos_sim > neg_sim).float().mean()
            base_proto_acc = (base_pos_sim > base_neg_sim).float().mean()
            delta_norm = (z_local - z_base).norm(dim=-1).mean()

        for key in ["loss", "nce", "neg_margin", "anchor", "logit_shift", "inside_shift", "outside_shift", "sample_weight"]:
            totals[key] += float(parts[key].item())
        totals["proto_acc"] += float(proto_acc.item())
        totals["base_proto_acc"] += float(base_proto_acc.item())
        totals["valid_ratio"] += float(valid.float().mean().item())
        totals["delta_norm"] += float(delta_norm.item())
        n_batches += 1

        if is_main_process() and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{float(parts['loss'].item()):.5f}", acc=f"{float(proto_acc.item()):.4f}")

    metrics = {key: value / max(1, n_batches) for key, value in totals.items()}
    return {key: distributed_mean(value, device) for key, value in metrics.items()}


def save_checkpoint(path, model, args, model_cfg, epoch, best_val):
    raw_model = model.module if isinstance(model, DDP) else model
    out = {
        "adapter_state_dict": raw_model.adapter.adapter.state_dict(),
        "adapter_type": "prototype_text_residual",
        "alpha": raw_model.adapter.beta,
        "beta": raw_model.adapter.beta,
        "clip_dim": raw_model.adapter.clip_dim,
        "dino_dim": raw_model.adapter.dino_dim,
        "hidden_dim": raw_model.adapter.hidden_dim,
        "projector_config": model_cfg,
        "projector_checkpoint": args.projector_checkpoint,
        "epoch": epoch,
        "best_val_loss": best_val,
        "train_pseudo": args.train_pseudo,
        "val_pseudo": args.val_pseudo,
        "filter": {
            "min_area": args.min_area,
            "max_area": args.max_area,
            "min_sam_score": args.min_sam_score,
            "max_instances": args.max_instances,
            "drop_generic_heads": not args.keep_generic_heads,
            "filter_index_json": args.filter_index_json,
            "bridge_feature_cache": args.bridge_feature_cache,
            "bridge_original_prob": args.bridge_original_prob,
            "bridge_val": args.bridge_val,
            "logit_shift_weight": args.logit_shift_weight,
            "inside_shift_margin": args.inside_shift_margin,
            "outside_shift_margin": args.outside_shift_margin,
            "outside_shift_weight": args.outside_shift_weight,
            "use_sample_weight": args.use_sample_weight,
            "weight_min": args.weight_min,
            "weight_score_floor": args.weight_score_floor,
            "weight_score_power": args.weight_score_power,
            "weight_area_center": args.weight_area_center,
            "weight_area_sigma": args.weight_area_sigma,
            "weight_instance_power": args.weight_instance_power,
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(out, path)


def train_prototype_adapter(args):
    set_seed(args.seed)
    distributed = setup_distributed()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if distributed and torch.cuda.is_available():
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"

    log("=== Loading filtered pseudo datasets ===")
    dataset_kwargs = {
        "min_area": args.min_area,
        "max_area": args.max_area,
        "min_sam_score": args.min_sam_score,
        "max_instances": args.max_instances,
        "drop_generic_heads": not args.keep_generic_heads,
        "filter_index_json": args.filter_index_json,
        "use_sample_weight": args.use_sample_weight,
        "weight_min": args.weight_min,
        "weight_score_floor": args.weight_score_floor,
        "weight_score_power": args.weight_score_power,
        "weight_area_center": args.weight_area_center,
        "weight_area_sigma": args.weight_area_sigma,
        "weight_instance_power": args.weight_instance_power,
    }
    train_dataset = FilteredSam3PrototypeDataset(
        args.train_pseudo,
        filter_split=args.train_filter_split,
        bridge_feature_cache=args.bridge_feature_cache,
        bridge_original_prob=args.bridge_original_prob,
        max_samples=args.max_train_samples,
        **dataset_kwargs,
    )
    val_dataset = FilteredSam3PrototypeDataset(
        args.val_pseudo,
        filter_split=args.val_filter_split,
        bridge_feature_cache=args.bridge_feature_cache if args.bridge_val else None,
        bridge_original_prob=args.bridge_original_prob,
        max_samples=args.max_val_samples,
        **dataset_kwargs,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False) if distributed else None
    pin_memory = str(device).startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
        drop_last=False,
    )

    log("=== Loading frozen projector ===")
    projector, model_cfg = build_base_projector(args.projector_config, args.projector_checkpoint, device)
    adapter = ResidualTextPrototypeAdapter(
        clip_dim=args.clip_dim,
        dino_dim=model_cfg.get("dino_embed_dim", args.dino_dim),
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        zero_init=not args.no_zero_init,
    )
    model = FrozenProjectorPrototypeAdapter(projector, adapter).to(device)
    if distributed:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(raw_model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log("=== Training prototype adapter ===")
    log(f"train samples after filter: {len(train_dataset)}")
    log(f"val samples after filter: {len(val_dataset)}")
    log(f"world size: {get_world_size()}")
    log(f"beta: {args.beta}")
    log(f"anchor weight: {args.anchor_weight}")
    log(f"bridge cache: {args.bridge_feature_cache or '<none>'}")
    log(f"bridge original prob: {args.bridge_original_prob}")
    log(f"logit shift weight: {args.logit_shift_weight}")
    log(f"sample weighting: {args.use_sample_weight}")
    log(f"save path: {args.out_path}")

    best_val = None
    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        log(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        log("train " + " ".join(f"{key}={value:.6f}" for key, value in train_metrics.items()))
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False)
        log("val   " + " ".join(f"{key}={value:.6f}" for key, value in val_metrics.items()))

        val_loss = val_metrics["loss"]
        should_save = not args.save_best_only or best_val is None or val_loss < best_val
        if should_save:
            best_val = val_loss
            if is_main_process():
                save_checkpoint(args.out_path, model, args, model_cfg, epoch + 1, best_val)
                log(f"checkpoint saved: {args.out_path}")

    if distributed:
        dist.barrier()
    cleanup_distributed()


def build_argparser():
    parser = argparse.ArgumentParser(description="Train SAM3 region-prototype text adapter")
    parser.add_argument("--train_pseudo", required=True)
    parser.add_argument("--val_pseudo", required=True)
    parser.add_argument("--projector_config", required=True)
    parser.add_argument("--projector_checkpoint", required=True)
    parser.add_argument("--out_path", default="weights/prototype_adapter.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--save_best_only", action="store_true")
    parser.add_argument("--clip_dim", type=int, default=512)
    parser.add_argument("--dino_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--no_zero_init", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--neg_margin", type=float, default=0.2)
    parser.add_argument("--neg_weight", type=float, default=0.5)
    parser.add_argument("--anchor_weight", type=float, default=10.0)
    parser.add_argument("--min_area", type=float, default=0.01)
    parser.add_argument("--max_area", type=float, default=0.55)
    parser.add_argument("--min_sam_score", type=float, default=0.70)
    parser.add_argument("--max_instances", type=int, default=3)
    parser.add_argument("--keep_generic_heads", action="store_true")
    parser.add_argument("--filter_index_json", default=None)
    parser.add_argument("--train_filter_split", default="train")
    parser.add_argument("--val_filter_split", default="val")
    parser.add_argument("--bridge_feature_cache", default=None)
    parser.add_argument("--bridge_original_prob", type=float, default=0.5)
    parser.add_argument("--bridge_val", action="store_true")
    parser.add_argument("--logit_shift_weight", type=float, default=0.0)
    parser.add_argument("--inside_shift_margin", type=float, default=0.02)
    parser.add_argument("--outside_shift_margin", type=float, default=0.0)
    parser.add_argument("--outside_shift_weight", type=float, default=5.0)
    parser.add_argument("--use_sample_weight", action="store_true")
    parser.add_argument("--weight_min", type=float, default=0.20)
    parser.add_argument("--weight_score_floor", type=float, default=0.60)
    parser.add_argument("--weight_score_power", type=float, default=1.0)
    parser.add_argument("--weight_area_center", type=float, default=0.16)
    parser.add_argument("--weight_area_sigma", type=float, default=1.0)
    parser.add_argument("--weight_instance_power", type=float, default=0.5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    return parser


def main():
    args = build_argparser().parse_args()
    train_prototype_adapter(args)


if __name__ == "__main__":
    main()
