"""投影层训练、验证和学习率调度工具（支持单卡 / 多卡 DDP）"""

import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import wandb as wandb_module
except ImportError:
    wandb_module = None

from src.loss import ContrastiveLoss


# -------------------------------
# 分布式训练工具
# -------------------------------

def is_dist_avail_and_initialized():
    """判断当前是否处于分布式训练环境"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """初始化分布式进程组（由 torchrun 自动设置环境变量）"""
    if "RANK" not in os.environ:
        # 非分布式模式，跳过
        return False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    return True


def cleanup_distributed():
    """销毁分布式进程组"""
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


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


def set_seed(seed):
    """设置随机种子，尽量保证训练可复现"""
    log_step("Setting random seed")
    log_kv("seed", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关闭 cuDNN 自动优化，减少随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 控制 Python hash 和 CUDA 矩阵运算的确定性
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    log_success("Seed configured")


def assign_learning_rate(optimizer, new_lr):
    """更新优化器中所有参数组的学习率"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    """线性 warmup 学习率"""
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    """构造先 warmup、再保持常量学习率的调度器"""
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    """构造先 warmup、再余弦衰减的学习率调度器"""
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            elapsed_steps = step - warmup_length
            decay_steps = max(1, steps - warmup_length)
            lr = 0.5 * (1 + np.cos(np.pi * elapsed_steps / decay_steps)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def _optional_to_device(batch, key, device):
    """若 batch 中存在某个可选字段，则移动到指定设备"""
    if key in batch:
        return batch[key].to(device)
    return None


def _distributed_concat_with_grad(tensor):
    """跨进程拼接 tensor，并保留当前 rank 的梯度"""
    if tensor is None or not is_dist_avail_and_initialized():
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    gathered[get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def _distributed_mean(value, device):
    """对所有 rank 的标量取平均"""
    if not is_dist_avail_and_initialized():
        return float(value)

    value_tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    value_tensor /= get_world_size()
    return float(value_tensor.item())


def _gather_contrastive_inputs(
    annotations,
    images,
    self_attn_maps,
    cls,
    text_input_mask,
    text_argmax,
):
    """DDP 下把各卡 batch 拼成全局 batch，用于全局负样本"""
    if not is_dist_avail_and_initialized():
        return annotations, images, self_attn_maps, cls, text_input_mask, text_argmax

    annotations = _distributed_concat_with_grad(annotations)
    images = _distributed_concat_with_grad(images)
    self_attn_maps = _distributed_concat_with_grad(self_attn_maps)
    cls = _distributed_concat_with_grad(cls)

    if text_input_mask is not None:
        text_input_mask = _distributed_concat_with_grad(text_input_mask)
    if text_argmax is not None:
        text_argmax = _distributed_concat_with_grad(text_argmax)

    return annotations, images, self_attn_maps, cls, text_input_mask, text_argmax


def _prepare_contrastive_inputs(batch, device, dino_model=None):
    """从 DataLoader 返回的 batch 中整理 ContrastiveLoss 所需输入"""
    annotations = batch["annotation"].to(device, dtype=torch.float32)
    images = batch["image"].to(device)

    text_argmax = _optional_to_device(batch, "text_argmax", device)
    text_input_mask = _optional_to_device(batch, "text_input_mask", device)

    if dino_model is not None:
        with torch.no_grad():
            dino_out = dino_model(images, is_training=True)
            images = dino_out["x_norm_patchtokens"]  # [B, N, D]
        self_attn_maps = None
        cls = None
    elif "self_attn_maps" in batch:
        # patch-token / attention-head weighting 模式下需要额外视觉信息
        self_attn_maps = batch["self_attn_maps"].to(device)
        cls = batch["dino_features"].to(device)
    else:
        self_attn_maps = None
        cls = None

    return annotations, images, self_attn_maps, cls, text_input_mask, text_argmax


def _save_head_activations(path, head_attivations, ann_ids, img_ids):
    """保存最后一个 epoch 中选中的注意力 head 或 patch index 信息"""
    head_attivations = torch.cat(head_attivations)
    ann_ids = torch.cat(ann_ids)
    img_ids = torch.cat(img_ids)

    act_dict = {}
    for activation, annotation_id, image_id in zip(head_attivations, ann_ids, img_ids):
        act_dict[annotation_id.item()] = {
            "image_id": image_id.item(),
            "activation_head": activation.item(),
        }

    with open(path, "w") as f:
        json.dump(act_dict, f)
    log_success(f"Activation heads saved: {path}")


def train(
    model,
    train_dataloader,
    contrastive_loss,
    optimizer,
    scheduler=None,
    wandb=False,
    save_head_attivations=None,
    n_epochs=0,
    dino_model=None,
):
    """训练一个 epoch，并返回批次 loss 均值"""
    train_batch_losses = []
    device = next(model.parameters()).device
    prev_iter = n_epochs * len(train_dataloader)

    head_attivations = []
    ann_ids = []
    img_ids = []

    if is_main_process():
        iterator = tqdm(
            train_dataloader,
            desc=_color(f"Train epoch {n_epochs + 1}", Console.CYAN),
            dynamic_ncols=True,
            leave=False,
        )
    else:
        iterator = train_dataloader

    for n_batch, batch in enumerate(iterator):
        annotations, images, self_attn_maps, cls, text_input_mask, text_argmax = (
            _prepare_contrastive_inputs(batch, device, dino_model=dino_model)
        )
        annotations, images, self_attn_maps, cls, text_input_mask, text_argmax = (
            _gather_contrastive_inputs(
                annotations,
                images,
                self_attn_maps,
                cls,
                text_input_mask,
                text_argmax,
            )
        )

        if scheduler is not None:
            scheduler(n_batch + prev_iter)

        if not save_head_attivations:
            loss = contrastive_loss(
                images,
                annotations,
                return_similarity_mat=False,
                self_attn_maps=self_attn_maps,
                cls=cls,
                text_input_mask=text_input_mask,
                text_argmax=text_argmax,
            )
        else:
            loss, batch_head_attivations = contrastive_loss(
                images,
                annotations,
                return_similarity_mat=False,
                self_attn_maps=self_attn_maps,
                cls=cls,
                text_input_mask=text_input_mask,
                text_argmax=text_argmax,
                return_index=True,
            )
            head_attivations.append(batch_head_attivations)
            ann_ids.append(batch["metadata"]["annotation_id"])
            img_ids.append(batch["metadata"]["image_id"])

        train_batch_losses.append(loss.item())
        if is_main_process() and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if wandb and wandb_module is not None:
            wandb_module.log({"train_loss": loss.item()})

    if save_head_attivations is not None and is_main_process():
        _save_head_activations(save_head_attivations, head_attivations, ann_ids, img_ids)

    local_mean = torch.mean(torch.tensor(train_batch_losses)).item()
    return _distributed_mean(local_mean, device)


def validate(model, val_dataloader, contrastive_loss, verbose=False, dino_model=None):
    """在验证集上计算平均 contrastive loss"""
    device = next(model.parameters()).device
    val_batch_losses = []

    val_iterator = tqdm(
        val_dataloader,
        desc=_color("Validate", Console.MAGENTA),
        dynamic_ncols=True,
        leave=False,
    ) if verbose else val_dataloader

    for batch in val_iterator:
        annotations, images, self_attn_maps, cls, text_input_mask, text_argmax = (
            _prepare_contrastive_inputs(batch, device, dino_model=dino_model)
        )
        annotations, images, self_attn_maps, cls, text_input_mask, text_argmax = (
            _gather_contrastive_inputs(
                annotations,
                images,
                self_attn_maps,
                cls,
                text_input_mask,
                text_argmax,
            )
        )

        with torch.no_grad():
            loss = contrastive_loss(
                images,
                annotations,
                return_similarity_mat=False,
                self_attn_maps=self_attn_maps,
                cls=cls,
                text_input_mask=text_input_mask,
                text_argmax=text_argmax,
            )

        val_batch_losses.append(loss.item())
        if verbose:
            val_iterator.set_postfix(loss=f"{loss.item():.5f}")

    local_mean = torch.mean(torch.tensor(val_batch_losses)).item()
    return _distributed_mean(local_mean, device)


def _build_optimizer(model, optimizer_name, lr, weight_decay):
    """根据名称创建优化器"""
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Optimizer {optimizer_name} not implemented")


def _build_scheduler(optimizer, scheduler_name, lr, warmup, total_steps):
    """创建训练中使用的 step-level 学习率调度器"""
    if scheduler_name == "linear" and warmup == 0:
        return None
    if scheduler_name == "linear" and warmup > 0:
        return const_lr(optimizer, lr, warmup, total_steps)
    if scheduler_name == "cosine":
        return cosine_lr(optimizer, lr, warmup, total_steps)
    return None


def _format_param_count(model):
    n_params = sum(p.numel() for p in model.parameters())
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.2f}M"
    if n_params >= 1_000:
        return f"{n_params / 1_000:.2f}K"
    return str(n_params)


def do_train(
    model,
    train_dataset,
    val_dataset,
    train_cfg,
    seed=123,
    optimizer_name="Adam",
    weight_decay=0.05,
    scheduler_name="linear",
    warmup=0,
    save_head_attivations=None,
    dino_model=None,
):
    """执行完整训练/验证循环，并返回选中的模型和 loss 记录（支持单卡/多卡 DDP）"""
    set_seed(seed)

    lr = train_cfg["lr"]
    ltype = train_cfg["ltype"]
    num_epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]

    margin = train_cfg.get("margin", 0.2)
    max_violation = train_cfg.get("max_violation", True)
    shuffle = train_cfg.get("shuffle", True)
    save_best_model = train_cfg.get("save_best_model", True)
    num_workers = train_cfg.get("num_workers", 8)

    distributed = is_dist_avail_and_initialized()
    world_size = get_world_size()
    rank = get_rank()

    if distributed and save_head_attivations is not None:
        if is_main_process():
            log_warning("save_head_activations 暂不支持 DDP，已在本次多卡训练中禁用")
        save_head_attivations = None

    if is_main_process():
        log_step("Building dataloaders")
        log_kv("distributed", distributed)
        log_kv("world size", world_size)
        log_kv("train samples", len(train_dataset))
        log_kv("val samples", len(val_dataset))
        log_kv("batch size (per GPU)", batch_size)
        log_kv("effective batch size", batch_size * world_size)
        log_kv("num workers", num_workers)
        log_kv("shuffle train", shuffle)

    # 分布式模式使用 DistributedSampler
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(shuffle and train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=torch.cuda.is_available(),
    )

    # 使用 DDP 包装模型
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    if is_main_process():
        log_step("Preparing training objective")
        log_kv("loss", ltype)
        log_kv("optimizer", optimizer_name)
        log_kv("lr", lr)
        log_kv("weight decay", weight_decay)
        log_kv("scheduler", scheduler_name)
        log_kv("warmup steps", warmup)
        log_kv("max violation", max_violation)
        log_kv("parameters", _format_param_count(model))

    # ContrastiveLoss 中使用的 sim 函数需要指向底层模型
    raw_model = model.module if distributed else model
    criterion = ContrastiveLoss(raw_model, margin=margin, max_violation=max_violation, ltype=ltype)
    optimizer = _build_optimizer(model, optimizer_name, lr, weight_decay)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = _build_scheduler(optimizer, scheduler_name, lr, warmup, total_steps)

    train_losses = torch.zeros(num_epochs)
    val_losses = torch.zeros(num_epochs)
    best_model_state = None
    best_val_loss = None

    if is_main_process():
        log_step("Start training")
        log_kv("epochs", num_epochs)
        log_kv("steps / epoch", len(train_dataloader))
        log_kv("total steps", total_steps)

    for epoch in range(num_epochs):
        # 分布式模式下每个 epoch 需要设置 sampler 的 epoch 以变换采样顺序
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            epoch_title = f"Epoch {epoch + 1}/{num_epochs}"
            print(_color(f"\n━━━━━━━━━━━━━━━━ {epoch_title} ━━━━━━━━━━━━━━━━", Console.BOLD + Console.BLUE))

        model.train()
        train_loss = train(
            model,
            train_dataloader,
            criterion,
            optimizer,
            scheduler,
            save_head_attivations=None if epoch < num_epochs - 1 else save_head_attivations,
            n_epochs=epoch,
            dino_model=dino_model,
        )
        train_losses[epoch] = train_loss

        model.eval()
        if is_main_process():
            log_info("Running validation...")
        val_loss = validate(model, val_dataloader, criterion, verbose=is_main_process(), dino_model=dino_model)
        val_losses[epoch] = val_loss

        is_best = save_best_model and (best_val_loss is None or val_loss < best_val_loss)
        if is_best:
            best_val_loss = val_loss
            best_model_state = deepcopy(raw_model.state_dict())

        if is_main_process():
            status = _color("best", Console.GREEN) if is_best else ""
            print(
                f"  "
                f"train_loss={_color(f'{train_loss:.6f}', Console.CYAN)}  "
                f"val_loss={_color(f'{val_loss:.6f}', Console.MAGENTA)}  "
                f"{status}"
            )

    # 还原最佳模型到 raw_model
    if save_best_model and best_model_state is not None:
        raw_model.load_state_dict(best_model_state)

    if distributed:
        dist.barrier()

    if is_main_process():
        log_success("Training finished")
        if best_val_loss is not None:
            log_kv("best val loss", f"{best_val_loss:.6f}")

    return raw_model, train_losses, val_losses
