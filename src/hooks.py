"""DINO 和 CLIP 特征的前向钩子与词元聚合工具"""

import torch

# -------------------------------
# 全局共享临时缓存
# -------------------------------
# 用于存储前向钩子捕获到的中间特征
feats = {}


# -------------------------------
# DINO 注意力钩子
# -------------------------------
def get_self_attention(module, input, output):
    """
    前向钩子：捕获 DINO 模型某注意力模块的 qkv 输出
    
    参数:
        module: 钩子注册的模块（attention 层）
        input: 前向输入
        output: 前向输出 qkv
    存储:
        feats["self_attn"] = output
    """
    feats["self_attn"] = output


def process_self_attention(
    output,
    batch_size,
    num_tokens,
    num_attn_heads,
    embed_dim,
    scale,
    num_global_tokens,
    ret_self_attn_maps=False,
):
    """
    将 DINO qkv 输出转换为 CLS token 到各 patch 的注意力权重
    
    参数:
        output: 钩子捕获的 qkv 输出
        batch_size: batch 大小
        num_tokens: token 总数（包含全局 token + patch token）
        num_attn_heads: 注意力头数量
        embed_dim: embedding 维度
        scale: q 缩放因子
        num_global_tokens: 全局 token 数量（例如 CLS token）
        ret_self_attn_maps: 是否返回原始注意力图
    
    返回:
        self_attn: batch 平均后的 CLS 到 patch token 注意力权重 [B, num_patch]
        self_attn_maps (可选): 原始每头注意力图 [B, heads, num_tokens, num_tokens]
    
    说明:
        - output 形状 [B, num_tokens, 3*embed_dim] (qkv拼接)
        - 将 qkv reshape 为 [3, B, heads, num_tokens, head_dim]
        - 取 CLS token 注意力对 patch token 的映射
        - self_attn 是对所有 heads 求平均并 softmax
    """
    qkv = output.reshape(
        batch_size,
        num_tokens,
        3,
        num_attn_heads,
        embed_dim // num_attn_heads,
    ).permute(2, 0, 3, 1, 4)

    q, k = qkv[0] * scale, qkv[1]
    attn = q @ k.transpose(-2, -1)
    self_attn_maps = attn[:, :, 0, num_global_tokens:]  # CLS -> patch
    self_attn = self_attn_maps.mean(dim=1).softmax(dim=-1)

    if ret_self_attn_maps:
        return self_attn, self_attn_maps
    return self_attn


# -------------------------------
# ViT 输出钩子
# -------------------------------
def get_vit_out(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    """保存 ViT 前向输出，用于后续分析或聚合"""
    feats["vit_out"] = output


# -------------------------------
# CLIP 文本特征钩子
# -------------------------------
def get_all_out_tokens(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    """保存 CLIP 最后一层词元输出 (token-level)"""
    feats["clip_txt_out_tokens"] = output


def get_clip_second_last_dense_out(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    """
    保存 CLIP 倒数第二层 transformer 输出
    
    输出转置为批次优先格式 [B, L, D]，方便 batch 处理
    """
    feats["clip_second_last_out"] = output.permute(1, 0, 2)


# -------------------------------
# CLIP token embedding 聚合工具
# -------------------------------
def average_text_tokens(text_embeddings, mask, keep_cls=False, keep_end_seq=False):
    """
    对有效 token embedding 求平均，可选择排除 CLS/EOS token
    
    参数:
        text_embeddings: [B, L, D] token embedding
        mask: [B, L] bool mask，标记有效 token
        keep_cls: 是否保留 CLS token
        keep_end_seq: 是否保留 EOS token
    
    返回:
        [B, D] 平均后的文本向量
    
    说明:
        - mask[:, 0] 对应 CLS token
        - mask[:, mask.sum(dim=1)-1] 对应 EOS token
        - masked_embeddings 对无效 token 置 0，再按有效数量平均
    """
    if not keep_end_seq:
        mask[torch.arange(mask.shape[0]), mask.sum(dim=1) - 1] = False
    if not keep_cls:
        mask[:, 0] = False

    masked_embeddings = text_embeddings * mask.unsqueeze(-1)
    sum_embeddings = masked_embeddings.sum(dim=1)
    valid_elements = mask.sum(dim=1, keepdim=True)
    return sum_embeddings / valid_elements