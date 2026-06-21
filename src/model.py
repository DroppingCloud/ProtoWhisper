"""将 CLIP 文本特征映射到 DINO 空间的投影层定义"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.hooks import feats, get_self_attention, process_self_attention

# 训练/推理时可能间接加载 DINOv2，屏蔽 xFormers 缺失提示，避免日志被 warning 淹没
warnings.filterwarnings("ignore", message=r".*xFormers is not available.*", category=UserWarning)


class ProjectionLayer(nn.Module):
    """
    CLIP 文本特征到 DINO 视觉空间的投影层

    核心作用：
    1. 将 CLIP text embedding 投影到 DINO embedding 维度；
    2. 可选地对 DINO patch token 进行注意力聚合；
    3. 根据不同 alignment_strategy 计算图文相似度
    """

    def __init__(
        self,
        act=nn.Tanh(),
        hidden_layer=False,
        cosine=True,
        dino_embed_dim=1024,
        clip_embed_dim=512,
        num_attn_head=16,
        weight_attn_heads=None,
        alignment_strategy="max_score",
        alpha=0.6,
        keep_cls=False,
        keep_end_seq=False,
    ):
        super().__init__()

        self.num_attn_head = num_attn_head

        # 最基本的投影层：CLIP 文本维度 -> DINO 视觉维度
        self.linear_layer = nn.Linear(clip_embed_dim, dino_embed_dim)

        # 可选隐藏层，用于增强投影头表达能力
        if hidden_layer:
            # 兼容旧配置：True 表示 1 层隐藏层；整数表示多层
            hidden_layer = 1 if hidden_layer is True else hidden_layer
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(dino_embed_dim, dino_embed_dim) for _ in range(hidden_layer)]
            )

        self.act = act
        self.cosine = cosine

        # 是否对 DINO 多头注意力图进行加权聚合：
        # None / static / conditioned
        self.weight_attn_heads = weight_attn_heads

        # 图文 token 对齐策略，如 max_score、weighted_avg、sum、mean 等
        self.alignment_strategy = alignment_strategy

        # token-level 文本对齐时是否保留 CLS/EOS token
        self.keep_cls = keep_cls
        self.keep_end_seq = keep_end_seq

        # nucleus-sampling 策略中的累计阈值
        self.alpha = alpha

        self.__init_attention_head_weights(dino_embed_dim)

    def __init_attention_head_weights(self, dino_embed_dim):
        """初始化多头注意力加权模块"""
        if self.weight_attn_heads == "static":
            # 静态策略：为每个 attention head 学习一个全局权重
            self.attn_weights = nn.Parameter(torch.rand(self.num_attn_head))

        elif self.weight_attn_heads == "conditioned":
            # 条件策略：根据当前图像 CLS 特征动态预测 head 权重
            self.weight_layer1 = nn.Linear(dino_embed_dim, dino_embed_dim)
            self.weight_layer2 = nn.Linear(dino_embed_dim, self.num_attn_head)

    @classmethod
    def from_config(cls, config):
        """
        从配置字典或 YAML 文件构建 ProjectionLayer

        config 可以是：
        1. dict：直接读取模型配置；
        2. str：YAML 文件路径，读取其中的 ["model"] 字段
        """
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)["model"]

        model = cls(
            act=cls.__build_activation(config.get("act", None)),
            hidden_layer=config.get("hidden_layer", False),
            cosine=config.get("cosine", True),
            dino_embed_dim=config.get("dino_embed_dim", 1024),
            num_attn_head=config.get("num_attn_head", 16),
            clip_embed_dim=config.get("clip_embed_dim", 512),
            weight_attn_heads=config.get("weight_attn_heads", None),
            alignment_strategy=config.get("alignment_strategy", "max_score"),
            alpha=config.get("alpha", 0.6),
            keep_cls=config.get("keep_cls", None),
            keep_end_seq=config.get("keep_end_seq", None),
        )

        # 可选：从已有 checkpoint 初始化
        if config.get("starting_checkpoint", None) is not None:
            model.load_state_dict(torch.load(config["starting_checkpoint"], "cpu"))

        return model

    @staticmethod
    def __build_activation(act):
        """根据配置字符串构建激活函数"""
        if act == "tanh":
            return nn.Tanh()
        if act == "relu":
            return nn.ReLU()
        if act == "sigmoid":
            return nn.Sigmoid()
        if act is not None:
            raise Exception("Unknown activation function")
        return None

    def compute_similarity(
        self,
        visual_embedding,
        textual_embedding,
        text_input_mask=None,
        return_index=False,
    ):
        """
        计算图文相似度

        支持两类输入：
        1. 向量级特征：
            visual_embedding: [B, D]
            textual_embedding: [B, D]

        2. token 序列特征：
            visual_embedding: [B, N, D]
            textual_embedding: [B, L, D] 或 [B, D]

        返回：
            相似度矩阵 [B, B]，其中第 i 行第 j 列表示 image_i 与 text_j 的相似度
        """
        if len(visual_embedding.shape) == 3 or len(textual_embedding.shape) == 3:
            return self.__compute_sequence_similarity(
                visual_embedding,
                textual_embedding,
                text_input_mask,
                return_index,
            )

        # 向量级相似度：文本特征与视觉特征矩阵乘法
        sims = textual_embedding @ visual_embedding.transpose(1, 0)

        if not return_index:
            return sims

        # 注意：这里原代码返回 index，但 index 未定义
        # 正常向量级相似度下不应开启 return_index
        return sims, None

    def __compute_sequence_similarity(
        self,
        visual_embedding,
        textual_embedding,
        text_input_mask,
        return_index,
    ):
        """
        处理 token 序列形式的图文对齐

        主要场景：
        - 图像侧是 patch tokens: [B, N, D]
        - 文本侧是句子向量或 token 序列
        """
        index = None

        if self.alignment_strategy == "weighted_avg":
            # 根据文本向量对每个视觉 patch 的相似度，对视觉 token 加权平均
            visual_embedding, textual_embedding = self.__require_visual_tokens_and_text_vectors(
                visual_embedding,
                textual_embedding,
            )
            sims = torch.einsum("ik,ijk->ij", textual_embedding, visual_embedding).softmax(dim=-1)
            visual_embedding = (visual_embedding * sims.unsqueeze(dim=-1)).mean(dim=1)
            sims = textual_embedding @ visual_embedding.transpose(1, 0)

        elif self.alignment_strategy == "sampled_attn_map":
            # 按文本-patch 相似度分布随机采样一个 patch 作为视觉表示
            visual_embedding, textual_embedding = self.__require_visual_tokens_and_text_vectors(
                visual_embedding,
                textual_embedding,
            )
            sims = torch.einsum("ik,ijk->ij", textual_embedding, visual_embedding).softmax(dim=-1)
            index = torch.multinomial(sims, 1).view(-1, 1, 1).expand(-1, 1, visual_embedding.shape[-1])
            visual_embedding = torch.gather(visual_embedding, 1, index).squeeze(1)
            sims = textual_embedding @ visual_embedding.transpose(1, 0)

        elif self.alignment_strategy == "max_score":
            # 选择与文本最相似的 patch token 作为当前图像表示
            sims = torch.einsum("ik,ijk->ij", textual_embedding, visual_embedding).softmax(dim=-1)
            index = sims.argmax(dim=-1)
            index_reshaped = index.view(-1, 1, 1).expand(-1, 1, visual_embedding.shape[-1])
            visual_embedding = torch.gather(visual_embedding, 1, index_reshaped).squeeze(1)
            sims = textual_embedding @ visual_embedding.transpose(1, 0)

        else:
            # 其他策略直接计算完整的 patch-token / word-token 对齐矩阵
            sims = self.__compute_full_token_alignment(
                visual_embedding,
                textual_embedding,
                text_input_mask,
            )

        if return_index:
            return sims, index
        return sims

    @staticmethod
    def __require_visual_tokens_and_text_vectors(visual_embedding, textual_embedding):
        """
        校验某些 alignment strategy 的输入格式

        这些策略要求：
        - visual_embedding 是视觉 token 序列 [B, N, D]
        - textual_embedding 是文本全局向量 [B, D]
        """
        if len(visual_embedding.shape) != 3 or len(textual_embedding.shape) != 2:
            raise Exception("Alignment strategy not implemented for this type of embeddings!")
        return visual_embedding, textual_embedding

    def __compute_full_token_alignment(
        self,
        visual_embedding,
        textual_embedding,
        text_input_mask,
    ):
        """
        计算完整 token-to-token 对齐相似度

        visual_embedding:
            [B_img, N, D]

        textual_embedding:
            [B_txt, L, D]

        输出：
            [B_img, B_txt] 相似度矩阵
        """
        # 若输入是全局向量 [B, D]，统一扩展成长度为 1 的 token 序列
        textual_embedding = (
            textual_embedding.unsqueeze(1)
            if len(textual_embedding.shape) == 2
            else textual_embedding
        )
        visual_embedding = (
            visual_embedding.unsqueeze(1)
            if len(visual_embedding.shape) == 2
            else visual_embedding
        )

        # 使用文本 token 序列时，需要 mask 去掉 padding / 可选去掉 CLS、EOS
        if textual_embedding.shape[1] > 1:
            assert text_input_mask is not None, "If we use all the textual embeddings, we need the input mask"
            self.__apply_text_token_mask(text_input_mask)

        im_set_batch = visual_embedding.size(0)
        im_set_len = visual_embedding.size(1)
        s_seq_batch = textual_embedding.size(0)
        s_seq_len = textual_embedding.size(1)

        # 构造所有 image-text 组合：
        # im_set: [B_img, B_txt, N, D]
        # s_seq:  [B_img, B_txt, L, D]
        im_set = visual_embedding.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)
        s_seq = textual_embedding.unsqueeze(0).expand(im_set_batch, -1, -1, -1)

        # token-to-token 相似度：
        # [B_img, B_txt, N, L]
        alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))

        # 将无效文本 token 的相似度置零
        if text_input_mask is not None:
            alignment_mask = text_input_mask.unsqueeze(1).unsqueeze(0).expand(
                im_set_batch,
                -1,
                im_set_len,
                -1,
            ).logical_not()
            alignments.masked_fill_(alignment_mask, value=0)

        return self.__reduce_alignments(alignments, s_seq_batch, s_seq_len)

    def __apply_text_token_mask(self, text_input_mask):
        """
        原地修改文本 token mask

        默认去掉：
        - EOS token；
        - CLS/SOS token

        注意：
            这里会直接修改传入的 text_input_mask
        """
        if not self.keep_end_seq:
            text_input_mask[
                torch.arange(text_input_mask.shape[0]),
                torch.sum(text_input_mask, dim=1) - 1,
            ] = False

        if not self.keep_cls:
            text_input_mask[:, 0] = False

    def __reduce_alignments(self, alignments, s_seq_batch, s_seq_len):
        """
        将 token-to-token 对齐矩阵归约为 image-text 相似度

        alignments:
            [B_img, B_txt, N_visual, L_text]
        """
        if self.alignment_strategy == "sum":
            return alignments.sum(dim=(2, 3))

        if self.alignment_strategy == "mean":
            return alignments.mean(dim=(2, 3))

        if self.alignment_strategy == "max-row_sum":
            # 对每个文本 token，取最匹配的视觉 token，再对文本 token 求和
            return alignments.max(2)[0].sum(2)

        if self.alignment_strategy == "nucleus-sampling":
            return self.__nucleus_sampling_similarity(alignments, s_seq_batch, s_seq_len)

        return None

    def __nucleus_sampling_similarity(self, alignments, s_seq_batch, s_seq_len):
        """
        nucleus-sampling 相似度归约

        思路：
        1. 对每个文本 token，取最大视觉 token 对齐分数；
        2. 按分数从高到低排序；
        3. 归一化后累计到 alpha 阈值；
        4. 只保留累计贡献靠前的 token 相似度
        """
        max_alignments = alignments.max(2)[0]
        sorted_alignments = max_alignments.sort(dim=2, descending=True)[0]

        mins = sorted_alignments.min(2)[0].unsqueeze(-1).expand(-1, -1, s_seq_len)
        maxs = sorted_alignments.max(2)[0].unsqueeze(-1).expand(-1, -1, s_seq_len)

        norm_alignments = (sorted_alignments - mins) / (maxs - mins)

        sums = norm_alignments.sum(dim=-1).unsqueeze(-1).expand(-1, -1, s_seq_len)
        norm_alignments = norm_alignments / sums

        cumsums = norm_alignments.cumsum(2)
        indices = torch.argmax((cumsums > self.alpha).int() + 1, dim=2)

        mask = (
            torch.arange(s_seq_len)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(s_seq_batch, s_seq_batch, s_seq_len)
            .to(indices.device)
            < indices.unsqueeze(-1).expand(-1, -1, s_seq_len) + 1
        )

        relevant_alignments = sorted_alignments * mask
        return relevant_alignments.sum(dim=2)

    def forward(
        self,
        visual_embedding,
        textual_embedding,
        ret_similarity_matrix=True,
        ret_embeds=False,
        self_attn_maps=None,
        cls=None,
        text_input_mask=None,
        return_index=False,
    ):
        """
        前向过程

        输入：
            visual_embedding:
                DINO 图像特征，可以是 [B, D] 或 [B, N, D]

            textual_embedding:
                CLIP 文本特征，可以是 [B, D] 或 [B, L, D]

        返回：
            默认返回图文相似度矩阵 [B, B]
        """
        # 如果启用 attention head weighting，先将 patch tokens 聚合成视觉向量
        if self.weight_attn_heads is not None:
            assert self_attn_maps is not None, (
                "In case we have attention maps weights, "
                "we have to weight patch tokens mean by the weighted self-attention maps"
            )
            visual_embedding = self.get_visual_embed(
                visual_embedding,
                self_attn_maps=self_attn_maps,
                cls=cls,
            )

        # 将 CLIP 文本特征投影到 DINO 空间
        textual_embedding = self.project_clip_txt(textual_embedding)

        # 可选 L2 归一化，使点积等价于余弦相似度
        if self.cosine:
            textual_embedding = F.normalize(textual_embedding, p=2, dim=-1)
            visual_embedding = F.normalize(visual_embedding, p=2, dim=-1)

        if ret_embeds:
            return textual_embedding, visual_embedding

        # 计算图文相似度
        similarity_result = self.compute_similarity(
            visual_embedding,
            textual_embedding,
            text_input_mask,
            return_index,
        )

        if return_index:
            similarity, index = similarity_result
        else:
            similarity = similarity_result

        # 若不返回完整矩阵，只返回 batch 内正样本对的相似度，即对角线
        if not ret_similarity_matrix:
            mask = torch.eye(len(similarity), device=similarity.device).bool()
            similarity = similarity[mask]

        if return_index:
            return similarity, index
        return similarity

    def get_visual_embed(self, visual_embedding, self_attn_maps=None, cls=None):
        """
        根据 self-attention map 聚合 DINO patch tokens

        输入：
            visual_embedding: [B, N, D]
            self_attn_maps: [B, H, N]
            cls: [B, D]，仅 conditioned 策略需要

        输出：
            聚合后的视觉向量 [B, D]
        """
        if self_attn_maps is None:
            return visual_embedding

        assert len(visual_embedding.shape) == 3, (
            "In case we have attention maps weights, the visual_embedding "
            "should contain patch embeddings, with shape BS x NUM_PATCHES x EMBED_DIM"
        )

        if self.weight_attn_heads == "conditioned":
            # 根据 CLS 特征动态预测每个 attention head 的权重
            assert cls is not None, "cls must be setted in case of dinamic attention weighting"
            attn_logits = self.weight_layer2(self.act(self.weight_layer1(cls)))
            normalized_attn_weights = attn_logits.softmax(dim=1)

            self_attn = (
                self_attn_maps * normalized_attn_weights.unsqueeze(dim=-1)
            ).mean(dim=1)

        else:
            # 使用全局可学习 head 权重
            normalized_attn_weights = self.attn_weights.softmax(dim=0)
            self_attn = (
                self_attn_maps
                * normalized_attn_weights.view(1, normalized_attn_weights.shape[0], 1)
            ).mean(dim=1)

        # 对 patch 维度归一化后，用注意力权重聚合 patch tokens
        self_attn = self_attn.softmax(dim=-1)
        return (self_attn.unsqueeze(-1) * visual_embedding).mean(dim=1)

    def project_clip_txt(self, textual_embedding):
        """
        将 CLIP 文本 embedding 投影到 DINO embedding 空间

        若配置 hidden_layer，则经过若干隐藏层进一步变换
        """
        x = self.linear_layer(textual_embedding.float())

        if hasattr(self, "hidden_layers"):
            for hidden_layer in self.hidden_layers:
                if self.act:
                    x = self.act(x)
                x = hidden_layer(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        """
        加载模型权重

        兼容旧版 checkpoint：
        旧代码中的 linear_layer2 会被映射到 hidden_layers.0
        """
        if "linear_layer2.weight" in state_dict:
            state_dict["hidden_layers.0.weight"] = state_dict.pop("linear_layer2.weight")
            state_dict["hidden_layers.0.bias"] = state_dict.pop("linear_layer2.bias")

        super().load_state_dict(state_dict, strict)

    def set_alignment_strategy(self, alignment_strategy):
        """动态修改测试或训练时的对齐策略"""
        self.alignment_strategy = alignment_strategy
        return

    def __len__(self):
        """返回模型参数总量"""
        return sum(p.numel() for p in self.parameters())
