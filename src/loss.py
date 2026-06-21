"""训练中使用的图文相似度对比损失实现"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Contrastive(nn.Module):
    """
    基于图文相似度矩阵的对比学习基础目标模块
    
    支持两种损失类型：
        - triplet：margin ranking loss，可选 hardest negative
        - infonce：对称 InfoNCE 损失（t2i / i2t）
    
    属性:
        margin: triplet loss 中的 margin
        sim: 相似度函数或矩阵
        max_violation: 是否只选择 hardest negative
        ltype: loss 类型 ('triplet' 或 'infonce')
        logit_scale: 可学习的 logit 缩放参数
    """

    def __init__(self, sim=None, margin=0, max_violation=False, ltype="triplet"):
        super().__init__()
        self.margin = margin
        self.sim = sim
        self.max_violation = max_violation
        self.ltype = ltype
        # 初始化 logit scale，用于 InfoNCE
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_contrastive_loss(self, scores):
        """
        根据配置选择对比损失
        
        参数:
            scores: 相似度矩阵 [batch_size, batch_size]，scores[i,j] 表示 i 图像与 j 文本相似度
        
        返回:
            loss 标量，已除以 batch_size^2 做归一化
        """
        if self.ltype == "infonce":
            loss = self.__compute_infonce_loss(scores)
        elif self.ltype == "triplet":
            loss = self.__compute_triplet_loss(scores)
        else:
            raise ValueError(f"{self.ltype} not known!")

        return loss / scores.shape[0] ** 2

    def __compute_infonce_loss(self, scores):
        """
        InfoNCE 对比损失（对称）
        
        计算公式:
            L = (CrossEntropy(logits_img2txt, labels) + CrossEntropy(logits_txt2img, labels)) / 2
        """
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * scores  # 图像->文本 logits
        logits_per_text = logits_per_image.t()   # 文本->图像 logits

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)

        # 对称损失
        return (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

    def __compute_triplet_loss(self, scores):
        """
        Triplet margin ranking loss:
            cost = max(0, margin + negative - positive)
        
        参数:
            scores: 相似度矩阵，scores[i,j] = sim(image_i, text_j)
        
        支持 max_violation:
            - 如果 True, 每行/列只取 hardest negative
        """
        # 对角线是正样本 (image_i, text_i)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)  # 对角线展开成矩阵
        d2 = diagonal.t().expand_as(scores)

        # margin ranking loss
        cost_s = (self.margin + scores - d1).clamp(min=0)  # image->text
        cost_im = (self.margin + scores - d2).clamp(min=0)  # text->image

        # 遮掉正样本对角线
        identity_mask = torch.eye(scores.size(0)) > 0.5
        if torch.cuda.is_available():
            identity_mask = identity_mask.to(scores.device)
        cost_s = cost_s.masked_fill_(identity_mask, 0)
        cost_im = cost_im.masked_fill_(identity_mask, 0)

        # 选择 hardest negative
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class ContrastiveLoss(Contrastive):
    """
    封装相似度模型，提供 forward 接口
    
    可以选择返回：
        - loss
        - 相似度矩阵
        - 对应索引
    """

    def __init__(self, sim, margin=0, max_violation=False, ltype="triplet"):
        super().__init__(sim=sim, margin=margin, max_violation=max_violation, ltype=ltype)

    def forward(
        self,
        im,
        s,
        return_similarity_mat=False,
        self_attn_maps=None,
        cls=None,
        text_input_mask=None,
        text_argmax=None,
        return_index=False,
    ):
        """
        前向计算对比损失
        
        参数:
            im: 图像特征
            s: 文本特征
            return_similarity_mat: 是否返回相似度矩阵
            self_attn_maps / cls / text_input_mask / text_argmax: 可选 attention 输入，用于复杂相似度计算
            return_index: 是否返回 index 信息（某些增强模式）
        
        返回:
            loss 或 (loss, similarity_matrix, index)
        """
        # 调用 sim 函数计算相似度矩阵
        scores_result = self.sim(
            im,
            s,
            ret_similarity_matrix=True,
            self_attn_maps=self_attn_maps,
            cls=cls,
            text_input_mask=text_input_mask,
            return_index=return_index,
        )

        if return_index:
            scores, index = scores_result
        else:
            scores = scores_result

        # 计算对比损失
        loss = self.compute_contrastive_loss(scores)
        outputs = [loss]
        if return_similarity_mat:
            outputs.append(scores)
        if return_index:
            outputs.append(index)

        if len(outputs) > 1:
            return tuple(outputs)
        return outputs[0]