"""Core DINOText model that combines DINO, CLIP, and the Talk2DINO projector."""

import itertools
import os
import pickle
from math import sqrt
import re
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
import torchvision.transforms as T
import clip

from src.open_vocabulary_segmentation.models.builder import MODELS
from src.open_vocabulary_segmentation.models.dinotext.pamr import PAMR
from src.open_vocabulary_segmentation.models.dinotext.masker import DINOTextMasker
import src.open_vocabulary_segmentation.us as us
from src.open_vocabulary_segmentation.datasets import get_template

from src.model import ProjectionLayer
from src.logit_adapter import LowRankLogitResidualAdapter
from src.loss import Contrastive
from src.hooks import average_text_tokens
from src.config import load_clip_model, load_dinov2_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class DINOText(nn.Module):
    
    def get_self_attention(self, module, input, output):
        self.feats['self_attn'] = output
    
    def get_all_out_tokens(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_txt_out_tokens'] = output
        
    def __init__(
            self, model_name, resize_dim, clip_model_name, proj_class, proj_name, proj_model,
            avg_self_attn_token=False, disentangled_self_attn_token=True, loss=None, pre_trained=True,
            unfreeze_last_text_layer=False, unfreeze_last_image_layer=False, is_eval=True,
            use_avg_text_token=False, keep_cls=False, keep_end_seq=False, with_bg_clean=False,

            use_affinity_refine=True,
            affinity_topk=64,
            affinity_temp=0.07,
            affinity_alpha=0.4,
            affinity_steps=8,
            affinity_class_chunk=64,

            use_multilayer_affinity_refine=False,
            ml_affinity_layers=(5, 8,),
            ml_mid_topk=24,
            ml_mid_temp=0.07,
            ml_mid_alpha=0.20,
            ml_mid_steps=1,

            ml_raw_weight=0.40,
            ml_final_weight=0.35,
            ml_mid_weight=0.25,

            local_adapter_checkpoint=None,
            local_adapter_name=None,
            local_adapter_alpha=None,
            proj_checkpoint=None,
            logit_adapter_checkpoint=None,
            logit_adapter_name=None,
            logit_adapter_gamma=None,
            **kwargs
    ):
        super().__init__()
        self.feats = {}
        self.model_name = model_name

        if 'dinov2' in model_name:
            self.model_family = 'facebookresearch/dinov2'
            self.model = load_dinov2_model(model_name, device=device)
        else:
            raise Exception("Only DINOv2 models are supported")

        self.image_transforms = T.Compose([
            T.Resize((resize_dim, resize_dim)),
            lambda x: T.ToTensor()(x) if not isinstance(x, torch.Tensor) else x / 255.0,
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.model.requires_grad_(False)

        self.clip_model_name = clip_model_name
        self.clip_model, _ = load_clip_model(clip_model_name, device=device)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        if unfreeze_last_text_layer:
            for param in self.clip_model.transformer.resblocks[-1].parameters():
                param.requires_grad = True
            for param in self.clip_model.ln_final.parameters():
                param.requires_grad = True
            self.clip_model.text_projection.requires_grad = True
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        with open(os.path.join('configs', f"{proj_class}.yaml"), 'r') as config_file:
            config = yaml.safe_load(config_file)['model']

        self.proj = ProjectionLayer.from_config(config)

        if pre_trained:
            projector_path = self.resolve_projector_path(proj_name, proj_checkpoint)
            self.load_projector_checkpoint(projector_path)
        self.proj.to(device)

        self.local_adapter = None
        self.local_adapter_alpha = None
        adapter_path = self.resolve_local_adapter_path(local_adapter_checkpoint, local_adapter_name)
        if adapter_path is not None:
            self.load_local_adapter(adapter_path, alpha_override=local_adapter_alpha)

        self.logit_adapter = None
        logit_adapter_path = self.resolve_logit_adapter_path(
            logit_adapter_checkpoint,
            logit_adapter_name,
        )
        if logit_adapter_path is not None:
            self.load_logit_adapter(logit_adapter_path, gamma_override=logit_adapter_gamma)

        self.masker = DINOTextMasker(similarity_type="cosine")
        self.masker = self.masker.eval()

        self.pamr = None

        self.avg_self_attn_token = avg_self_attn_token
        self.disentangled_self_attn_token = disentangled_self_attn_token

        if self.avg_self_attn_token or self.disentangled_self_attn_token or is_eval:
            self.model.blocks[-1].attn.qkv.register_forward_hook(self.get_self_attention)
            self.num_global_tokens = 5 if 'reg' in model_name else 1
            self.num_attn_heads = self.model.num_heads
            self.scale = 0.125

        self.use_avg_text_token = use_avg_text_token
        if self.use_avg_text_token:
            self.clip_model.ln_final.register_forward_hook(self.get_all_out_tokens)
            self.keep_cls = keep_cls
            self.keep_end_seq = keep_end_seq

        self.with_bg_clean = with_bg_clean    

        # Affinity Refine
        self.use_affinity_refine = use_affinity_refine
        self.affinity_topk = affinity_topk
        self.affinity_temp = affinity_temp
        self.affinity_alpha = affinity_alpha
        self.affinity_steps = affinity_steps
        self.affinity_class_chunk = affinity_class_chunk

        # Multi Layer Affinity Refine
        self.use_multilayer_affinity_refine = use_multilayer_affinity_refine
        self.ml_affinity_layers = list(ml_affinity_layers)
        self.ml_mid_topk = ml_mid_topk
        self.ml_mid_temp = ml_mid_temp
        self.ml_mid_alpha = ml_mid_alpha
        self.ml_mid_steps = ml_mid_steps
        self.ml_raw_weight = ml_raw_weight
        self.ml_final_weight = ml_final_weight
        self.ml_mid_weight = ml_mid_weight

    def resolve_projector_path(self, proj_name, checkpoint):
        """解析旧 projector checkpoint 路径"""
        if checkpoint:
            return checkpoint
        return os.path.join("weights", f"{proj_name}.pth")

    def load_projector_checkpoint(self, checkpoint_path):
        """加载旧 projector，并拦截 adapter 权重误用"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "adapter_state_dict" in checkpoint:
            raise ValueError(
                f"{checkpoint_path} looks like a local adapter checkpoint, "
                "but DINOText needs a base projector checkpoint here. "
                "Please restore the original projector weights or set "
                "model.proj_checkpoint to a valid base projector file."
            )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        self.proj.load_state_dict(checkpoint)

    def resolve_local_adapter_path(self, checkpoint, name):
        """解析 adapter checkpoint 路径"""
        if checkpoint:
            return checkpoint
        if name:
            return os.path.join("weights", f"{name}.pth")
        return None

    def load_local_adapter(self, checkpoint_path, alpha_override=None):
        """加载 residual local adapter，用于增强文本局部对齐"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("adapter_state_dict", checkpoint)

        clean_state = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]
            if clean_key.startswith("adapter."):
                clean_key = clean_key[len("adapter."):]
            clean_state[clean_key] = value

        if "linear_layer.weight" in clean_state or "hidden_layers.0.weight" in clean_state:
            raise ValueError(
                f"{checkpoint_path} looks like a base projector checkpoint, "
                "but local_adapter_checkpoint must point to the residual adapter checkpoint."
            )

        if "0.weight" not in clean_state or "2.weight" not in clean_state:
            raise KeyError(
                "adapter checkpoint must contain adapter_state_dict with "
                "'0.weight' and '2.weight'."
            )

        hidden_dim, clip_dim = clean_state["0.weight"].shape
        dino_dim, hidden_dim_2 = clean_state["2.weight"].shape
        if hidden_dim != hidden_dim_2:
            raise ValueError(
                f"adapter hidden dim mismatch: {hidden_dim} vs {hidden_dim_2}"
            )

        adapter = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dino_dim),
        )
        adapter.load_state_dict(clean_state)
        adapter.eval()
        adapter.requires_grad_(False)
        adapter.to(device)

        self.local_adapter = adapter
        self.local_adapter_alpha = (
            float(alpha_override)
            if alpha_override is not None
            else float(checkpoint.get("alpha", 0.1))
        )

        print(
            f"[DINOText] Loaded local adapter: {checkpoint_path} "
            f"(alpha={self.local_adapter_alpha})"
        )

    def resolve_logit_adapter_path(self, checkpoint, name):
        """解析 logit residual adapter checkpoint 路径"""
        if checkpoint:
            return checkpoint
        if name:
            return os.path.join("weights", f"{name}.pth")
        return None

    def load_logit_adapter(self, checkpoint_path, gamma_override=None):
        """加载 logit-level residual adapter，不改变文本或视觉 embedding"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("adapter_state_dict", checkpoint)
        dino_dim = int(checkpoint.get("dino_dim", 768))
        rank_dim = int(checkpoint.get("rank_dim", 128))
        gamma = float(gamma_override) if gamma_override is not None else float(checkpoint.get("gamma", 0.1))

        adapter = LowRankLogitResidualAdapter(
            dino_dim=dino_dim,
            rank_dim=rank_dim,
            gamma=gamma,
            zero_init=False,
        )
        adapter.load_state_dict(state_dict)
        adapter.eval()
        adapter.requires_grad_(False)
        adapter.to(device)
        self.logit_adapter = adapter

        print(
            f"[DINOText] Loaded logit adapter: {checkpoint_path} "
            f"(gamma={gamma}, rank_dim={rank_dim})"
        )
    
    def process_self_attention(self, output, batch_size, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=False):
        qkv = output.reshape(batch_size, num_tokens, 3, num_attn_heads, embed_dim // num_attn_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        self_attn_maps = attn[:, : , 0, num_global_tokens:]
        self_attn = self_attn_maps.mean(dim=1)
        self_attn = self_attn.softmax(dim=-1)
        if ret_self_attn_maps:
            return self_attn, self_attn_maps
        else:
            return self_attn
    
    def encode_text(self, tokenized_texts):
        x = self.clip_model.encode_text(tokenized_texts)
        return x
    
    def encode_image(self, images):
        batch_size, _, _, _ = images.shape
        self_attn_maps = None
        x = self.model(images, is_training=(self.avg_self_attn_token or self.disentangled_self_attn_token))
        batch_size, num_tokens, embed_dim = x['x_norm_patchtokens'].shape
        num_tokens = num_tokens + self.num_global_tokens
        if self.avg_self_attn_token or self.disentangled_self_attn_token:
            self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
        if self.avg_self_attn_token:
            x = (self_attn.unsqueeze(-1) * x['x_norm_patchtokens']).mean(dim=1)
        elif self.disentangled_self_attn_token:
            self_attn_maps = self_attn_maps.softmax(dim=-1)
            x = (x['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)

        return x, self_attn_maps

    def forward(self, image, text, return_logit_scale=False):
        with torch.no_grad():
            txt_embed_clip = self.encode_text(text)
            
        img_embed, self_attn_maps = self.encode_image(image)
        
        img_embed, txt_embed = self.proj(img_embed, txt_embed_clip, ret_embeds=True, self_attn_maps=self_attn_maps)
        if self.local_adapter is not None:
            txt_embed = self.project_text_with_adapter(txt_embed_clip, normalize=True)
        
        if return_logit_scale:
            return txt_embed, img_embed, self.logit_scale

        return txt_embed, img_embed

    def project_text_with_adapter(self, text_embs, normalize=True):
        """旧 projector 输出加上 adapter residual"""
        text_embs = text_embs.float()
        projected = self.proj.project_clip_txt(text_embs)

        if self.local_adapter is not None:
            delta = self.local_adapter(text_embs)
            projected = projected + self.local_adapter_alpha * delta

        if normalize:
            projected = us.normalize(projected, dim=-1)

        return projected
        
    @torch.no_grad()
    def build_dataset_class_tokens(self, template_set, classnames):
        tokens = []
        templates = get_template(template_set)
        for classname in classnames:
            tokens.append(
                clip.tokenize([template.format(classname) for template in templates])
            )
        tokens = torch.stack(tokens)
        return tokens

    @torch.no_grad()
    def build_text_embedding(self, text):
        """
        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens

        Returns:
            text_embs
        """
        text = text.to(next(self.parameters()).device)
        num_classes, num_templates = text.shape[:2]
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        chunk_size = 32
        N = text.size(0)
        
        if not self.use_avg_text_token:
            text_embs = torch.cat([
                self.clip_model.encode_text(text[i:i + chunk_size])
                for i in range(0, N, chunk_size)
            ])
        else:
            text_embs = []
            for i in range(0, N, chunk_size):
                self.clip_model.encode_text(text[i:i + chunk_size])
                text_embs.append(average_text_tokens(self.feats['clip_txt_out_tokens'] @ self.clip_model.text_projection, text[i:i + chunk_size] > 0, self.keep_cls, self.keep_end_seq))
            text_embs = torch.cat(text_embs)

        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates)
        text_embs = text_embs.mean(dim=1).float()
        text_embs = self.project_text_with_adapter(text_embs, normalize=True)

        return text_embs

    def apply_pamr(self, image, mask):
        image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
        if self.pamr is None:
            pamr_iter = 10
            pamr_kernel = [1, 2, 4, 8, 12, 24]
            self.pamr = PAMR(pamr_iter, pamr_kernel)
            self.pamr.eval()
            self.pamr.to(next(self.parameters()).device)

        mask = self.pamr(image, mask)
        return mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b
    
    # Affinity Refine ===========================
    def affinity_logit_refinement(
        self,
        mask,
        image_feat,
        topk=32,
        temp=0.1,
        alpha=0.3,
        steps=1,
        class_chunk=64,
        exclude_self=True,
    ):
        """
        Refine class score maps using DINO patch-to-patch affinity.

        Args:
            mask: [B, num_classes, H, W]
                Class score maps at DINO patch resolution.
            image_feat: [B, C, H, W]
                DINO patch features at the same resolution as mask.
            topk:
                Number of nearest patch neighbors retained for affinity propagation.
            temp:
                Temperature for affinity softmax. Smaller means sharper propagation.
            alpha:
                Residual propagation strength.
            steps:
                Number of propagation iterations.
            class_chunk:
                Chunk size over classes to reduce memory.

        Returns:
            refined_mask: [B, num_classes, H, W]
        """
        orig_dtype = mask.dtype

        bs, c, h, w = image_feat.shape
        _, num_classes, mh, mw = mask.shape
        assert h == mh and w == mw, f"image_feat and mask spatial size mismatch: {(h, w)} vs {(mh, mw)}"

        hw = h * w

        # [B, C, H, W] -> [B, HW, C]
        patch_feat = image_feat.flatten(2).transpose(1, 2).contiguous()
        patch_feat = F.normalize(patch_feat.float(), dim=-1)

        # Dense patch affinity: [B, HW, HW]
        affinity = torch.bmm(patch_feat, patch_feat.transpose(1, 2))

        # Negative similarity usually means unrelated patches.
        # Removing it makes propagation safer.
        affinity = affinity.clamp(min=0)

        if exclude_self:
            eye = torch.eye(hw, device=affinity.device, dtype=torch.bool).unsqueeze(0)
            affinity = affinity.masked_fill(eye, -1e4)

        # Keep only top-k neighbors for each patch.
        k = min(topk, hw)
        topk_vals, topk_idx = torch.topk(affinity, k=k, dim=-1)

        # Row-normalized affinity.
        topk_vals = F.softmax(topk_vals / temp, dim=-1)

        sparse_affinity = torch.zeros_like(affinity)
        sparse_affinity.scatter_(dim=-1, index=topk_idx, src=topk_vals)

        # [B, num_classes, H, W] -> [B, num_classes, HW]
        score = mask.flatten(2).float()

        # Propagate class scores over DINO affinity graph.
        # refined_score[:, :, i] = sum_j A[i, j] * score[:, :, j]
        for _ in range(steps):
            refined_chunks = []

            for start in range(0, num_classes, class_chunk):
                end = min(start + class_chunk, num_classes)
                cur = score[:, start:end, :]  # [B, chunk, HW]

                propagated = torch.bmm(cur, sparse_affinity.transpose(1, 2))

                # Residual update to avoid over-smoothing.
                cur = (1.0 - alpha) * cur + alpha * propagated
                refined_chunks.append(cur)

            score = torch.cat(refined_chunks, dim=1)

        refined_mask = score.reshape(bs, num_classes, h, w).to(orig_dtype)
        return refined_mask
    
    # Intermediatae Hook ===========================
    def _resolve_dino_layer(self, layer_index):
        num_layers = len(self.model.blocks)
        return layer_index if layer_index >= 0 else num_layers + layer_index

    @torch.no_grad()
    def get_intermediate_patch_features(self, img_preprocessed, layer_ids):
        """
        Get DINOv2 intermediate patch tokens.

        Args:
            img_preprocessed: [B, 3, H, W]
            layer_ids: list of block indices, e.g. [8] or [-4]

        Returns:
            feats: list of [B, C, h, w]
        """
        resolved_layers = [self._resolve_dino_layer(layer) for layer in layer_ids]

        outputs = self.model.get_intermediate_layers(
            img_preprocessed,
            n=resolved_layers,
            reshape=False,
            return_class_token=False,
            norm=True,
        )

        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]

        feats = []

        for x in outputs:
            if isinstance(x, (tuple, list)):
                x = x[0]

            # x: [B, N, C]
            b, n, c = x.shape
            h = w = int(sqrt(n))
            assert h * w == n, f"Cannot reshape {n} patch tokens into square grid."

            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            feats.append(x)

        return feats
    
    # Multi Layer Affinity Refine ===========================
    def multi_layer_affinity_logit_refinement(
        self,
        raw_mask,
        final_image_feat,
        img_preprocessed,
    ):
        """
        Multi-layer affinity refinement.

        raw_mask:
            [B, num_classes, h, w], generated by final-layer DINO tokens and text embeddings.

        final_image_feat:
            [B, C, h, w], final-layer DINO patch tokens.

        img_preprocessed:
            [B, 3, H, W], normalized input image for DINO.
        """

        # 1. Final-layer affinity refinement
        refined_final = self.affinity_logit_refinement(
            mask=raw_mask,
            image_feat=final_image_feat,
            topk=self.affinity_topk,
            temp=self.affinity_temp,
            alpha=self.affinity_alpha,
            steps=self.affinity_steps,
            class_chunk=self.affinity_class_chunk,
            exclude_self=True,
        )

        # 2. Middle-layer affinity refinement
        mid_feats = self.get_intermediate_patch_features(
            img_preprocessed,
            self.ml_affinity_layers,
        )

        refined_mid_list = []

        for mid_feat in mid_feats:
            refined_mid = self.affinity_logit_refinement(
                mask=raw_mask,
                image_feat=mid_feat,
                topk=self.ml_mid_topk,
                temp=self.ml_mid_temp,
                alpha=self.ml_mid_alpha,
                steps=self.ml_mid_steps,
                class_chunk=self.affinity_class_chunk,
                exclude_self=True,
            )
            refined_mid_list.append(refined_mid)

        if len(refined_mid_list) == 1:
            refined_mid = refined_mid_list[0]
        else:
            refined_mid = torch.stack(refined_mid_list, dim=0).mean(dim=0)

        # 3. Weighted fusion
        weight_sum = self.ml_raw_weight + self.ml_final_weight + self.ml_mid_weight

        mask = (
            self.ml_raw_weight * raw_mask
            + self.ml_final_weight * refined_final
            + self.ml_mid_weight * refined_mid
        ) / weight_sum

        return mask
    
    @torch.no_grad()
    def generate_masks(
            self, image, img_metas, text_emb, classnames, text_is_token=False, apply_pamr=False, background_func="weighted_average_sigmoid", lambda_bg=0.2,
    ):
        """Generate masks for each text embeddings

        Args:
            image [B, 3, H, W]

        Returns:
            softmask [B, N, H, W]: softmasks for each text embeddings
        """

        H, W = image.shape[2:]

        pH, pW = image.shape[2:]
        num_classes = text_emb.shape[0]
        batch_size = image.shape[0]

        image = image[:, [2, 1, 0], :, :]  # BGR to RGB
        ori_image = image.clone()

        img_preprocessed = self.image_transforms(image).to(next(self.parameters()).device)
        image_feat = self.model.forward_features(img_preprocessed)['x_norm_patchtokens']

        batch_size, num_tokens, embed_dim = image_feat.shape

        b, np, c = image_feat.shape
        np_h = np_w = int(sqrt(np))

        # 保留原始 patch tokens 用于交叉注意力
        image_feat = image_feat.reshape(b, np_h, np_w, c).permute(0, 3, 1, 2)

        self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens + self.num_global_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
        mask, simmap = self.masker.forward_seg(image_feat, text_emb, hard=False)

        if self.logit_adapter is not None:
            residual = self.logit_adapter.forward_dense(text_emb, image_feat)
            simmap = simmap + residual
            _, mask = self.masker.sim2mask(simmap, deterministic=True)

        # ================= Affinity Refinement =====================
        if self.use_multilayer_affinity_refine:
            mask = self.multi_layer_affinity_logit_refinement(
                raw_mask=mask,
                final_image_feat=image_feat,
                img_preprocessed=img_preprocessed,
            )

        elif self.use_affinity_refine:
            mask = self.affinity_logit_refinement(
                mask=mask,
                image_feat=image_feat,
                topk=self.affinity_topk,
                temp=self.affinity_temp,
                alpha=self.affinity_alpha,
                steps=self.affinity_steps,
                class_chunk=self.affinity_class_chunk,
                exclude_self=True,
            )


        if self.with_bg_clean:
            mask = self.similarity_assignment_weighted(mask, image_feat, self_attn_maps, text_emb, lambda_bg)

        mask = F.interpolate(mask, (pH, pW), mode='bilinear', align_corners=True)

        if apply_pamr:
            for c in range(0, mask.shape[1], 30):
                mask[:, c:c + 30] = self.apply_pamr(ori_image, mask[:, c:c + 30])

        assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

        return mask, simmap

    def similarity_assignment_weighted(self, mask, image_feat, self_attn_maps, text_emb, lambda_bg=0.2):
        bs, c, h, w = image_feat.shape
        bs, num_classes, h, w = mask.shape
        bs, num_heads, hw = self_attn_maps.shape
        image_feat = image_feat.reshape(bs, c, hw)
        num_classes, c = text_emb.shape
        avg_head_embed = (self_attn_maps.unsqueeze(2) * image_feat.unsqueeze(1)).mean(dim=-1)
        avg_head_embed = avg_head_embed / avg_head_embed.norm(dim=-1, keepdim=True)
        avg_head_embed = avg_head_embed.permute(0, 2, 1)
        head_text_sim = text_emb.unsqueeze(0) @ avg_head_embed
        head_text_sim = (head_text_sim).softmax(dim=-1)
        head_text_sim_sum = head_text_sim.sum(dim=-1)
        
        self_attn_maps_repeat = self_attn_maps.unsqueeze(1).repeat(1, num_classes, 1, 1)
        head_text_sim_repeat = head_text_sim.unsqueeze(-1).repeat(1, 1, 1, hw)
        avg_self_attn_per_class = (self_attn_maps_repeat * head_text_sim_repeat).sum(dim=2) / head_text_sim_sum.unsqueeze(-1).repeat(1, 1, hw)
        avg_self_attn_per_class = avg_self_attn_per_class.softmax(dim=-1)
        
        min_self_attn = avg_self_attn_per_class.min().item()
        max_self_attn = avg_self_attn_per_class.max().item()
        max_self_attn = max(max_self_attn, max_self_attn - min_self_attn)
        avg_self_attn_per_class = avg_self_attn_per_class - min_self_attn
        avg_self_attn_per_class = avg_self_attn_per_class / max_self_attn
        avg_self_attn_per_class = avg_self_attn_per_class * (mask.max() - mask.min()) + mask.min()
        mask = mask.reshape(num_classes, hw)
        mask_output = (mask + lambda_bg * avg_self_attn_per_class).reshape(bs, num_classes, h, w) / (1 + lambda_bg)
        return mask_output
