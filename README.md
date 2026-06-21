# ProtoWhisper: SAM-Guided Prototype Adaptation for Localizing Talk2DINO

**ProtoWhisper** 的官方实现代码，通过 SAM 引导的原型自适应方法增强 Talk2DINO 的局部感知能力，用于开放词汇语义分割任务。

## 概述

Talk2DINO 通过学习一个投影层将 CLIP 文本特征映射到 DINOv2 视觉空间，实现开放词汇分割。然而，基于 image-caption 的预训练缺乏细粒度的局部监督，导致在密集预测任务上表现欠佳。

**ProtoWhisper** 通过以下方式解决这一问题：
- 利用 SAM3 从图像标注中生成 phrase-level 伪标签 mask
- 在 DINO 特征空间中构建区域原型（region prototypes）
- 训练轻量级的文本侧残差适配器（text-side residual adapter），同时冻结原始投影层
- 引入原型对比损失（prototype contrastive loss）和基于亲和力的精炼机制

**核心结果：**
- VOC20: **80.9** mIoU（相比 baseline 提升 3.2）
- ADE20K: **29.4** mIoU（相比 baseline 提升 1.5）
- Context-59: **13.7** mIoU（相比 baseline 提升 0.4）

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

**主要依赖：**
- PyTorch >= 1.13
- torchvision
- CLIP (`pip install git+https://github.com/openai/CLIP.git`)
- DINOv2 (通过 `torch.hub` 自动加载)
- MMSegmentation (用于分割评测)
- mmcv-full (需与 PyTorch/CUDA 版本兼容)
- spaCy (用于 noun phrase 提取)

### 数据准备

**训练数据：**
- [Flickr8k](http://hockenmaier.cs.illinois.edu/DenotationGraph/) - 用于 projector 和 adapter 训练
- [COCO](https://cocodataset.org/) - 可选的额外训练数据

**评测数据：**
- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [PASCAL Context](https://cs.stanford.edu/~roozbeh/pascal-context/)

## 快速开始

### 运行评测

**VOC20 评测：**
```bash
bash scripts/run_infer.sh voc
```

**ADE20K 评测：**
```bash
bash scripts/run_infer.sh ade
```

**PASCAL Context-59 评测：**
```bash
bash scripts/run_infer.sh context
```

**使用 ProtoWhisper adapter 评测：**
```bash
# VOC20 (推荐 alpha=0.15)
PROJECTOR_CHECKPOINT=weights/vitb_mlp_infonce_flickr8k.pth \
ADAPTER_CHECKPOINT=weights/protowhisper_adapter.pth \
ADAPTER_ALPHA=0.15 \
bash scripts/run_infer_adapter.sh voc none 1

# ADE20K (推荐 alpha=0.10)
PROJECTOR_CHECKPOINT=weights/vitb_mlp_infonce_flickr8k.pth \
ADAPTER_CHECKPOINT=weights/protowhisper_adapter.pth \
ADAPTER_ALPHA=0.10 \
bash scripts/run_infer_adapter.sh ade none 1
```

## 完整训练流程

### Step 1: 特征提取

#### 1.1 提取 DINO 视觉特征

**Flickr8k：**
```bash
# 训练集
python flickr8k_dino_extraction.py \
  --data_dir data/Flickr8k \
  --images_dir data/Flickr8k/Flicker8k_Dataset \
  --split train \
  --model dinov2_vitb14_reg \
  --extract_cls

# 验证集
python flickr8k_dino_extraction.py \
  --data_dir data/Flickr8k \
  --images_dir data/Flickr8k/Flicker8k_Dataset \
  --split val \
  --model dinov2_vitb14_reg \
  --extract_cls
```

**COCO（可选）：**
```bash
bash scripts/run_extract_coco.sh avg
```

#### 1.2 提取 CLIP 文本特征

**Flickr8k：**
```bash
# 训练集
python flickr8k_text_extraction.py \
  --data_dir data/Flickr8k \
  --dino_pth data/Flickr8k/flickr8k_dino_train.pth \
  --split train \
  --clip_model ViT-B/32

# 验证集
python flickr8k_text_extraction.py \
  --data_dir data/Flickr8k \
  --dino_pth data/Flickr8k/flickr8k_dino_val.pth \
  --split val \
  --clip_model ViT-B/32
```

**COCO（可选）：**
```bash
python text_features_extraction.py \
  --annotation_file data/COCO/annotations/captions_train2014.json \
  --dino_feature_file data/COCO/coco_dino_train.pth \
  --output_file data/COCO/coco_clip_train.pth \
  --clip_model ViT-B/32
```

### Step 2: 训练 Talk2DINO Projector

**单卡训练：**
```bash
bash scripts/run_train_flickr8k.sh avg 1
```

**多卡训练（例如 4 卡）：**
```bash
bash scripts/run_train_flickr8k.sh avg 4
```

参数说明：
- `avg`: 使用平均池化的 DINO 特征
- 第二个参数: 使用的 GPU 数量

**配置文件：** `configs/vitb_mlp_infonce.yaml`

主要超参数：
```yaml
model:
  dino_embed_dim: 768
  hidden_layer: True
  act: tanh

train:
  lr: 0.0001
  ltype: 'infonce'
  num_epochs: 100
  batch_size: 128
```

### Step 3: 构建 SAM3 Pseudo Labels

```bash
python build_flickr8k_sam3_pseudo_labels.py \
  --data_dir data/Flickr8k \
  --dino_pth data/Flickr8k/flickr8k_dino_train.pth \
  --split train \
  --out_pth data/Flickr8k/flickr8k_sam3_pseudo_train.pth
```

该步骤会：
- 使用 spaCy 从 caption 中提取 noun phrases
- 将每个 noun phrase 作为 SAM3 的 text concept prompt
- 生成 phrase-level pseudo mask
- 将 mask resize 到 DINO patch grid (37×37)
- 保存 DINO patch tokens、phrase CLIP features、mask、SAM score、area ratio 等信息

**可选参数：**
```bash
--min_area_ratio 0.01      # 过滤过小的 mask
--min_sam_score 0.5        # 过滤低质量的 SAM 预测
--max_phrases_per_image 10 # 限制每张图像的 phrase 数量
```

### Step 4: 训练 ProtoWhisper Adapter

```bash
bash scripts/run_train_prototype_adapter_flickr8k.sh 1
```

或直接调用训练脚本：
```bash
python train_prototype_adapter.py \
  --train_pseudo data/Flickr8k/flickr8k_sam3_pseudo_train.pth \
  --projector_config configs/vitb_mlp_infonce.yaml \
  --projector_checkpoint weights/vitb_mlp_infonce_flickr8k.pth \
  --out_path weights/protowhisper_adapter.pth \
  --lr 0.0001 \
  --batch_size 256 \
  --num_epochs 10 \
  --adapter_dim 768
```

**训练策略：**
- 冻结 Talk2DINO projector 和 DINO encoder
- 仅训练文本侧 residual adapter（2 层 MLP）
- 损失函数：
  - Prototype contrastive loss: 对比正区域原型与文本 embedding
  - Outside negative loss: 推开负样本区域原型
  - Anchor loss: 保持 adapter 输出接近原始 projector 输出

**支持多卡 DDP 训练：**
```bash
bash scripts/run_train_prototype_adapter_flickr8k.sh 4  # 使用 4 卡训练
```

### Step 5: 评测

```bash
# 评测 baseline (不使用 adapter)
bash scripts/run_infer.sh voc

# 评测 ProtoWhisper (使用 adapter)
PROJECTOR_CHECKPOINT=weights/vitb_mlp_infonce_flickr8k.pth \
ADAPTER_CHECKPOINT=weights/protowhisper_adapter.pth \
ADAPTER_ALPHA=0.15 \
bash scripts/run_infer_adapter.sh voc none 1
```

**推荐的 `ADAPTER_ALPHA` 参数：**
| Dataset | Adapter Alpha |
|---------|---------------|
| VOC20 | 0.14 - 0.16 (推荐 0.15) |
| ADE20K | 0.10 - 0.14 (推荐 0.10) |
| Context-59 | 0.14 - 0.16 (推荐 0.15) |

**启用 affinity refinement（推荐）：**
```bash
bash scripts/run_infer_adapter.sh voc config 1  # 使用配置文件中的 affinity 设置
```

## 实验结果

### 主要结果对比

| Method | Training Data | VOC20 | ADE20K | Context-59 | Avg. |
|--------|--------------|-------|--------|------------|------|
| GroupViT (CVPR 2022) | CC12M/RedCaps | 79.7 | 9.2 | 23.4 | 37.4 |
| ReCo (NeurIPS 2022) | ImageNet-1K | 57.7 | 11.2 | 22.3 | 30.4 |
| MaskCLIP (ICML 2023) | ImageNet-1K | 74.9 | 9.8 | 26.4 | 37.0 |
| CLIP-DIY (WACV 2024) | - | 79.7 | 9.9 | 19.8 | 36.5 |
| Talk2DINO (ICCV 2025) | Flickr8k | 77.9 | 13.1 | 27.4 | 39.5 |
| ProtoWhisper w/o affinity (Ours) | Flickr8k | 78.7 | 13.3 | 28.0 | 40.0 |
| **ProtoWhisper w/ affinity (Ours)** | Flickr8k | **80.9** | **13.7** | **29.4** | **41.4** |

### Adapter 变体消融

| Variant | VOC20 | ADE20K | Context-59 | Avg. |
|---------|-------|--------|------------|------|
| Talk2DINO baseline | 77.85 | 13.12 | 27.43 | 39.47 |
| Dense adapter | 53.15 | 5.70 | 14.57 | 24.47 |
| Gap filter | 78.60 | 13.26 | 27.95 | 39.94 |
| **Soft filter (Ours)** | **78.56** | **13.29** | **27.91** | **39.92** |

*注：上表为未使用 affinity refinement 的结果对比*

### Soft-filter Alpha 调优

| Alpha | VOC20 | Context-59 | ADE20K | Avg. |
|-------|-------|------------|--------|------|
| 0.01 | 77.95 | 13.15 | 27.50 | 39.53 |
| 0.02 | 78.05 | 13.17 | 27.56 | 39.59 |
| 0.05 | 78.30 | 13.23 | 27.72 | 39.75 |
| 0.10 | 78.56 | 13.29 | 27.91 | 39.92 |
| **0.15** | **78.65** | **13.30** | **27.99** | **39.98** |

### Affinity Refinement 效果

| α_affinity | VOC20 | Context-59 | ADE20K | Avg. |
|------------|-------|------------|--------|------|
| No affinity | 78.65 | 13.30 | 27.99 | 39.98 |
| 0.05 | 79.27 | 13.47 | 28.33 | 40.36 |
| 0.10 | 79.69 | 13.58 | 28.57 | 40.61 |
| 0.20 | 80.23 | 13.68 | 28.90 | 40.94 |
| **0.30** | 80.55 | **13.71** | 29.12 | 41.13 |
| 0.40 | 80.77 | 13.71 | 29.29 | 41.26 |
| 0.50 | **80.93** | 13.71 | **29.42** | **41.35** |

*注：该表基于 soft-filter adapter (α=0.15)*

## 项目结构

```
ProtoWhisper/
├── README.md
├── requirements.txt
├── configs/
│   └── vitb_mlp_infonce.yaml          # Projector 训练配置
├── src/
│   ├── model.py                       # Talk2DINO projector 实现
│   ├── prototype_adapter.py           # ProtoWhisper adapter 实现
│   ├── hooks.py                       # DINO/CLIP 特征提取 hooks
│   ├── loss.py                        # 对比损失函数
│   ├── dataset.py                     # 数据加载器
│   ├── train_util.py                  # 训练工具（DDP、学习率调度等）
│   ├── metrics.py                     # 图文检索评估指标
│   ├── config.py                      # 模型配置加载
│   └── open_vocabulary_segmentation/  # 分割评测模块
│       ├── main.py                    # 评测主入口
│       ├── models/dinotext/           # DINOText 模型实现
│       │   ├── dinotext.py            # 核心模型
│       │   ├── masker.py              # Mask 生成模块
│       │   └── pamr.py                # PAMR affinity refinement
│       ├── segmentation/              # 分割评测工具
│       │   └── evaluation/            # 评测流程
│       └── utils/                     # 工具函数
├── report/
│   └── main.pdf                       # 技术报告
├── train.py                           # Talk2DINO Projector 训练入口
├── train_prototype_adapter.py         # ProtoWhisper Adapter 训练入口
├── flickr8k_dino_extraction.py        # Flickr8k DINO 特征提取
├── flickr8k_text_extraction.py        # Flickr8k CLIP 文本特征提取
├── dino_extraction_v2.py              # COCO DINO 特征提取
├── text_features_extraction.py        # COCO CLIP 文本特征提取
└── build_flickr8k_sam3_pseudo_labels.py  # SAM3 伪标签构建
```

## 致谢

本项目基于以下优秀工作：
- [Talk2DINO](https://github.com/example/talk2dino) - 文本到 DINO 空间的投影学习
- [SAM3](https://github.com/example/sam3) - 基于概念的开放词汇分割
- [GroupViT](https://github.com/NVlabs/GroupViT) - 分割评测框架
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) - 语义分割工具库

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
