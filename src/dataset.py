"""预提取特征和图文训练样本的数据集定义"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class DinoClipDataset(Dataset):
    """基于预提取 DINO 图像特征和 CLIP 文本特征的数据集（支持 .pth 文件）"""

    def __init__(
        self,
        features_file,
        features_name="dino_features",
        text_features="ann_feats",
        load_attn_maps=False,
    ):
        self.__load_pth_dataset(features_file, features_name, text_features, load_attn_maps)

    def __getitem__(self, idx):
        """返回单个样本，包括图像特征、文本特征和可选注意力/mask信息"""
        sample = self.data[idx]
        result = {
            "annotation": sample["annotation"],
            "image": sample["image"],
            "metadata": {
                "annotation_id": sample["annotation_id"],
                "image_id": sample["image_id"],
            },
        }

        # 可选字段，仅在特定特征提取模式下存在
        for optional_key in ("self_attn_maps", "dino_features", "text_input_mask", "text_argmax"):
            if optional_key in sample:
                result[optional_key] = sample[optional_key]

        return result

    def __len__(self):
        return len(self.data)

    def __load_pth_dataset(
        self,
        features_file,
        features_name="dino_features",
        text_features="ann_feats",
        load_attn_maps=False,
    ):
        """加载预提取特征的 .pth 文件"""
        print("Loading dataset...")
        data = torch.load(features_file, map_location="cpu")
        print("Dataset loaded!")

        images = {image["id"]: image for image in data["images"]}
        self.data = {}

        # 遍历 annotations 构建训练样本
        for idx, annotation in enumerate(data["annotations"]):
            image = images[annotation["image_id"]]
            self.data[idx] = self.__build_feature_sample(
                annotation,
                image,
                features_name,
                text_features,
                load_attn_maps,
                include_text_argmax=True,
            )

    def __build_feature_sample(
        self,
        annotation,
        image,
        features_name,
        text_features,
        load_attn_maps,
        include_text_argmax,
    ):
        """构造单个训练样本字典"""
        image_feature_key = features_name
        sample = {
            "annotation": self.__get_text_feature(annotation, text_features),
            "image": image[image_feature_key],
            "image_id": annotation["image_id"],
            "annotation_id": annotation["id"],
        }

        if load_attn_maps:
            sample["self_attn_maps"] = image["self_attn_maps"]
            sample["dino_features"] = image["dino_features"]

        if text_features == "clip_txt_out_tokens":
            sample["text_input_mask"] = annotation["text_input_mask"]

        if include_text_argmax and text_features == "clip_second_last_out":
            sample["text_argmax"] = annotation["text_argmax"]

        return sample

    @staticmethod
    def __get_text_feature(annotation, text_features):
        """解析指定文本特征字段，支持 token 平均模式"""
        if text_features != "clip_txt_out_tokens_avg":
            return annotation[text_features]

        # 对有效 token 特征取平均得到文本全局表示
        # 区别于 EOS token 对应的句子向量表示
        mask = annotation["text_input_mask"]
        mask[mask.sum() - 1] = False    # 去掉 EOS
        mask[0] = False                 # 去掉 CLS
        return annotation["clip_txt_out_tokens"][mask].mean(dim=0)


class COCOCaptions(Dataset):
    """COCO 图像/文本描述数据集，用于在线特征提取"""

    def __init__(
        self,
        ann_path,
        data_dir,
        split="train",
        image_transform=None,
        text_transform=None,
        device="cuda",
    ):
        self.data = torch.load(ann_path)
        self.data_dir = data_dir
        self.split = split
        self.samples = self.__build_samples(split)
        self.n_imgs = len(self.samples)
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.device = device

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        annotation = sample["annotation"]
        image = Image.open(os.path.join(self.data_dir, sample["image_path"]))

        if image.mode == "L":
            image = image.convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        if self.text_transform:
            annotation = self.text_transform(annotation)[0]

        return {"image": image, "annotation": annotation}

    def __len__(self):
        return self.n_imgs

    def __build_samples(self, split):
        """根据 split 筛选对应划分的图像标注"""
        images = {image["id"]: image for image in self.data["images"]}
        samples = []

        for annotation in self.data["annotations"]:
            image = images[annotation["image_id"]]
            if split not in image["file_name"]:
                continue
            samples.append(
                {
                    "annotation": annotation["caption"],
                    "image_path": image["file_name"],
                }
            )

        return samples


