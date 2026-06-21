"""COCO 风格标注与 WebDataset 分片之间的转换工具"""

import argparse
import json
import os
import re
import tarfile
from io import BytesIO

import torch
import webdataset as wds
from tqdm import tqdm


def _torch_to_bytes(obj):
    """用 torch.save 将 Python 对象序列化到内存缓冲区"""
    buffer = BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return buffer


def _copy_image_fields_to_annotation(annotation, image):
    """将图像级特征字段合并到 annotation 记录中，便于写入 WDS"""
    for field in image.keys():
        if field == "id":
            continue
        annotation[field] = image[field]
    return annotation


def create_webdataset_single_shard(data, output_tar_path):
    """根据 COCO 风格图像/标注数据创建单个 WebDataset tar 分片"""
    if os.path.exists(output_tar_path):
        os.remove(output_tar_path)
        print(f"Old {output_tar_path} has been deleted successfully.")

    annotations_map = {annotation["image_id"]: annotation for annotation in data["annotations"]}

    with tarfile.open(output_tar_path, "w") as tar:
        for image in data["images"]:
            image_id = image["id"]
            if image_id not in annotations_map:
                continue

            annotation = _copy_image_fields_to_annotation(annotations_map[image_id], image)
            buffer = _torch_to_bytes(annotation)

            tar_info = tarfile.TarInfo(name=f"{image_id}.pth")
            tar_info.size = buffer.getbuffer().nbytes
            tar.addfile(tar_info, buffer)

    print(f"{output_tar_path} created succesfully!")


def create_webdataset_tar(data, output_tar_dir, n_shards=1, offset=0):
    """
    将 COCO 风格数据拆分为 WebDataset tar 分片

    n_shards: 输出 tar 文件的分片数量
    offset: 第一个输出分片的编号偏移
    """
    step_ann = len(data["annotations"]) // n_shards
    step_imm = len(data["images"]) // n_shards

    for shard_index in range(n_shards):
        start_ann = step_ann * shard_index
        end_ann = (step_ann * (shard_index + 1) - 1) if (shard_index + 1) < n_shards else len(data["annotations"])
        start_imm = step_imm * shard_index
        end_imm = (step_imm * (shard_index + 1) - 1) if (shard_index + 1) < n_shards else len(data["images"])

        shard_data = {
            "annotations": data["annotations"][start_ann:end_ann],
            "images": data["images"][start_imm:end_imm],
        }
        output_tar_path = os.path.join(output_tar_dir, f"shard{str(shard_index + offset).zfill(5)}.tar")
        create_webdataset_single_shard(shard_data, output_tar_path)


def _build_shard_pattern(shards_dir, n_splits, batch_offset):
    """为选中的 tar 文件区间构造 brace-expanded 分片路径模式"""
    tar_list = [filename for filename in os.listdir(shards_dir) if "tar" in filename]
    n_files = max([int(re.findall(r"\d+", filename)[0]) for filename in tar_list]) + 1
    match = re.search(r"([a-zA-Z]+)\d+", tar_list[0])
    prefix = "" if match is None else match.group(1)

    batch_dim = n_files // n_splits
    start = batch_dim * batch_offset
    end = batch_dim * (batch_offset + 1) - 1
    return os.path.join(shards_dir, f"{prefix}{{{str(start).zfill(5)}..{str(end).zfill(5)}}}.tar")


def _convert_json_sample(elem):
    """将 JSON WebDataset 样本转换为 COCO 风格 annotation/image 记录"""
    obj = elem["json"]
    annotation = {
        "image_id": obj["key"],
        "id": obj["key"],
        "caption": obj["caption"],
    }
    image = {
        "id": obj["key"],
        "file_name": obj["url"],
        "height": obj["height"],
        "width": obj["width"],
        "jpg": elem["jpg"],
    }
    return annotation, image


def _convert_pth_sample(elem):
    """将 PTH WebDataset 样本转换为 COCO 风格 annotation/image 记录"""
    obj = elem["pth"]
    annotation_fields = ["image_id", "id", "caption", "text_features"]
    image_fields = [
        "id",
        "file_name",
        "height",
        "width",
        "dino_features",
        "avg_self_attn_out",
        "second_last_out",
        "patch_tokens",
        "self_attn_maps",
        "disentangled_self_attn",
    ]
    annotation = {field: obj[field] for field in annotation_fields if field in obj}
    image = {field: obj[field] for field in image_fields if field in obj}
    return annotation, image


def cc2coco_format(shards_dir, n_splits=4, batch_offset=0):
    """
    读取 WebDataset 分片目录，并转换为 COCO 风格字典

    n_splits 和 batch_offset 用于选择一个连续分片区间，便于分布式或分阶段预处理
    """
    in_path = _build_shard_pattern(shards_dir, n_splits, batch_offset)

    print(f"Reading webdataset {in_path}")
    dataset = wds.WebDataset(in_path).decode("pil")
    data = {"annotations": [], "images": []}

    for elem in tqdm(dataset):
        if "json" in elem:
            annotation, image = _convert_json_sample(elem)
        else:
            annotation, image = _convert_pth_sample(elem)

        data["annotations"].append(annotation)
        data["images"].append(image)

    return data


def read_coco_format_wds(ann_path):
    """读取单个 COCO 风格 WebDataset tar，并返回 image/annotation 列表"""
    print(f"Reading webdataset {ann_path}")
    dataset = wds.WebDataset(ann_path).decode("pil")

    annotations = []
    images = []
    for elem in tqdm(dataset):
        obj = elem["pth"]
        annotation_fields = ["id", "image_id", "caption", "ann_feats"]
        image_fields = ["id", "file_name", "height", "width"]
        annotations.append({field: obj[field] for field in annotation_fields})
        images.append({field: obj[field] for field in image_fields})

    return {
        "images": images,
        "annotations": annotations,
    }


def build_argparser():
    """定义 WebDataset 转换命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, required=True, help="Directory of the output files")
    parser.add_argument("--out_shards", type=int, default=16, help="Number of splits of the output")
    parser.add_argument("--shards_dir", type=str, required=True, help="Directory of the webdataset")
    return parser


def save_coco_shards(data, out_path, out_shards):
    """将 COCO 风格数据保存为单个 PTH 文件或多个 PTH 分片"""
    if out_shards == 1:
        torch.save(data, out_path)
        return

    os.makedirs(out_path, exist_ok=True)
    step = len(data["annotations"]) // out_shards
    for shard_index in range(out_shards):
        start = step * shard_index
        end = (step * (shard_index + 1) - 1) if shard_index < out_shards else len(data["annotations"]) - 1

        shard_data = {
            "annotations": data["annotations"][start:end],
            "images": data["images"][start:end],
        }
        save_path = os.path.join(out_path, f"shard{shard_index}.pth")
        torch.save(shard_data, save_path)
        print(f"Saved elements between {start} and {end} at {save_path}")


def main():
    args = build_argparser().parse_args()
    data = cc2coco_format(args.shards_dir)

    print(f"Dataset composed by {len(data['images'])} couple (text, image)")
    save_coco_shards(data, args.out_path, args.out_shards)


if __name__ == "__main__":
    main()
