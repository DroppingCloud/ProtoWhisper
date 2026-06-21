"""图文 embedding 配对和检索指标工具"""

import argparse

import numpy
import torch
from tqdm import tqdm

from src.model import ProjectionLayer


def _recall_stats(ranks):
    """根据正确样本排名计算 R@1/R@5/R@10、Median Rank 和 Mean Rank"""
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return r1, r5, r10, medr, meanr


def _normalize_for_numpy_similarity(images, captions):
    """归一化图像和文本特征，便于用点积近似余弦相似度"""
    captions = captions.astype(numpy.float32)
    images = images.astype(numpy.float32)
    captions = captions / numpy.linalg.norm(captions, axis=0)
    images = images / numpy.linalg.norm(images, axis=0)
    return images, captions


def i2t(images, captions, npts=None, return_ranks=False, model=None):
    """
    图像到文本检索

    输入假设:
        images 和 captions 已按 COCO 顺序展开：
        每张图像对应 5 条 caption，因此 images 中同一图像特征会重复 5 次

    检索目标:
        对每张图像，在所有 caption 中检索与其匹配的 5 条文本
        只要这 5 条中的任意一条排得足够靠前，就算召回成功
    """
    if model is not None:
        device = next(model.parameters()).device

    # COCO 中每张图对应 5 条 caption
    if npts is None:
        npts = images.shape[0] // 5

    index_list = []
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)

    for index in tqdm(range(npts)):
        # 每组 5 条 caption 对应同一张图，因此取第 5*index 个图像特征即可
        image = images[5 * index].reshape((1,) + images.shape[1:])

        if model is not None:
            # 使用自定义相似度模型计算 image 与所有 caption 的相似度
            image_tensor = torch.tensor(image).to(device)
            captions_tensor = torch.tensor(captions).to(device)
            with torch.no_grad():
                distances = model(
                    image_tensor.expand((captions_tensor.shape[0],) + (image_tensor.shape[1:])),
                    captions_tensor,
                    ret_similarity_matrix=False,
                ).cpu().detach().numpy()
        else:
            # 默认直接使用归一化特征点积计算相似度
            image, captions = _normalize_for_numpy_similarity(image, captions)
            distances = numpy.dot(image, captions.T).flatten()

        # 相似度从高到低排序
        sorted_indices = numpy.argsort(distances)[::-1]
        index_list.append(sorted_indices[0])

        # 正确 caption 是 [5*index, 5*index+4] 这 5 个位置，取其中排名最高者
        rank = 1e20
        for caption_index in range(5 * index, 5 * index + 5, 1):
            current_rank = numpy.where(sorted_indices == caption_index)[0][0]
            if current_rank < rank:
                rank = current_rank

        ranks[index] = rank
        top1[index] = sorted_indices[0]

    metrics = _recall_stats(ranks)
    if return_ranks:
        return metrics, (ranks, top1)
    return metrics


def t2i(images, captions, npts=None, return_ranks=False, model=None):
    """
    文本到图像检索

    输入假设:
        images 和 captions 已按 COCO 顺序展开：
        每张图像对应 5 条 caption，因此 images 中同一图像特征会重复 5 次

    检索目标:
        对每条 caption，在所有唯一图像中检索其对应图像
    """
    if model is not None:
        device = next(model.parameters()).device

    if npts is None:
        npts = images.shape[0] // 5

    # 每 5 条样本对应同一张图，仅保留唯一图像特征
    unique_images = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)

    for index in tqdm(range(npts)):
        # 当前图像对应的 5 条文本查询
        queries = captions[5 * index:5 * index + 5]

        if model is not None:
            # 使用自定义相似度模型计算每条 caption 与所有图像的相似度
            images_tensor = torch.tensor(unique_images).to(device)
            queries_tensor = torch.tensor(queries).to(device)
            with torch.no_grad():
                distances = numpy.array(
                    [
                        model(
                            images_tensor,
                            query.unsqueeze(0).expand(images_tensor.shape[0], -1),
                            ret_similarity_matrix=False,
                        ).cpu().detach().numpy()
                        for query in queries_tensor
                    ]
                )
        else:
            # 默认直接使用归一化特征点积计算相似度
            unique_images, queries = _normalize_for_numpy_similarity(unique_images, queries)
            distances = numpy.dot(queries, unique_images.T)

        sorted_indices = numpy.zeros(distances.shape)

        # 对每条 caption，找到其对应图像 index 在排序列表中的名次
        for query_index in range(len(sorted_indices)):
            sorted_indices[query_index] = numpy.argsort(distances[query_index])[::-1]
            ranks[5 * index + query_index] = numpy.where(sorted_indices[query_index] == index)[0][0]
            top1[5 * index + query_index] = sorted_indices[query_index][0]

    metrics = _recall_stats(ranks)
    if return_ranks:
        return metrics, (ranks, top1)
    return metrics


def _build_image_feature_map(data, feature_name, model=None):
    """
    建立 image_id 到视觉特征的映射

    若 model=None:
        直接读取 data["images"] 中指定字段作为图像特征

    若 model 不为空:
        调用 model.get_visual_embed() 对 patch/self-attention 等特征进行模型相关聚合
        主要用于测试时的自定义视觉对齐策略

    """
    actual_key = feature_name

    if model is None:
        return {image["id"]: image[actual_key] for image in data["images"]}

    device = next(model.parameters()).device

    return {
        image["id"]: model.get_visual_embed(
            image[feature_name].unsqueeze(0).to(device),
            image["self_attn_maps"].unsqueeze(0).to(device),
            image["dino_features"].unsqueeze(0).to(device)
            if model.weight_attn_heads == "conditioned"
            else None,
        ).squeeze(0).detach().cpu()
        for image in data["images"]
    }


def get_image_and_text_tensor(
    path,
    feature_name="dino_features",
    text_features="ann_feats",
    model=None,
    return_capts_and_imms=False,
):
    """
    按 COCO 检索评估格式构造图像特征张量和文本特征张量

    输出排列:
        每张图像重复 5 次，与该图像的 5 条 caption 一一配对

    返回:
        imm_feats:
            图像特征张量，shape 通常为 [5N, D] 或 [5N, L, D]

        ann_feats:
            文本特征张量，shape 通常为 [5N, D]

    说明:
        这样展开后，i2t/t2i 可以假设：
        第 i 张图像对应 captions[5*i : 5*i+5]
    """
    data = torch.load(path)

    # image_id -> image feature
    images = _build_image_feature_map(data, feature_name, model=model)

    # image_id -> file_name，当前函数中暂未实际写入返回列表
    imm_paths = {image["id"]: image["file_name"] for image in data["images"]}

    annotations = {}
    capts = {}

    # 按 image_id 聚合 caption 特征和原始文本
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]] = (
            [annotation[text_features]] + annotations.get(annotation["image_id"], [])
        )
        capts[annotation["image_id"]] = (
            [annotation["caption"]] + capts.get(annotation["image_id"], [])
        )

    imm_feats, ann_feats = None, None
    imm_file_names = []
    ann_texts = []

    for image_id in tqdm(annotations.keys()):
        # 若图像特征是一维 [D]，depth=1；
        # 若是 patch token [L, D]，depth=L
        depth = 1 if len(images[image_id].shape) == 1 else images[image_id].shape[0]

        # 将同一张图的视觉特征复制到与 caption 数量一致
        image_feature = images[image_id].expand(len(annotations[image_id]), depth, -1)

        # 对全局向量 [D]，去掉额外 depth 维度，得到 [num_caption, D]
        if depth == 1:
            image_feature = image_feature.squeeze(dim=1)

        # 拼接所有图文对
        if ann_feats is None:
            ann_feats = torch.stack(annotations[image_id])
            imm_feats = image_feature
        else:
            ann_feats = torch.cat((ann_feats, torch.stack(annotations[image_id])))
            imm_feats = torch.cat((imm_feats, image_feature))

    if not return_capts_and_imms:
        return imm_feats, ann_feats

    return imm_feats, ann_feats, imm_file_names, ann_texts


def build_argparser():
    """定义检索评估命令行参数"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--custom_alignment",
        default=False,
        action="store_true",
        help="是否在测试时使用模型自定义相似度/对齐策略",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="模型配置文件路径，用于构建 ProjectionLayer",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="模型权重路径；为空时不对输入特征做投影",
    )
    parser.add_argument(
        "--img_features",
        type=str,
        default="avg_self_attn_out",
        help="图像特征字段名，例如 dino_features、avg_self_attn_out、patch_tokens",
    )
    parser.add_argument(
        "--text_features",
        type=str,
        default="ann_feats",
        help="文本特征字段名，例如 ann_feats、clip_txt_out_tokens、clip_second_last_out",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../coco2014_b14_448/test.pth",
        help="测试数据 .pth 路径",
    )

    return parser


def main():
    args = build_argparser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造 COCO 检索评估所需的图文特征张量
    images, texts = get_image_and_text_tensor(
        args.test_data,
        args.img_features,
        text_features=args.text_features,
    )

    print("Model results (t2i, i2t):")

    if not args.custom_alignment and args.weights is not None:
        # 常规评估：只加载投影层，将文本特征投影到与图像特征可比的空间
        proj = ProjectionLayer.from_config(args.config)
        proj.load_state_dict(torch.load(args.weights, "cpu"))
        proj.to(device)
        texts = proj.project_clip_txt(texts.to(device).float()).detach().cpu()

    # 自定义对齐：把模型本身传入 t2i/i2t，由 model.forward 计算相似度
    alignment = proj if args.custom_alignment else None

    t2i_res = t2i(images.numpy(), texts.numpy(), model=alignment)
    print(" & ".join(f"{x:.1f}" if i != 3 else f"{int(x)}" for i, x in enumerate(t2i_res)))

    i2t_res = i2t(images.numpy(), texts.numpy(), model=alignment)
    print(" & ".join(f"{x:.1f}" if i != 3 else f"{int(x)}" for i, x in enumerate(i2t_res)))


if __name__ == "__main__":
    main()