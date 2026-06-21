# ------------------------------------------------------------------------------
# GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------

"""MMSeg custom-import configuration for Talk2DINO segmentation modules."""

custom_imports = dict(
    imports=["segmentation.datasets.coco_object", "segmentation.datasets.pascal_voc"],
    allow_failed_imports=False,
)
