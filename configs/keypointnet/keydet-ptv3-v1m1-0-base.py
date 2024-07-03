_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 16  # bs: total bs in all gpus
num_worker = 10
batch_size_val = None
empty_cache = False
enable_amp = False

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=2,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,  # TODO: set to True in JZ
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        # dict(type="FocalLoss", loss_weight=1.0, alpha=0.05),
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1, weight=[0.05, 0.95]),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 10
eval_epoch = 10
# optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
# scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
# dataset settings
dataset_type = "KeyPointNetDataset"
data_root = "data/keypointnet"
cache_data = False
class_id2names = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel", }

data = dict(
    num_classes=2,
    ignore_index=-1,
    names=["background", "keypoint"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        category="airplane",
        class_id2names=class_id2names,
        transform=[
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category", "segment"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        category="airplane",
        class_id2names=class_id2names,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category", "segment"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        category="airplane",
        class_id2names=class_id2names,
        transform=[
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category", "segment", "name", "inverse", "origin_segment", "origin_coord"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=True,
    )
)
"""
test=dict(
    type=dataset_type,
    split="test",
    data_root=data_root,
    category="airplane",
    class_id2names=class_id2names,
    transform=[
        dict(type="Copy", keys_dict={"segment": "origin_segment"}),
        dict(
            type="GridSample",
            grid_size=0.025,
            hash_type="fnv",
            mode="train",
            keys=("coord", "color", "segment"),
            return_inverse=True,
        ),
    ],
    test_mode=True,
    test_cfg=dict(
        voxelize=dict(
            type="GridSample",
            grid_size=0.01,
            hash_type="fnv",
            mode="test",
            return_grid_coord=True,
            keys=("coord", "color"),
        ),
        crop=None,
        post_transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "index"),
                feat_keys=["coord"],
            ),
        ],
        aug_transform=[
            [dict(type="RandomScale", scale=[0.9, 0.9])],
            [dict(type="RandomScale", scale=[0.95, 0.95])],
            [dict(type="RandomScale", scale=[1, 1])],
            [dict(type="RandomScale", scale=[1.05, 1.05])],
            [dict(type="RandomScale", scale=[1.1, 1.1])],
            [
                dict(type="RandomScale", scale=[0.9, 0.9]),
                dict(type="RandomFlip", p=1),
            ],
            [
                dict(type="RandomScale", scale=[0.95, 0.95]),
                dict(type="RandomFlip", p=1),
            ],
            [
                dict(type="RandomScale", scale=[1, 1]),
                dict(type="RandomFlip", p=1)],
            [
                dict(type="RandomScale", scale=[1.05, 1.05]),
                dict(type="RandomFlip", p=1),
            ],
            [
                dict(type="RandomScale", scale=[1.1, 1.1]),
                dict(type="RandomFlip", p=1),
            ],
        ],
    ),
),
"""

# hooks as in _base_ for SegSem

# tester as in _base_ for SegSem
test = dict(type="KeyDetTester", verbose=True)
