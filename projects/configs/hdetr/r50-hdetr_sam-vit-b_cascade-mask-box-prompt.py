_base_ = [
    '../_base_/datasets/coco_panoptic.py', '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/instance_segment_anything/'

model = dict(
    type='DetWrapperInstanceSAMMaskPrompt',
    det_wrapper_type='hdetr',
    det_wrapper_cfg=dict(aux_loss=True,
                         backbone='resnet50',
                         num_classes=91,
                         cache_mode=False,
                         dec_layers=6,
                         dec_n_points=4,
                         dilation=False,
                         dim_feedforward=2048,
                         drop_path_rate=0.2,
                         dropout=0.0,
                         enc_layers=6,
                         enc_n_points=4,
                         focal_alpha=0.25,
                         frozen_weights=None,
                         hidden_dim=256,
                         k_one2many=6,
                         lambda_one2many=1.0,
                         look_forward_twice=True,
                         masks=False,
                         mixed_selection=True,
                         nheads=8,
                         num_feature_levels=4,
                         num_queries_one2many=1500,
                         num_queries_one2one=300,
                         position_embedding='sine',
                         position_embedding_scale=6.283185307179586,
                         remove_difficult=False,
                         topk=100,
                         two_stage=True,
                         use_checkpoint=False,
                         use_fp16=False,
                         with_box_refine=True),
    stage_2_with_box_p=True,
    det_model_ckpt='ckpt/r50_hdetr.pth',
    num_classes=80,
    model_type='vit_b',
    sam_checkpoint='ckpt/sam_vit_b_01ec64.pth',
    use_sam_iou=True,
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
