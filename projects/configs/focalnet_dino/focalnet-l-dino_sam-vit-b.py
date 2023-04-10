_base_ = [
    '../_base_/datasets/coco_panoptic.py', '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/instance_segment_anything/'

model = dict(
    type='DetWrapperInstanceSAM',
    det_wrapper_type='focalnet_dino',
    det_wrapper_cfg=dict(num_classes=91,
                         param_dict_type='default',
                         ddetr_lr_param=False,
                         onecyclelr=False,
                         modelname='dino',
                         frozen_weights=None,
                         backbone='focalnet_L_384_22k_fl4',
                         focal_levels=4,
                         focal_windows=3,
                         use_checkpoint=False,
                         dilation=False,
                         position_embedding='sine',
                         pe_temperatureH=20,
                         pe_temperatureW=20,
                         return_interm_indices=[0, 1, 2, 3],
                         backbone_freeze_keywords=None,
                         enc_layers=6,
                         dec_layers=6,
                         unic_layers=0,
                         pre_norm=False,
                         dim_feedforward=2048,
                         hidden_dim=256,
                         dropout=0.0,
                         nheads=8,
                         num_queries=900,
                         query_dim=4,
                         num_patterns=0,
                         pdetr3_bbox_embed_diff_each_layer=False,
                         pdetr3_refHW=-1,
                         random_refpoints_xy=False,
                         fix_refpoints_hw=-1,
                         dabdetr_yolo_like_anchor_update=False,
                         dabdetr_deformable_encoder=False,
                         dabdetr_deformable_decoder=False,
                         use_deformable_box_attn=False,
                         box_attn_type='roi_align',
                         dec_layer_number=None,
                         num_feature_levels=5,
                         enc_n_points=4,
                         dec_n_points=4,
                         decoder_layer_noise=False,
                         dln_xy_noise=0.2,
                         dln_hw_noise=0.2,
                         add_channel_attention=False,
                         add_pos_value=False,
                         two_stage_type='standard',
                         two_stage_pat_embed=0,
                         two_stage_add_query_num=0,
                         two_stage_bbox_embed_share=False,
                         two_stage_class_embed_share=False,
                         two_stage_learn_wh=False,
                         two_stage_default_hw=0.05,
                         two_stage_keep_all_tokens=False,
                         num_select=300,
                         transformer_activation='relu',
                         batch_norm_type='FrozenBatchNorm2d',
                         masks=False,
                         aux_loss=True,
                         set_cost_class=2.0,
                         set_cost_bbox=5.0,
                         set_cost_giou=2.0,
                         no_interm_box_loss=False,
                         focal_alpha=0.25,
                         decoder_sa_type='sa',  # ['sa', 'ca_label', 'ca_content']
                         matcher_type='HungarianMatcher',  # or SimpleMinsumMatcher
                         decoder_module_seq=['sa', 'ca', 'ffn'],
                         nms_iou_threshold=-1,
                         dec_pred_bbox_embed_share=True,
                         dec_pred_class_embed_share=True,
                         use_dn=False,
                         dn_number=100,
                         dn_box_noise_scale=0.4,
                         dn_label_noise_ratio=0.5,
                         embed_init_tgt=True,
                         dn_labelbook_size=91,
                         match_unstable_error=True,
                         # for ema
                         use_ema=False,
                         ema_decay=0.9997,
                         ema_epoch=0,
                         use_detached_boxes_dec_out=False),
    det_model_ckpt='ckpt/focalnet_l_dino.pth',
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
