num_cls = 2
root_dir = '/opt/space_host/zhongnanchang/'

# model settings
model = dict(
    type='RetinaNet',
    pretrained=root_dir+'mmdet_models/pretrained/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='GARetinaHead',
        num_classes=num_cls,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        octave_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
        # anchoring_means=[.0, .0, .0, .0],
        # anchoring_stds=[0.07, 0.07, 0.14, 0.14],
        # target_means=(.0, .0, .0, .0),
        # target_stds=[0.07, 0.07, 0.11, 0.11],
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[1.0, 1.0, 1.0, 1.0],
        target_means=(.0, .0, .0, .0),
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        #loss_bbox=dict(type='SmoothL1Loss', beta=0.04, loss_weight=1.0)))
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))

# training and testing settings
train_cfg = dict(
    ga_assigner=dict(
        type='ApproxMaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0.4,
        ignore_iof_thr=-1),
    ga_sampler=dict(
        type='RandomSampler',
        num=256,
        pos_fraction=0.5,
        neg_pos_ub=-1,
        add_gt_as_proposals=False),
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    center_ratio=0.2,
    ignore_ratio=0.5,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=300)
# dataset settings
dataset_type = 'VOCDatasetHead'
data_root = '/opt/space_host/data_xiaozu/head_data/'
#img_norm_cfg = dict(mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=img_norm_cfg['mean'], to_rgb=img_norm_cfg['to_rgb'], ratio_range=(1, 4)),
    dict(type='Resize', multiscale_mode='range', img_scale=[(704, 704), (512, 512)], keep_ratio=False),
    #dict(type='Resize', img_scale=(608,608), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file= data_root + 'CrowdHuman/ImageSets/Main/smallest.txt',
        img_prefix=data_root + 'CrowdHuman/',
		pipeline=train_pipeline ),
    val=dict(
        type=dataset_type,
        ann_file= data_root + 'CrowdHuman/ImageSets/Main/smallest.txt',
        img_prefix=data_root + 'CrowdHuman/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        #ann_file= data_root + 'CrowdHuman/ImageSets/Main/test_small_small.txt',
        ann_file= data_root + 'CrowdHuman/ImageSets/Main/test.txt',
        img_prefix=data_root + 'CrowdHuman/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[15, 18])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
#device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = root_dir+'mmdet_models/work_dirs/ga_retinanet_r50_caffe_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
