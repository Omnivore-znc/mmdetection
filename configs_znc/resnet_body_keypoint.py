num_cls = 3
num_points = 17
root_dir = '/opt/space_host/zhongnanchang/'
# model settings
input_width = 80
input_height = 128
model = dict(
    type='PointBoxSingleStageDetector',
    #pretrained= root_dir+'mmdet_models/work_dirs/blaze_body_keypoint/latest.pth',
    pretrained=None,
    backbone=dict(
        type='ResNetWH',
        depth=18,
        num_stages=4,
        strides=(2, 2, 2, 2),
        out_indices=(3,),
        #frozen_stages=1,
        input_width = input_width,
        input_height = input_height,
        style='pytorch'),
    neck=None,
    point_head=dict(
        type='KeypointHead',
        num_classes=num_cls,
        num_points=num_points,
        num_fcs=3,
        out_channels_fc=1024,
        target_means=(0.5, 0.5),
        target_stds=(0.05, 0.05),
        double_regress=False))
# model training and testing settings
cudnn_benchmark = True
train_cfg = dict(
    smoothl1_beta=1.,
    debug=False)
test_cfg = dict(score_thr=0.02)
# dataset settings
dataset_type = 'BodyKeypointDataset'
data_root = '/opt/space_host/data_xiaozu/keypoint_coco2017/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=False)
#img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_point=True),
    # dict(
        # type='PhotoMetricDistortion',
        # brightness_delta=32,
        # contrast_range=(0.5, 1.5),
        # saturation_range=(0.5, 1.5),
        # hue_delta=18),
    # dict(
    #     type='Expand',
    #     mean=img_norm_cfg['mean'],
    #     to_rgb=img_norm_cfg['to_rgb'],
    #     ratio_range=(1, 4)),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
    #     min_crop_size=0.3),

    dict(type='RandomCropPoint', crop_size=((input_height, input_width)), min_num_points=3, crop_ratio=0.5),
    dict(type='RandomRotatePoint', pad=(123.675, 116.28, 103.53), max_rotate_degree=10, rotate_ratio=0.5),
    dict(type='Resize', img_scale=(input_width, input_height),
         keep_ratio=False,
         center_padded=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=16),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_points', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_width, input_height),
        flip=False,
        transforms=[
            dict(type='Resize',
                 keep_ratio=False,
                 center_padded=False),
            #dict(type='Resize', img_scale=(input_width, input_height), keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=16),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
        ann_file= data_root + 'idx_list-21w_train.txt',
        img_prefix=data_root,
            min_size=8,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file= data_root + 'idx_list-21w_val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    #val=None,
    test=dict(
        type=dataset_type,
        ann_file= data_root + 'idx_list-21w_val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
#optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer = dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[100, 150, 170])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
checkpoint_config = dict(interval=5)
total_epochs = 180
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir =  root_dir+'mmdet_models/work_dirs/blaze_body_keypoint1912250000_nokeepratio'
load_from = None
resume_from = None
workflow = [('train', 1)]
