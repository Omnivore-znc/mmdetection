num_cls = 3
num_points = 17
root_dir = '/opt/space_host/zhongnanchang/'
# model settings
input_width = 64 #80 error
input_height = 128
model = dict(
    type='PointBoxSingleStageDetector',
    # pretrained= '/share_data/pretrained_models/hrnetv2_w18_imagenet_pretrained.pth',
    # pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNetWH_V2',  #smallV1
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(1,),
                num_channels=(32,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(2, 2),
                num_channels=(16, 32)),
            stage3=dict(
                num_modules=1,
                num_branches=3,
                block='BASIC',
                num_blocks=(2, 2, 2),
                num_channels=(16, 32, 64)),
            stage4=dict(
                num_modules=1,
                num_branches=4,
                block='BASIC',
                num_blocks=(2, 2, 2, 2),
                num_channels=(16, 32, 64, 128))
        ),
        input_width=input_width,
        input_height=input_height,
    ),
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
data_root = '/data_point/keypoint_coco2017/augmentation+test/'
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

    dict(type='RandomRotatePoint', pad=(123.675, 116.28, 103.53), max_rotate_degree=20, rotate_ratio=0.5),
    dict(type='RandomCropPoint', crop_size=((input_height, input_width)), min_num_points=3, crop_ratio=0.5),
    dict(type='RandomHalfBody', min_num_points=8, halfbody_ratio=0.3, scale_factor=1.2),
    dict(type='RandomErasePointV2', area_ratio_range=(0.01, 0.1), min_aspect_ratio=0.5, max_attempt=30, erase_ratio=0.5),
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
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=dataset_type,
        ann_file= data_root + 'train_coco_keypoints_ori_aug_all.txt',
        # ann_file= data_root + 'test_coco_keypoints.txt',
        img_prefix=data_root,
            min_size=8,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file= data_root + 'test_coco_keypoints.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    #val=None,
    test=dict(
        type=dataset_type,
        ann_file= data_root + 'test_coco_keypoints.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
#optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10000,  # 5000
    warmup_ratio=1.0 / 5,
    step=[150, 200, 230, 250])
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
checkpoint_config = dict(interval=1)
total_epochs = 300
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir =  './tools/work_dirs/blaze_point_json/hrnetV2_smallV1_halfbody'
load_from = None
resume_from = None #'./tools/work_dirs/blaze_point_json/blaze_rotate_crop_flip_erase_lr0.06_fc3/latest.pth'
workflow = [('train', 1)]
