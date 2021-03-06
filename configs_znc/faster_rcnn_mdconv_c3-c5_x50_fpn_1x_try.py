num_cls = 30
root_dir = '/opt/space_host/zhongnanchang/'
#norm_cfg = dict(type='SyncBN', requires_grad=True)
# model settings
model = dict(
    type='FasterRCNN',
    pretrained=root_dir+'mmdet_models/pretrained/resnext50_32x4d-7cdf4587.pth',
    backbone=dict(
        type='ResNeXt',
        depth=50,
      	groups=32,
      	base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
    		dcn=dict(
            modulated=True,
          	groups=32,
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        #anchor_scales=[8],
        #anchor_ratios=[0.5, 1.0, 2.0],
        anchor_scales=[4, 6, 10],
        anchor_ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
        anchor_strides=[4, 8, 16, 32, 64],
        #anchor_strides=[2, 4, 8, 16, 32],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=num_cls,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',#'OHEMSampler',
            num=256,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.01, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'VOCDatasetGridDefect'
data_root = '/opt/space_host/data_xiaozu/grid_defect_29classes_new/grid_defect_29classes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=img_norm_cfg['mean'], to_rgb=img_norm_cfg['to_rgb'], ratio_range=(1, 4)),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(type='Resize', multiscale_mode='range', img_scale=[(736, 736), (576, 576)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32), #此转换要放各种数据变换之后，紧邻'DefaultFormatBundle'之前
    dict(type='DefaultFormatBundle'),  #此转换要放'Pad'之后，紧邻'Collect'之前
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(704, 704),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= data_root + 'ImageSets/Main/trainval_pack1_pack2_bdz_round1all_xyselfget_aug_maybetest.txt',
        img_prefix=data_root ,
		pipeline=train_pipeline ),
    val=dict(
        type=dataset_type,
        ann_file= data_root + 'ImageSets/Main/test_pack1_pack2_small.txt',
        img_prefix=data_root ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file= data_root + 'ImageSets/Main/test_pack1_pack2.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.035, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[18, 21])
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
total_epochs = 23
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = root_dir+'mmdet_models/work_dirs/faster_rcnn_mdconv_c3-c5_x50_fpn_1x_normal_results'
load_from = None#root_dir+'pretrained/faster-x50-mdconv-epoch35.pth'
resume_from = None #= root_dir+'/work_dirs/faster_rcnn_mdconv_c3-c5_x50_fpn_1x/latest.pth'
workflow = [('train', 1)]
