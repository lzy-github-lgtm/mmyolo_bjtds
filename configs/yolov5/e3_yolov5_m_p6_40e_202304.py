# TODO: Need to solve the problem of multiple file_client_args parameters
# _file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
_file_client_args = dict(backend='disk')

# ========================Frequently modified parameters======================
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.01     
lr_factor = 0.1         # Learning rate scaling factor 
weight_decay = 0.0005
# Maximum training epochs
max_epochs = 40         
# warmup取  max_epoch 、 warmup_mim_iter 两者中的最大值
warmup_epochs = 1       # warmup_epochs
warmup_mim_iter = 200   # warmup iters
# Save model checkpoint and validation intervals
save_checkpoint_intervals = 10
# The maximum checkpoints to keep.
max_keep_ckpts = 3

# ========================Data modified parameters======================
# -----data related-----
num_classes = 80  # Number of classes for classification
img_scale = (1280, 1280)  # width, height
# -----data_augment-----
affine_scale = 0.9      # YOLOv5RandomAffine scaling ratio
mixup_prob = 0.1
# -----data path-----
data_root = 'datasets/'# Root path of data
# Path of train annotation file
train_ann_file = 'annotations/trainval.json'
train_data_prefix = 'images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/test.json'
val_data_prefix = 'images/'  # Prefix of val image path
# Path of test annotation file
test_ann_file = 'annotations/test.json'
test_data_prefix = 'images/'  # Prefix of val image path

# Batch size of a single GPU during training
train_batch_size_per_gpu = 3
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 2 * train_batch_size_per_gpu
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2 * val_batch_size_per_gpu


# ========================Possible modified parameters========================
loss_cls_weight = 0.3
loss_obj_weight = 0.7
loss_bbox_weight = 0.05
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.67
# The scaling factor that controls the width of the network structure
widen_factor = 0.75
# -----model related-----
# Strides of multi-scale prior box
strides = [8, 16, 32, 64]
num_det_layers = len(strides)  # The number of model output scales
# Basic size of multi-scale prior box
anchors = [
    [(19, 27), (44, 40), (38, 94)],  # P3/8
    [(96, 68), (86, 152), (180, 137)],  # P4/16
    [(140, 301), (303, 264), (238, 542)],  # P5/32
    [(436, 615), (739, 380), (925, 792)]  # P6/64
]
# -----train val related-----
prior_match_thr = 4.  # Priori box matching threshold
# The obj loss weights of the three output layers
obj_level_weights = [4.0, 1.0, 0.25, 0.06]

# -----train val related-----
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.05,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.5),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image
# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        arch='P6',
        out_indices=(2, 3, 4, 5),
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 768, 1024],
        out_channels=[256, 512, 768, 1024],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 768, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight *
            (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight *
            ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights),
    test_cfg=model_test_cfg)


# -----data related-----
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# classes = ['AirOutletBlocker', 'AntirollTorsionBar', 'Arm', 'AutoPassing', 'Axle', 'Baffle', 'BoltHead', 'BoltNut', 'Boot', 'BrakeClamp', 'BrakeCylinder', 'BrakePad', 'Clip', 'CotterPin', 'Coupling', 'EndSheet', 'GearCase', 'Grinder', 'GroundingDevice', 'LateralDamper', 'LateralStop', 'LockSpring', 'LockingWire', 'MagneticBoltHolder', 'Mirror', 'Nameplate', 'Nozzle', 'OilLevelMirror', 'OilPlugB', 'OilPlugS', 'ParkingBrakeCylinder', 'PipeJoint', 'Putty', 'RadiatingRib', 'Rubber', 'Sander', 'SingleLockingWire', 'StoneSweeper', 'TractionRod', 'WheelTread', 'WholeCotterPin', 'abnormal_Feather', 'crack', 'lost_BoltHead', 'lost_OilPlugS', 'lost_Putty']
classes = [
'OilPlugS',
'BoltHead',
'BoltNut',
'WholeCotterPin',
'Coupling',
'LockingWire',
'BrakeCylinder',
'Boot',
'WheelTread',
'CotterPin',
'BrakePad',
'BrakeClamp',
'TailCotterPin',
'OilLevelMirror',
'SingleLockingWire',
'Mirror',
'OilPlugB',
'Putty',
'Nameplate',
'Rubber',
'Axle',
'TractionRod',
'MagneticBoltHolder',
'Sander',
'Nozzle',
'RadiatingRib']
METAINFO = {
    'classes': tuple(classes + ["None"] * (num_classes - len(classes))),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122), (191, 162, 208)]
}

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_pipeline]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        metainfo=METAINFO,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor= strides[-1],      # 根据最后一层feature maps的跨度而定
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        metainfo=METAINFO,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=test_data_prefix),
        ann_file=test_ann_file,
        metainfo=METAINFO,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox',
    classwise=True, 
    iou_thrs=[0.5])
# test_evaluator = val_evaluator
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + test_ann_file,
    metric='bbox',
    classwise=True, 
    iou_thrs=[0.5])

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,  # 验证间隔，每 10 个 epoch 验证一次
    dynamic_intervals=[(280, 1)]) # 到 280 epoch 开始切换为间隔 1 的评估方式
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    clip_grad=None,                                         # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',    # MMYOLO 中默认采用 Hook 方式进行优化器超参数的调节
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,        
        warmup_mim_iter=warmup_mim_iter,
        ),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals, # 每 save_checkpoint_intervals 轮保存 1 次权重文件
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts),     # 最多保存 max_keep_ckpts 个权重文件
    logger=dict(type='LoggerHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'),
        )  
   
# custom_hooks 是一个列表。用户可以在这个字段中加入自定义的钩子，例如 EMAHook。
custom_hooks = [
    dict(
        type='EMAHook',             # 实现权重 EMA(指数移动平均) 更新的 Hook
        ema_type='ExpMomentumEMA',  # YOLO 中使用的带动量 EMA
        momentum=0.0001,            # EMA 的动量参数
        update_buffers=True,        # 是否计算模型的参数和缓冲的 running averages
        strict_load=False,
        priority=49,                # 优先级略高于 NORMAL(50)
        )
]

default_scope = 'mmyolo'    # 默认的注册器域名，默认从此注册器域中寻找模块。
env_cfg = dict(
    cudnn_benchmark=True,                  # 是否启用 cudnn benchmark, 推荐单尺度训练时开启，可加速训练
    mp_cfg=dict(mp_start_method='fork',     # 使用 fork 来启动多进程。‘fork’ 通常比 ‘spawn’ 更快，但可能存在隐患。
                opencv_num_threads=0),      # 关闭 opencv 的多线程以避免系统超负荷
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = "checkpoints/yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_230453-49564d58.pth"
resume = False
work_dir = "work_dirs/bjtds/fine_scan/e3_yolov5_m_P6_40e_20230507"
seed = 42
