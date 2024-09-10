data_root = '/SHARE_ST/icl/Neurips2024/zeroseg_kdu_khy/data/VOCdevkit/VOC2010'
dataset_type = 'PascalContext60Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, interval=1, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    P=0.13,
    T=0.13,
    clip_path='ViT-B/16',
    kl_sizes=[
        0.8818000732589452,
        0.6986692183215663,
        0.921282873960587,
        0.31929478988312043,
        0.5740025086474836,
        0.9426262948605687,
    ],
    logit_scale=50,
    name_path='./configs/cls_context60.txt',
    prob_thd=0.5,
    type='CLIPForSegmentation')
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/SegmentationContext/val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        data_root=
        '/SHARE_ST/icl/Neurips2024/zeroseg_kdu_khy/data/VOCdevkit/VOC2010',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                448,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalContext60Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        448,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='./outputs/',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './hyper-context'
