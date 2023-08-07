# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

metainfo = {
    'classes': ('human','ball', ),
}

custom_imports=dict(imports=['utils.transforms'])

img_size = (1920, 1440) # w, h

# imagenet mean/std
img_mean = (123.675, 116.28, 103.53)
img_std = (58.395, 57.12, 57.375)

# if you want to use built-in CopyPaste in mmdet, uncomment below config
# load_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='RandomChoiceResize', 
#         scales=[
#             (3680, 3080), (3200, 2400),
#             (2680, 2080), (2000, 1400),
#             (1920, 1440), (1800, 1200),
#             (1600, 1024), (1333, 800),
#             (1624, 1234), (2336, 1752), (2456, 2054)], # last 3 scales are original resolution of image
#         keep_ratio=True),
#     # photometric transform
#     dict(type='PhotoMetricDistortion'),
#     # geometric transform
#     dict(type='ShearX', max_mag=5.0, prob=0.2, img_border_value=img_mean),
#     dict(type='ShearY', max_mag=5.0, prob=0.2, img_border_value=img_mean),
#     dict(type='TranslateX', max_mag=0.05, prob=0.2, img_border_value=img_mean),
#     dict(type='TranslateY', max_mag=0.05, prob=0.2, img_border_value=img_mean),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomCrop',
#         crop_size=img_size,
#         recompute_bbox=True,
#         allow_negative_crop=True),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type='Pad', size=img_size)
# ]
# train_pipeline = [
#     dict(type='CopyPaste', max_num_pasted=100),
#     dict(type='PackDetInputs')
# ]
# train_dataloader = dict(
#     batch_size=8,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type='MultiImageMixDataset',
#         dataset=dict(
#             type='CocoDataset',
#             data_root='data/',
#             ann_file='train.json',
#             data_prefix=dict(img='train/'),
#             metainfo=dict(classes=(
#                 'human',
#                 'ball',
#             )),
#             filter_cfg=dict(filter_empty_gt=True, min_size=32),
#             pipeline=load_pipeline,
#             backend_args=None),
#         pipeline=train_pipeline
#     ))

# config for using custom copypaste
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='SpecialCopyPaste', 
        crop_dir='data/cropped_objects', 
        crop_anno='crop.json',
        max_num_objects=[10,20]),
    # dict(type='Resize', scale=(
    #     1920,
    #     1440,
    # ), keep_ratio=True),
    dict(
        type='RandomChoiceResize', 
        scales=[
            (3680, 3080), (3200, 2400),
            (2680, 2080), (2000, 1400),
            (1920, 1440), (1800, 1200),
            (1600, 1024), (1333, 800),
            (1624, 1234), (2336, 1752), (2456, 2054)], # last 3 scales are original resolution of image
        keep_ratio=True),
    # photometric transform
    dict(type='PhotoMetricDistortion'),
    # geometric transform
    dict(type='ShearX', max_mag=5.0, prob=0.2, img_border_value=img_mean),
    dict(type='ShearY', max_mag=5.0, prob=0.2, img_border_value=img_mean),
    dict(type='TranslateX', max_mag=0.05, prob=0.2, img_border_value=img_mean),
    dict(type='TranslateY', max_mag=0.05, prob=0.2, img_border_value=img_mean),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomCrop',
        crop_size=img_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # dict(type='Pad', size=img_size),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        metainfo = metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        metainfo = metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric=['segm', 'bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='test.json',
#         data_prefix=dict(img='test/'),
#         metainfo = metainfo,
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric=['segm', 'bbox'],
#     format_only=True,
#     ann_file=data_root + 'test.json',
#     outfile_prefix='./work_dirs/coco_instance/test')
