_base_ = './mask-rcnn_r50_fpn_1x_vipriors.py'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(ann_file='poly_train.json', pipeline=train_pipeline))

val_dataloader = dict(dataset=dict(ann_file='poly_val.json'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=_base_.data_root + 'poly_val.json')
test_evaluator = val_evaluator