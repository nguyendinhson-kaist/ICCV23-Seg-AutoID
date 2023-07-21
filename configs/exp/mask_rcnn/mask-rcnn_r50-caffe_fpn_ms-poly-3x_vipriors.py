_base_ = './mask-rcnn_r50-caffe_fpn_ms-poly-1x_vipriors.py'

train_cfg = dict(max_epochs=36)
# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[10, 20, 30],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0001))

model = dict(
    backbone = dict(
        init_cfg = None))
