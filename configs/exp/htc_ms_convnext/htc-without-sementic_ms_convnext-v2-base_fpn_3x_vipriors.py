_base_ = './htc-without-sementic_ms_convnext-v2-base_fpn_1x_vipriors.py'

train_cfg = dict(max_epochs=200)
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.05, by_epoch=True, begin=0, end=10),
    dict(
        type='CosineAnnealingLR',
        T_max=190,
        by_epoch=True,
        begin=10,
        end=200)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2.5e-4, weight_decay=0.02))