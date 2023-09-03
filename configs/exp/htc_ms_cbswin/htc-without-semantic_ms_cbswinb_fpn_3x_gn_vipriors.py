_base_ = './htc-without-semantic_ms_cbswinb_fpn_1x_gn_vipriors.py'

train_cfg = dict(max_epochs=720)
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.05, by_epoch=True, begin=0, end=10),
    dict(
        type='CosineAnnealingLR',
        T_max=710,
        by_epoch=True,
        begin=10,
        end=720)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.02))