_base_ = './htc-without-semantic_ms_cbr50_fpn_1x_vipriors.py'

train_cfg = dict(max_epochs=36)
# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.05, by_epoch=False, begin=0, end=20),
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=36,
    #     by_epoch=True,
    #     milestones=[10, 20, 30],
    #     gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.0001))