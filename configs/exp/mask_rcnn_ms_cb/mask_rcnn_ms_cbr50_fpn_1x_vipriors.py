_base_ = [
    '../../_base_/models/mask-rcnn_r50_fpn.py',
    '../../_base_/datasets/vipriors_instance.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

custom_imports=dict(imports=['modules'])

model = dict(
    type='MaskScoringRCNN',
    backbone = dict(
        type='CBResNet',
        cb_del_stages=1,
        cb_inplanes=[64, 256, 512, 1024, 2048],
        frozen_stages=-1,
        init_cfg = None),
    neck=dict(
        type='CBFPN',
    ),
    roi_head = dict(
        type='MaskScoringRoIHead',
        bbox_head = dict(num_classes = 2),
        mask_head = dict(num_classes = 2),
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=2)
    ),
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5))
)

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)