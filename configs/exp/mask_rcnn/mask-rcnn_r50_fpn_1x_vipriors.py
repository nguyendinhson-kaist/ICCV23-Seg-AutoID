_base_ = [
    '../../_base_/models/mask-rcnn_r50_fpn.py',
    '../../_base_/datasets/vipriors_instance.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone = dict(
        frozen_stages=-1,
        init_cfg = None),
    roi_head = dict(
        bbox_head = dict(num_classes = 2),
        mask_head = dict(num_classes = 2)
    )
)