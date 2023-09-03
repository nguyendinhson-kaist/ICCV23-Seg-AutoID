_base_ = './mask-rcnn_swinb_fpn_1x_vipriors.py'

model = dict(
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24]),
    neck=dict(in_channels=[96, 192, 384, 768])
)
