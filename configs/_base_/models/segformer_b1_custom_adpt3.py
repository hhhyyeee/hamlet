# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: BN instead of SyncBN

_base_ = ['../../_base_/models/segformer.py']

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='OthersEncoderDecoder',
    pretrained='pretrained/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch',
        pet_cls='Adapter',
        adapt_blocks=[2, 3],
        aux_classifier=True
        ),
    decode_head=dict(
        type='OriginalSegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256, conv_kernel_size=1),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
