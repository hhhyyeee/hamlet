# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: BN instead of SyncBN

# _base_ = ['../../_base_/models/segformer.py']

# model settings
# norm_cfg = dict(type='BN', requires_grad=True)
# find_unused_parameters = True
# model = dict(
#     type='OthersEncoderDecoder',
#     pretrained='pretrained/mit_b5.pth',
#     backbone=dict(
#         type='mit_b5',
#         style='pytorch',
#         aux_classifier=False
#         ),
#     decode_head=dict(
#         type='OriginalSegFormerHead',
#         in_channels=[64, 128, 320, 512],
#         in_index=[0, 1, 2, 3],
#         channels=128,
#         dropout_ratio=0.1,
#         num_classes=19,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         align_corners=False,
#         decoder_params=dict(embed_dim=768, conv_kernel_size=1),
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))

imnet_model = dict(
    type='OthersEncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch',
        aux_classifier=False
        ),
    decode_head=dict(
        type='OriginalSegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=768, conv_kernel_size=1),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
