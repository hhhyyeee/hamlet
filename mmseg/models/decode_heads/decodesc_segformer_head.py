# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications from https://github.com/lhoyer/DAFormer: Model construction with loop
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


@HEADS.register_module()
class DecodeSCSegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    def __init__(self, **kwargs):
        super(DecodeSCSegFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        conv_kernel_size = decoder_params['conv_kernel_size']

        self.linear_c = {}
        for i, in_channels in zip(self.in_index, self.in_channels):
            self.linear_c[str(i)] = MLP(
                input_dim=in_channels, embed_dim=embedding_dim)
        self.linear_c = nn.ModuleDict(self.linear_c)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * len(self.in_index) + 320,
            # in_channels=embedding_dim * len(self.in_index),
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=kwargs['norm_cfg'])

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

        # self.test_conv = nn.Conv2d(
        #     in_channels=105, out_channels=768, kernel_size=3, padding=1)

    def forward(self, inputs, conv_feats):
        a=1
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     print(f.shape)

        # conv
        a=1
        # conv_feat_conv = conv_feats[0]
        # conv_feat = conv_feats[-1].reshape(1, -1, 128, 128)
        # conv_feat_conv = self.test_conv(conv_feat)
        conv_feat = conv_feats[0]
        # if conv_feat.shape[-1] != 128:
        #     conv_feat = nn.functional.interpolate(conv_feat, size=(128, 128))
        #     # conv_feat = conv_feat.reshape(conv_feat.shape[0], conv_feat.shape[1], conv_feat.shape[2], 128)
        a=1

        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}, {self.linear_c[str(i)]}')
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        a=1
        _c[5] = conv_feat

        _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        x = self.linear_pred(x)

        return x

    def forward_train(self, inputs, conv_feats, img_metas, gt_semantic_seg,
                      train_cfg, confidence=None, seg_weight=None):
        a=1
        seg_logits = self.forward(inputs, conv_feats)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        return losses

    def forward_test(self, inputs, conv_feats, img_metas, test_cfg):
        a=1
        return self.forward(inputs, conv_feats)

    #!DEBUG
    def calculate_entropy(self, inputs):
        seg_logits = self.forward(inputs)
        probs = torch.nn.functional.softmax(seg_logits, dim=1)
        return self.entropy(probs).item(), self.confidence(probs).item()
