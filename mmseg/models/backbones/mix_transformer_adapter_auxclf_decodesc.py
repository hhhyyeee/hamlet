# Obtained from: https://github.com/NVlabs/SegFormer
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# Modifications:
# - Only forward first module if module == 1

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.models.builder import BACKBONES
from mmseg.models.pet import Adapter, KVLoRA
from mmseg.models.pet_mixin import AdapterMixin
from mmseg.utils import get_root_logger

import timm

from mmseg.models.pet.vitadapter import SpatialPriorModule, Injector, deform_inputs
from torch.nn.init import normal_

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module, AdapterMixin):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim                              #512
        self.num_heads = num_heads                  #8
        head_dim = dim // num_heads                 #64
        self.scale = qk_scale or head_dim**-0.5     #8

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module, AdapterMixin):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, H, W):
        a=1

        # kwargs = {"Adapter": [H, W]}
        kwargs = {"H": H, "W": W}
        # kwargs = {}

        x = x + self.drop_path(self.attn(self.norm1(x), H, W)) # MSA
        x = x + self.drop_path(
            self.adapt_module("mlp", self.norm2(x), **kwargs)  # MLP in AdaptFormer
        )

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W


@BACKBONES.register_module()
class MixVisionTransformer(BaseModule):

    def __init__(self,

                 #default
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 style=None,
                 pretrained=None,
                 init_cfg=None,
                 freeze_patch_embed=False,
                 **cfg):
        super().__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        # --- PET
        a=1
        PET = "adapt_blocks" in cfg
        if PET:
            adapt_blocks = cfg["adapt_blocks"]
            pet_cls = cfg["pet_cls"]
            pet_kwargs = {"scale": None}

            a=1
            self.embed_dims_adapter = [_dim for idx, _dim in enumerate(embed_dims) if idx in adapt_blocks]

        # --- auxiliary classifier
        self.clf_flag = cfg["aux_classifier"]
        if self.clf_flag:
            a=1
            _embed_dim = embed_dims[2]
            # vit_adapter_embed_dim = embed_dims[1] #(64, 128, 320, 512)
            self.stem = SpatialPriorModule(embed_dim=_embed_dim)
            self.injector = Injector(dim=_embed_dim, num_heads=8, n_levels=3) #embed_dims[2]가 320인데 320/8=40으로 딱 떨어질 수 있도록 num_heads 설정

            self.level_embed = nn.Parameter(torch.zeros(3, _embed_dim))
            normal_(self.level_embed)
        
        # --- custom decoder
        self.decoder_custom = "decoder_custom" in cfg
        a=1

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        if PET:
            #!DEBUG (order matters)
            self.adapt_blocks = adapt_blocks
            self.pet_cls = pet_cls
            self.pet_kwargs = pet_kwargs

            # self.pets_emas = nn.ModuleList([])
            self.pets = self.create_pets()
            # logger

            self.attach_pets_mit()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = get_root_logger()
        if self.pretrained is None:
            logger.info('Init mit from scratch.')
            for m in self.modules():
                self._init_weights(m)
        elif isinstance(self.pretrained, str):
            logger.info('Load mit checkpoint.')
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze_or_not_modules(self, modules: list, requires_grad=False):
        if not modules:
            return
        for k, v in self._modules.items():
            for m in modules:
                if str(m) in k:
                    for p in v.parameters():
                        p.requires_grad = requires_grad

    def forward_features_orig(self, x, modules):
        B = x.shape[0]
        outs = []

        # stage 1
        if 1 in modules:
            x, H, W = self.patch_embed1(x)
            for i, blk in enumerate(self.block1):
                x = blk(x, H, W)
            x = self.norm1(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        # stage 2
        if 2 in modules:
            x, H, W = self.patch_embed2(x)
            for i, blk in enumerate(self.block2):
                x = blk(x, H, W)
            x = self.norm2(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        # stage 3
        if 3 in modules:
            x, H, W = self.patch_embed3(x)
            for i, blk in enumerate(self.block3):
                x = blk(x, H, W)
            x = self.norm3(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        # stage 4
        if 4 in modules:
            x, H, W = self.patch_embed4(x)
            for i, blk in enumerate(self.block4):
                x = blk(x, H, W)
            x = self.norm4(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward_features(self, x):
        if not self.clf_flag:
            return self.forward_features_orig(x, modules=[1, 2, 3, 4])

        a=1
        # stage 0: clone input x

        # === stem 입력으로 _x라는 클론 텐서를 넣기 위한 몸부림... 하지만 실패
        # _x = x.detach().clone()
        # _x = torch.Tensor.new_tensor(x, requires_grad=True)
        # _x = torch.tensor(x)
        # === 그냥 x를 넣어보자...

        c1, c2, c3, c4 = self.stem(x) # c3 사용시 backprop 문제없이 됨 -> 문제없을듯
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        a=1

        deform_inputs1, deform_inputs2 = deform_inputs(x)
        B = x.shape[0]
        outs = []

        # stage 1-1: MiT block 1~2
        # block 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # block 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        # inj_x = x.detach().clone()
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 1-2: aux_clf (STEM?)
        a=1

        # block 3
        x, H, W = self.patch_embed3(x)
        # inj_x = x.detach().clone()
        # inj_x = torch.Tensor(inj_x, requires_grad=True)
        # inj_x = torch.tensor(x)

        # (+) stage 1-3: fusion (INJECTOR?)
        a=1
        # x = self.injector(query=x, feat=c, H=H, W=W)
        x = self.injector(query=x, reference_points=deform_inputs1[0], feat=c,
                          spatial_shapes=deform_inputs1[1], level_start_index=deform_inputs1[2])

        # x = x + c3 #!DEBUG

        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # block 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs
 

    def forward_features_ordered(self, x):
        # if not self.clf_flag:
        #     return self.forward_features_orig(x, modules=[1, 2, 3, 4])

        # stage 0: aux_clf feature extract
        c1, c2, c3, c4 = self.stem(x) # c3 사용시 backprop 문제없이 됨 -> 문제없을듯
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        deform_inputs1, deform_inputs2 = deform_inputs(x)
        B = x.shape[0]
        outs = []

        # stage 1-1: MiT block 1~2
        # block 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # block 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # block 3
        x, H, W = self.patch_embed3(x)

        # (+) stage 1-3: fusion (INJECTOR?)
        x = self.injector(query=x, reference_points=deform_inputs1[0], feat=c,
                          spatial_shapes=deform_inputs1[1], level_start_index=deform_inputs1[2])

        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # block 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs, (c1, c2, c3, c4, c)


    def forward(self, x, module=4):
        modules = range(1, module + 1)
        if self.decoder_custom:
            x = self.forward_features_ordered(x)
        else:
            x = self.forward_features(x)
        a=1
        # x = self.head(x)

        return x

    # --- LAE pets
    def create_pets(self):
        return self.create_pets_mit()

    def create_pets_mit(self):
        assert self.pet_cls in ["Adapter", "LoRA", "Prefix"], "ERROR 1"

        n = len(self.adapt_blocks)
        embed_dims = self.embed_dims_adapter
        depths = self.depths

        assert len(embed_dims) == n, "ERROR 2"

        if self.pet_cls == "Adapter":
            adapter_list_list = []
            for idx, (_block_idx, embed_dim) in enumerate(zip(self.adapt_blocks, embed_dims)):
                adapter_list = []
                for _ in range(depths[_block_idx]):
                    kwargs = dict(**self.pet_kwargs)
                    kwargs["embed_dim"] = embed_dim
                    # adapter_list.append(Adapter(**kwargs))
                    adapter_list.append(Adapter(**kwargs))
                adapter_list_list.append(nn.ModuleList(adapter_list))
            a=1
            # return nn.ModuleList(adapter_list)
            return adapter_list_list

        if self.pet_cls == "LoRA":
            lora_list = []
            for idx, embed_dim in enumerate(embed_dims):
                kwargs = dict(**self.pet_kwargs)
                kwargs["in_features"] = embed_dim
                kwargs["out_features"] = embed_dim
                lora_list.append(KVLoRA(**kwargs))
            return nn.ModuleList(lora_list)
    
    def attach_pets_mit(self):
        assert self.pet_cls in ["Adapter", "LoRA", "Prefix"], "ERROR 1"

        pets = self.pets
        if self.pet_cls == "Adapter":
            for _idx, (_dim_idx, _dim) in enumerate(zip(self.adapt_blocks, self.embed_dims_adapter)):
            # for _dim_idx, _dim in enumerate(self.embed_dims_adapter):
                for _depth_idx in range(self.depths[_dim_idx]):
                    eval(f"self.block{_dim_idx + 1}")[_depth_idx].attach_adapter(mlp=pets[_idx][_depth_idx])
            return

        if self.pet_cls == "LoRA":
            for i, b in enumerate(self.adapt_blocks):
                a=1
                eval(f"self.block{b}").attn.attch_adapter(qkv=pets[i])

    # --- ViTAdapter
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4



class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


@BACKBONES.register_module()
class mit_b0(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b1(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b2(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b3(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b4(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b5(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)
