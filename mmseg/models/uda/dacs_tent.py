# Modifications:
# - Include code to handle domain indicator
# - Modify eval phase
# - Include relevant logs

# ---------------------------------------------------------------
# Obtained from https://github.com/lhoyer/DAFormer.

# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs
# ---------------------------------------------------------------

import math
import os
import random
import time
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmcv.runner import load_checkpoint
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module, CustomUDADecorator
from mmseg.models.utils.dacs_transforms import (
    denorm,
    get_class_masks,
    get_mean_std,
    strong_transform,
)
from mmseg.models.utils.visualization import (
    subplotimg,
    colorize_mask,
    Cityscapes_palette,
)
from mmseg.utils.utils import downscale_label_ratio

from mmseg.models.uda.tent import softmax_entropy #!DEBUG


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS_TENT(CustomUDADecorator):
    def __init__(self, **cfg):
        super(DACS_TENT, self).__init__(**cfg)

        self.local_iter = 0
        # self.max_iters = cfg['max_iters']
        self.alpha = cfg["alpha"]
        self.pseudo_threshold = cfg["pseudo_threshold"]
        self.psweight_ignore_top = cfg["pseudo_weight_ignore_top"]
        self.psweight_ignore_bottom = cfg["pseudo_weight_ignore_bottom"]

        # feature distance configs
        self.fdist_lambda = cfg["imnet_feature_dist_lambda"]
        self.fdist_classes = cfg["imnet_feature_dist_classes"]
        self.fdist_scale_min_ratio = cfg["imnet_feature_dist_scale_min_ratio"]
        self.enable_fdist = self.fdist_lambda > 0

        # image mixing configs
        self.mix = cfg["mix"]
        self.blur = cfg["blur"]
        self.color_jitter_s = cfg["color_jitter_strength"]
        self.color_jitter_p = cfg["color_jitter_probability"]

        self.debug_img_interval = cfg["debug_img_interval"]
        self.print_grad_magnitude = cfg["print_grad_magnitude"]
        assert self.mix == "class"

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg["model"])
        self.ema_model = build_segmentor(ema_cfg)

        # ===== custom configs =====
        self.target_fdist_lambda = cfg.get("imnet_feature_dist_target_lambda", 0)
        self.enable_target_fdist = self.target_fdist_lambda > 0

        self.imnet_orig_flag = cfg.get("imnet_model", None) is not None

        if (self.enable_fdist or self.enable_target_fdist):
            if not self.imnet_orig_flag:
                self.imnet_model = build_segmentor(deepcopy(cfg["model"]))
            else:
                self.imnet_model = build_segmentor(deepcopy(cfg["imnet_model"]))
        else:
            self.imnet_model = None

        # If we are starting from a pretrain model
        if "segmentator_pretrained" in cfg:
            checkpoint = load_checkpoint(
                self.ema_model, cfg["student_pretrained"], map_location="cpu"
            )
            self.ema_model.CLASSES = checkpoint["meta"]["CLASSES"]
            self.ema_model.PALETTE = checkpoint["meta"]["PALETTE"]

        if self.imnet_model is not None:
            checkpoint_imnet = load_checkpoint(
                self.imnet_model, cfg["segmentator_pretrained"], map_location="cpu"
            )
            self.imnet_model.CLASSES = checkpoint_imnet["meta"]["CLASSES"]
            self.imnet_model.PALETTE = checkpoint_imnet["meta"]["PALETTE"]

        self.benchmark = "benchmark" in cfg

        # modular update configs
        self.num_module = 4
        self.modular_model = self.model_type == "ModularEncoderDecoder"
        self.current_weight = None

        self.dacs_ratio = self.fixed_dacs = 0.5
        self.use_domain_indicator = cfg["use_domain_indicator"]

        a=1
        if cfg["freeze_backbone"]:
            from mmseg.models.utils import freeze #!DEBUG
            freeze(self.model.backbone)
            freeze(self.ema_model.backbone)

        self.target_only = True #cfg.get("target_only", None)
        self.tent = cfg.get("tent_for_dacs", None)


    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(
            self.get_ema_model().parameters(), self.get_model().parameters()
        ):
            if not param.data.shape:  # scalar tensor
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = (
                    alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
                )

    # @profile
    def _val_step(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas):
        log_vars = {}
        student = self.get_model()
        student.eval()
        self.get_imnet_model().eval()
        with torch.no_grad():
            # emulate prediction for FPS and generate pseudo-label
            logits = student.encode_decode(target_img, target_img_metas)
            logits_softmax = torch.softmax(logits.detach(), dim=1)
            _, pseudo_label = torch.max(logits_softmax, dim=1)

            # confidence of static
            ps_label = pseudo_label[:, None, :, :]
            static_losses = self.get_imnet_model().forward_train(
                target_img,
                img_metas,
                ps_label,
                return_feat=True,
                # module=1,
                # confidence=True,
            )
            static_losses.pop("features")
            _, static_log_vars = self._parse_losses(static_losses, mode=self.current_weight)
            static_losses = add_prefix(static_log_vars, "static")
            log_vars.update(static_losses)
        return log_vars

    def val_step(self, data_batch, **kwargs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        log_vars = self._val_step(**data_batch)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        outputs = dict(log_vars=log_vars, num_samples=len(data_batch["img_metas"]), time=elapsed)
        return outputs

    def student_teacher_logs(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas):
        log_vars = {}
        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        with torch.no_grad():
            ema_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)

            ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
            _, pseudo_label = torch.max(ema_softmax, dim=1)

            # confidence of teacher
            teacher_losses = self.get_ema_model().entropy_prediction(target_img)
            teacher_losses = add_prefix(teacher_losses, "teacher")
            log_vars.update(teacher_losses)

    # @profile
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.get_model().train()

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        log_vars.pop("loss", None)

        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch["img_metas"]), time=elapsed
        )
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        if False: #!DEBUG
        # if self.get_training_policy() == "MAD" and self.num_module < 4:
            # print(self.num_module)
            feat_diff.requires_grad = True
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            if self.get_model().decodesc_flag:
                # decodesc mode - output looks like (outs, (c1, c2, c3, c4, c))
                # we only need `outs` here, nothing else
                feat_imnet = feat_imnet[0]
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1  # if not self.modular_model else self.num_module - 1

        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = (
                downscale_label_ratio(
                    gt, scale_factor, self.fdist_scale_min_ratio, self.num_classes, 255
                )
                .long()
                .detach()
            )
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay], fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses({"loss_imnet_feat_dist": feat_dist})
        feat_log.pop("loss", None)
        return feat_loss, feat_log

    def calc_target_feat_dist(self, img, gt=None, feat=None):
        assert self.enable_target_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            if self.get_model().decodesc_flag:
                # decodesc mode - output looks like (outs, (c1, c2, c3, c4, c))
                # we only need `outs` here, nothing else
                feat_imnet = feat_imnet[0]
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1  # if not self.modular_model else self.num_module - 1

        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = (
                downscale_label_ratio(
                    gt, scale_factor, self.fdist_scale_min_ratio, self.num_classes, 255
                )
                .long()
                .detach()
            )
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay], fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])

        feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.target_fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses({"loss_imnet_feat_dist": feat_dist})
        feat_log.pop("loss", None)
        return feat_loss, feat_log

    # @profile
    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        #!DEBUG
        target_gt_semantic_seg = kwargs.get("target_gt_semantic_seg", None)

        #  module to forward
        # main_model = self.get_main_model()

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            "mix": None,
            "color_jitter": random.uniform(0, 1),
            "color_jitter_s": self.color_jitter_s,
            "color_jitter_p": self.color_jitter_p,
            "blur": random.uniform(0, 1) if self.blur else 0,
            "mean": means[0].unsqueeze(0),  # assume same normalization
            "std": stds[0].unsqueeze(0),
        }

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        if target_gt_semantic_seg is not None:
            pseudo_prob = torch.max(target_gt_semantic_seg, dim=1)[0]
            pseudo_label = pseudo_prob
        else:
            ema_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)
            ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
            pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, : self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom :, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg, self.dacs_ratio)

        for i in range(batch_size):
            strong_parameters["mix"] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])),
            )
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])),
            )
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        if (self.target_only is not None) and (self.target_only == True):
            mixed_img = target_img
            mixed_lbl = pseudo_label.unsqueeze(1)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img,
            img_metas,
            mixed_lbl,
            pseudo_weight,
            return_feat=True,
        )
        mix_feat = mix_losses.pop("features")
        mix_losses = add_prefix(mix_losses, "mix")
        mix_loss, mix_log_vars = self._parse_losses(mix_losses, mode=self.current_weight)
        log_vars.update(mix_log_vars)

        # ImageNet `Target` feature distance
        if self.enable_target_fdist:
            feat_loss, feat_log = self.calc_target_feat_dist(target_img, pseudo_label.unsqueeze(1), mix_feat)
            log_vars.update(add_prefix(feat_log, "target_imnet"))
            # if self.print_grad_magnitude:
            #     params = self.get_model().backbone.parameters()
            #     fd_grads = [p.grad.detach() for p in params if p.grad is not None]
            #     fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
            #     grad_mag = calc_grad_magnitude(fd_grads)
            #     mmcv.print_log(f"Target Fdist Grad.: {grad_mag}", "mmseg")

        # confidence of the student of the target and simulate prediction
        self.get_model().eval()
        with torch.no_grad():
            logits = self.get_model().encode_decode(target_img, target_img_metas)
            logits_softmax = torch.softmax(logits.detach(), dim=1)
            pseudo_prob, pseudo_label_student = torch.max(logits_softmax, dim=1)

            # confidence of little static
            if self.use_domain_indicator:
                self.get_imnet_model().eval()
                ps_label = pseudo_label_student[:, None, :, :]
                static_losses = self.get_imnet_model().forward_train(
                    target_img,
                    img_metas,
                    ps_label,
                    return_feat=True,
                    # module=1,
                    # confidence=True,
                )
                static_losses.pop("features")
                _, static_log_vars = self._parse_losses(static_losses, mode=self.current_weight)
                static_losses = add_prefix(static_log_vars, "static")
                log_vars.update(static_losses)
        self.get_model().train()

        a=1
        logits = self.get_model().whole_inference(
            target_img, target_img_metas, rescale=True
        )
        tent_loss = softmax_entropy(logits)
        tent_loss = tent_loss.mean()
        log_vars.update({"tent.loss": tent_loss.item()})

        # backward pass
        if self.enable_target_fdist:
            (feat_loss + mix_loss + tent_loss).backward()
        else:
            (mix_loss + tent_loss).backward()

        # if not self.benchmark and self.local_iter % 20 == 0: #!DEBUG
        if not self.benchmark and self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg["work_dir"], "class_mix_debug")
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)

            #  get 200mm images
            if any(
                [(v in target_img_metas[0]["filename"]) \
                for v in ["200m", "fog", "night", "rain", "snow"]]):
            # if ("200m" in target_img_metas[0]["filename"]) or ("night" in target_img_metas[0]["filename"]):
                #  target
                def plot_img(img, metas, domain):
                    img = img[0].cpu()
                    img = img.permute(1, 2, 0)
                    img = img.numpy()
                    name = "_".join(metas[0]["ori_filename"].split("/")[-1].split("_")[:-1])
                    if domain == "mixed":
                        name = ""
                    plt.imsave(
                        os.path.join(
                            out_dir,
                            f"{domain}_img_{name}_{(self.local_iter + 1):06d}.png",
                        ),
                        img,
                    )
                    return name

                target_name = plot_img(vis_trg_img, target_img_metas, "target")
                source_name = plot_img(vis_img, img_metas, "source")
                plot_img(vis_mixed_img, target_img_metas, "mixed")
                #  pseudo-label
                img = pseudo_label[0].cpu()
                img = img.numpy()
                img = colorize_mask(img, Cityscapes_palette)
                img.save(
                    os.path.join(
                        out_dir,
                        f"psuedolabel_{target_name}_{(self.local_iter + 1):06d}.png",
                    )
                )
                #  source gt
                img = gt_semantic_seg[0].cpu()
                img = img.numpy()
                img = img.squeeze(0)
                img = colorize_mask(img, Cityscapes_palette)
                img.save(
                    os.path.join(
                        out_dir,
                        f"source_gt_{source_name}_{(self.local_iter + 1):06d}.png",
                    )
                )
                #  mixed gt
                img = mixed_lbl[0].cpu()
                img = img.numpy()
                img = img.squeeze(0)
                img = colorize_mask(img, Cityscapes_palette)
                img.save(os.path.join(out_dir, f"mixed_gt_{(self.local_iter + 1):06d}.png"))
                #  student prediction
                img = pseudo_label_student[0].cpu()
                img = img.numpy()
                img = colorize_mask(img, Cityscapes_palette)
                img.save(
                    os.path.join(out_dir, f"student_prediction_{(self.local_iter + 1):06d}.png")
                )

            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        "hspace": 0.1,
                        "wspace": 0,
                        "top": 0.95,
                        "bottom": 0,
                        "right": 1,
                        "left": 0,
                    },
                )
                subplotimg(axs[0][0], vis_img[j], "Source Image")
                subplotimg(axs[1][0], vis_trg_img[j], "Target Image")
                subplotimg(axs[0][1], gt_semantic_seg[j], "Source Seg GT", cmap="cityscapes")
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    "Target Seg (Pseudo) GT",
                    cmap="cityscapes",
                )
                subplotimg(axs[0][2], vis_mixed_img[j], "Mixed Image")
                subplotimg(axs[1][2], mix_masks[j][0], "Domain Mask", cmap="gray")
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(axs[1][3], mixed_lbl[j], "Seg Targ", cmap="cityscapes")
                subplotimg(axs[0][3], pseudo_weight[j], "Pseudo W.", vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        "FDist Mask",
                        cmap="gray",
                    )
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        "Scaled GT",
                        cmap="cityscapes",
                    )
                for ax in axs.flat:
                    ax.axis("off")
                plt.savefig(os.path.join(out_dir, f"{(self.local_iter + 1):06d}_{j}.png"))
                # save_meta_imgname = target_img_metas[j]["ori_filename"].split('/')[-1].replace(".png", "")
                # plt.savefig(os.path.join(out_dir, f"{(self.local_iter + 1):06d}_{j}_{save_meta_imgname}.png"))
                plt.close()

        self.local_iter += 1
        return log_vars
