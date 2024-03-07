from mmseg.core import add_prefix
from mmseg.ops import resize

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class OthersEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        **cfg
        ):
        super(OthersEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        a=1
        num_module = 4
        self.decodesc_flag = "decoder_custom" in backbone
    
    def get_main_model(self):
        return self.main_model
    
    def extract_feat_decodesc(self, img):
        a=1
        x, c = self.backbone(img)
        a=1
        if self.with_neck:
            x = self.neck(x)
        
        return x, c
    
    def _decode_head_forward_train_decodesc(self, x, c, img_metas,
                                            gt_semantic_seg, seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, c, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    
    def _decode_head_forward_test_decodesc(self, x, c, img_metas):
        seg_logits = self.decode_head.forward_test(x, c, img_metas, self.test_cfg)
        return seg_logits

    def forward_train(self, img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=False):
        a=1
        if self.decodesc_flag:
            x, c = self.extract_feat_decodesc(img)
            a=1
        else:
            x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x

        if self.decodesc_flag:
            a=1
            loss_decode = self._decode_head_forward_train_decodesc(x, c, img_metas,
                                                                   gt_semantic_seg, seg_weight)
        else:
            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                          gt_semantic_seg, seg_weight)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        return losses
    
    def encode_decode(self, img, img_metas):
        a=1
        if self.decodesc_flag:
            x, c = self.extract_feat_decodesc(img)
            out = self._decode_head_forward_test_decodesc(x, c, img_metas)
        else:
            x = self.extract_feat(img)
            out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def entropy_prediction(self, img):
        x = self.extract_feat(img)

        entr, conf = self.decode_head.calculate_entropy(x)

        return {f"confidence": conf, "entropy": entr}

<<<<<<< HEAD
    # #!DEBUG
    # def rgb_to_one_hot(self, rgb_map, class_rgb_values=PALETTE):
    #     """
    #     Convert RGB label map to one-hot encoded multi-channel tensor using broadcasting.

    #     Args:
    #         rgb_map (Tensor): RGB label map tensor with shape (height, width, 3).
    #         class_rgb_values (List[Tuple[int, int, int]]): List of RGB values for each class.

    #     Returns:
    #         Tensor: One-hot encoded label tensor with shape (height, width, num_classes).
    #     """
    #     height, width, _ = rgb_map.shape
    #     num_classes = len(class_rgb_values)

    #     # Convert RGB map tensor to a tensor of shape (height, width, 1, 3)
    #     rgb_map_tensor = rgb_map.unsqueeze(2)

    #     # Convert class RGB values to a tensor of shape (1, 1, num_classes, 3)
    #     class_rgb_tensor = torch.tensor(class_rgb_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    #     # Compute element-wise equality between the RGB map tensor and class RGB tensor
    #     equality = (rgb_map_tensor == class_rgb_tensor)

    #     # Reduce along the last dimension to get one-hot encoded tensor
    #     one_hot_labels = equality.all(dim=-1).float()

    #     return one_hot_labels

    # def get_target_gt_seg(self, img, img_metas):

    #     """
    #     sample :: img_metas
    #         [
    #             {
    #                 'filename': '/data/datasets/Cityscapes/leftImg8bit/train/bremen/bremen_000243_000019_leftImg8bit.png',
    #                 'ori_filename': 'bremen/bremen_000243_000019_leftImg8bit.png',
    #                 'ori_shape': (1024, 2048, 3), 'img_shape': (512, 512, 3),
    #                 'pad_shape': (512, 512, 3),
    #                 'scale_factor': array([0.5, 0.5, 0.5, 0.5], dtype=float32),
    #                 'flip': False,
    #                 'flip_direction': 'horizontal',
    #                 'img_norm_cfg': {
    #                     'mean': array([123.675, 116.28 , 103.53 ], dtype=float32),
    #                     'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True
    #                 }
    #             }
    #         ]
    #     """

    #     seg_list = []   
    #     for _meta in img_metas:
    #         img_filepath = _meta["filename"]
    #         if "weather_datasets" in img_filepath:
    #             gt_filename = os.path.basename(img_filepath).replace('leftImg8bit', 'gtFine_color')
    #             gt_filepath = os.path.join("/data/datasets/Cityscapes/gtFine/train", gt_filename.split('_')[0], gt_filename)
    #         else:
    #             gt_filename = os.path.basename(img_filepath).replace('leftImg8bit', 'gtFine_color')
    #             gt_filepath = os.path.dirname(img_filepath).replace('leftImg8bit', 'gtFine') + '/' + gt_filename

    #         assert os.path.isfile(gt_filepath), f"no target seg map :< {gt_filepath}"

    #         seg_map = torchvision.io.read_image(gt_filepath, torchvision.io.ImageReadMode.RGB)

    #         resize_transform = torchvision.transforms.Resize(_meta["img_shape"][:2])
    #         seg_map = resize_transform(seg_map)

    #         seg_map = self.rgb_to_one_hot(seg_map.permute(1, 2, 0))
    #         seg_list.append(seg_map.permute(2, 0, 1))

    #     return torch.stack(seg_list, 0).to("cuda")

=======
>>>>>>> bc0f1fc5e91bc6afda8edc0ae0b859d42eea6023

