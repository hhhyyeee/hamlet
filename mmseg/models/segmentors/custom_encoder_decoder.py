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


