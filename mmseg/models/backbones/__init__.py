from run_experiments import CUSTOM
if CUSTOM:
    # from .mix_transformer_adapter import  (MixVisionTransformer, mit_b0, mit_b1, mit_b2,
    #                                        mit_b3, mit_b4, mit_b5)
    # from .mix_transformer_adapter_auxclf import (MixVisionTransformer, mit_b0, mit_b1, mit_b2,
    #                                              mit_b3, mit_b4, mit_b5)
    from .mix_transformer_adapter_auxclf_decodesc import (MixVisionTransformer, mit_b0, mit_b1, mit_b2,
                                                          mit_b3, mit_b4, mit_b5)
else:
    from .mix_transformer import (MixVisionTransformer, mit_b0, mit_b1, mit_b2,
                                  mit_b3, mit_b4, mit_b5)
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
# from .swin_transformer import SwinTransformer

from .mix_transformer_evp_adapter_auxclf import (MixVisionTransformerEVP, mit_b0_evp, mit_b1_evp, mit_b2_evp,
                                                 mit_b3_evp, mit_b4_evp, mit_b5_evp)

__all__ = [
    'ResNet',
    'ResNetV1c',
    'ResNetV1d',
    'ResNeXt',
    'ResNeSt',
    'MixVisionTransformer',
    'mit_b0',
    'mit_b1',
    'mit_b2',
    'mit_b3',
    'mit_b4',
    'mit_b5',
    # 'SwinTransformer'
    'MixVisionTransformerEVP',
    'mit_b5_evp'
]
