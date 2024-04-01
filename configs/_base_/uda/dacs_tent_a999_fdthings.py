# UDA with Thing-Class ImageNet Feature Distance + Increased Alpha
_base_ = ['dacs_tent.py']
uda = dict(
    alpha=0.999,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None, #이거 빼면 thing-class가 아닌것임
    imnet_feature_dist_scale_min_ratio=0.75,
    target_only=True,
    imnet_feature_dist_target_lambda=0.005,
    tent_for_dacs=True,
)
