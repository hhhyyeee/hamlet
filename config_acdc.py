wandb_project = "Hamlet-ACDC"

datasets = [ # (source, target)
    ("cityscapes", "acdc")
]

domain_order = [
    ["fog", "night", "rain", "snow"] + ["rain", "night", "fog"]
]
num_epochs = 3

models = [ # (architecture, backbone)
    # ("segformer", "mitb0_custom"),
    # ("segformer", "mitb1_custom"),
    # ("segformer", "mitb0_custom_adpt1"),

    # ("segformer", "mitb1_custom_adpt1"),
    # ("segformer", "mitb1_custom_adpt2"),
    # ("segformer", "mitb1_custom_adpt3"),
    # ("segformer", "mitb1_custom_adpt3-false"),
    # ("segformer", "mitb1_custom_adpt4"),
    # ("segformer", "mitb1_custom_adpt5-debug"),
    # ("segformer", "mitb1_custom_adpt6"),
    # ("segformer", "mitb1_custom_adpt7"),
    # ("segformer", "mitb1_custom_adpt8"),
    # ("segformer", "mitb1_custom_adpt8-debug"),
    # ("segformer", "mitb1_custom_adpt9"),

    # ("segformer", "mitb2_custom_adpt1"),
    # ("segformer", "mitb2_custom_adpt3"),
    # ("segformer", "mitb2_custom_adpt8"),

    # ("segformer", "mitb3_custom_adpt1"),
    # ("segformer", "mitb3_custom_adpt2"),
    # ("segformer", "mitb3_custom_adpt3"),
    # ("segformer", "mitb3_custom_adpt4"),
    # ("segformer", "mitb3_custom_adpt8"),
    # ("segformer", "mitb3_custom"),
    
    # ("segformer", "mitb5_custom_adpt0.evp"),
    # ("segformer", "mitb5_custom_adpt0.evp-fs"),
    # ("segformer", "mitb5_custom_adpt1"),
    # ("segformer", "mitb5_custom_adpt2"),
    # ("segformer", "mitb5_custom_adpt2.kldiv")
    # ("segformer", "mitb5_custom_adpt2.evp")
    # ("segformer", "mitb5_custom_adpt2.evp-lp")
    # ("segformer", "mitb5_custom_adpt2.evp-fs")
    # ("segformer", "mitb5_custom_adpt2.evp-fus")
    ("segformer", "mitb5_custom_adpt2.cvp")
    # ("segformer", "mitb5_custom_adpt3"),
    # ("segformer", "mitb5_custom_adpt4"),
    # ("segformer", "mitb5_custom_adpt5"),
    # ("segformer", "mitb5_custom_adpt8"),
    # ("segformer", "mitb5_custom"),

    # ("segformer", "mitb1")
    # ("segformer", "mitb5")
    # ("upernet", "swin")
]
udas = [
    "dacs_online", # Hamlet UDA
]

max_lr = [
    0.001, #1e-3
]

lr = [
    # 6e-5
    0.00015,
    # 0.000075,
    # 0.000015,
]

lr_policy = [
    "adaptive_slope",
]

lr_far_domain = [
    0.000015 * 4,
]

train = True

modular_training = [
    False,
]
training_policy = [  # options True:['MAD_UP', 'RANDOM', 1]
    "MAD_UP",
    # 1
]

alphas = [
    0.1,
]

batchnorm_trained = True  # Set to False to train lightweight decoder
train_lightweight_decoder = False

buffer = [
    1000,
]

buffer_policy = [
    "rare_class_sampling",
    # 'uniform'
]

temperature = [
    1.75,
]

mad_time_update = [
    True,
]

domain_indicator = [
    False,
]

dynamic_dacs = [
    # None,
    (0.5, 0.75)
]

base_iters = [
    750,
]

threshold_indicator = (0.23, -0.23)

reduce_training = [
    (0.25, 0.75),
]

batch_size = 1
iters = 40000

# modules_update = "random_modules/random_[0.25, 0.25, 0.25, 0.25].npy"
modules_update = "random_modules/online_random.npy"
# modules_update = None
# pretrained_segmentator = "pretrained/mitb1_uda.pth"
# pretrained_segmentator = "pretrained/mitb5_uda.pth"
# pretrained_segmentator = "pretrained/mit_b5.pth"
# pretrained_segmentator = "pretrained/segformer.b0.1024x1024.city.160k.replace.pth"
# pretrained_segmentator = "pretrained/segformer.b1.1024x1024.city.160k.pth"
# pretrained_segmentator = "pretrained/segformer.b1.1024x1024.city.160k.replace.pth"
# pretrained_segmentator = "pretrained/segformer.b2.1024x1024.city.160k.replace.pth"
# pretrained_segmentator = "pretrained/segformer.b3.1024x1024.city.160k.replace.pth"
# pretrained_segmentator = "pretrained/segformer.b5.1024x1024.city.160k.replace.pth"
# pretrained_segmentator = "pretrained/segformer.b5.1024x1024.city.160k.pth"      #segformer (evaluation)

# student_pretrained = pretrained_segmentator

# pretrained_segmentator = "pretrained/mit_b1.replace.pth"
# student_pretrained = "pretrained/segformer.b1.1024x1024.city.160k.replace.pth"

pretrained_segmentator = "pretrained/mit_b5.replace.pth"
# student_pretrained = "pretrained/segformer.b5.1024x1024.city.160k.replace.pth"
# student_pretrained = "work_dirs/a6000-d4/SegFormer/20240322_053454/iter_138000.replace.pth"
# student_pretrained = '/ssd_data1/hyewon/hamlet/work_dirs/a6000-d4/SegFormer/20240402_064146/iter_40000.replace.pth'
# student_pretrained = "work_dirs/a6000-d4/MiT-pretrain/20240321_062741/epoch_6.imnet_adapter.pth"
student_pretrained = "pretrained/segformer.b5.1024x1024.city.160k.replace.pth"
# student_pretrained = "work_dirs/a6000-d4/SegFormer/20240329_073349/iter_80000.replace.pth" #adpt0-evp
# student_pretrained = "work_dirs/a6000-d4/SegFormer/20240329_074002/latest.replace.pth" #adpt2-evp

seed = [0]
perfect_determinism = False
deterministic = False

#!DEBUG
freeze_backbone = True
pmult = False
imnet_original = "mitb5_custom"

