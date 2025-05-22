tag = 'stage1.5'

train_script_name = 'train_alto'
run_dir = 'runs'

###############Train#########################
dtype = "fp16"  # float32 fp16 bf16 fp8
enable_amp = dtype!="float32"

num_gpus = 8
batch_size_per_gpu = 1
total_batch_size = batch_size_per_gpu * num_gpus

# world_size = num_gpus
print_freq = 50  # don't print too often
test_freq = print_freq * 50  # also the model-saving frequency
seed = 3613
compile = False
num_max_save_models = 5
test_distributed = True
test_before_train = False

# set to true if model is differ than original
use_gradient_ckpt = True  # unet_requires_grad
find_unused_parameters = False

###############Model EMA#####################
use_ema = False
ema_beta = 0.97  # ema_update_every=20
ema_update_every = 2 * print_freq  # every update cost about 7sec

###############Optimizer#####################
gradient_accumulation_steps = 1
warm_steps = 1_000
total_steps = 300_000
max_epochs = 9999
learning_rate = 1e-5
min_lr = 5e-6  # learning_rate / 10 usually
weight_decay = 0.1
beta1 = 0.99
beta2 = 0.999  # make a bit bigger when batch size per iter is small
lr_schedule = 'cosine'
max_grad_norm = 1
vis_all = True
experiment = {
    'project': "alto_stage1.5",
    'name': "alto_stage1.5", 
    'init_weight':   "yayafengzi/ALToLLM-8B/alto.pth",
    'sam_checkpoint': "sam_vit_l_0b3195.pth" #308M
}
model = {
    'vq_model': {
        'codebook_size': 1024,
        'token_size': 12,
        'use_l2_norm': True,
        'commitment_cost': 0.01,   # 0
        # vit arch
        'vit_enc_model_size': "large",
        'vit_dec_model_size': "large",
        'vit_enc_patch_size': 16,
        'vit_dec_patch_size': 16,
        'num_latent_tokens': 32,
        'finetune_decoder': False,
        'finetune_encoder': False,
        'finetune_length': True,
        'pretrained_tokenizer_weight': "pretrained/maskgit-vqgan-imagenet-f16-256.bin"
    },
    'use_random_morphology_augment': False,
    'use_random_length': False,
    'test_length': 32,
    'use_vae': True,
    'length_exp': 0.1,
    'use_random_not_adaptive': False,
}

dataset = {
    'params': {
        'train_sam1b_path': "./example/sa1b.json",
        'eval_sam1b_path': "./example/sa1b.json",
        'num_workers_per_gpu': 12
    },
    'preprocessing': {
        'resize_shorter_edge': 256,
        'crop_size': 256,
        'random_crop': True,
        'random_flip': True,
    }
}
losses = {
    'reconstruction_weight': 1.0,   #1
    'bce_weight': 0.0, #2.0,
    'dice_weight': 0.0, #0.5,
    'quantizer_weight': 1.0,
    'length_weight': 0.01
}

