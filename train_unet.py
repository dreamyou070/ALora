import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
import torch
import os
from model.diffusion_model import transform_models_if_DDP
from model.unet import unet_passing_argument
from utils import get_epoch_ckpt_name, save_model, prepare_dtype, arg_as_list
from utils.attention_control import passing_argument
from utils.accelerator_utils import prepare_accelerator
from utils.optimizer_utils import get_optimizer, get_scheduler_fix
from data.prepare_dataset import call_dataset
from attention_store.normal_activator import passing_normalize_argument
from data.mvtec import passing_mvtec_argument
from model import call_model_package
from diffusers import AutoencoderKL
from safetensors import load_file

def main(args):

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    print(f' *** output_dir : {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(args.logging_dir, exist_ok=True)

    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f'\n step 2. dataset and dataloader')
    if args.seed is None: args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader = call_dataset(args)

    print(f'\n step 3. preparing accelerator')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f'\n step 4. model ')
    weight_dtype, save_dtype = prepare_dtype(args)
    text_encoder, vae, unet, network, position_embedder = call_model_package(args, weight_dtype, accelerator, True)
    # [1] text_encoder
    del vae, network, position_embedder
    # [2] vae
    #vae_base_dir = r'/home/dreamyou070/AnomalLora_OriginCode/result/MVTec/transistor/vae_train/train_vae_20240302_6_distill_recon'
    #vae_config_dir = os.path.join(vae_base_dir, 'vae_config.json')
    #with open(vae_config_dir, 'r') as f :
    #    vae_config_dict = json.load(f)
    #vae = AutoencoderKL.from_config(pretrained_model_name_or_path=vae_config_dict)
    #vae.load_state_dict(load_file(os.path.join(vae_base_dir, f'vae_models/vae_89.safetensors')))
    # [3] unet
    from model.unet import UNet2DConditionModel
    #config_dir = os.path.join(r'/home/dreamyou070/AnomalLora_OriginCode/result/MVTec/transistor/unet_train/train_unet_20240303',
    #                          'unet_config.json')
    ##unet.save_config(config_dir)
    #unet_config =


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    # step 2. dataset
    parser.add_argument('--data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument("--anomal_source_path", type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--reference_check", action='store_true')
    parser.add_argument("--latent_res", type=int, default=64)
    parser.add_argument("--use_small_anomal", action='store_true')
    parser.add_argument("--beta_scale_factor", type=float, default=0.8)
    parser.add_argument("--anomal_p", type=float, default=0.04)
    # step 3. preparing accelerator
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--lowram", action="store_true", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--d_dim", default=320, type=int)
    # step 4. model
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='facebook/diffusion-dalle')
    parser.add_argument("--clip_skip", type=int, default=None,
                        help="use output of nth layer from back of text encoder (n>=1)")
    parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                        help="max token length of text encoder (default for 75, 150 or 225) / text encoder", )
    parser.add_argument("--vae_scale_factor", type=float, default=0.18215)
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_dim", type=int, default=64, help="network dimensions (depends on each network) ")
    parser.add_argument("--network_alpha", type=float, default=4, help="alpha for LoRA weight scaling, default 1 ", )
    parser.add_argument("--network_dropout", type=float, default=None, )
    parser.add_argument("--network_args", type=str, default=None, nargs="*", )
    parser.add_argument("--dim_from_weights", action="store_true", )
    parser.add_argument("--use_multi_position_embedder", action="store_true", )

    # step 5. optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                 help="AdamW , AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov,"
                "SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP,"
                             "DAdaptLion, DAdaptSGD, AdaFactor", )
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="use 8bit AdamW optimizer(requires bitsandbytes)", )
    parser.add_argument("--use_lion_optimizer", action="store_true",
                        help="use Lion optimizer (requires lion-pytorch)", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping")
    parser.add_argument("--optimizer_args", type=str, default=None, nargs="*",
                        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") ', )
    # lr
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module")
    parser.add_argument("--lr_scheduler_args", type=str, default=None, nargs="*",
                        help='additional arguments for scheduler (like "T_max=100")')
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", help="scheduler to use for lr")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler (default is 0)", )
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                        help="Number of restarts for cosine scheduler with restarts / cosine with restarts", )
    parser.add_argument("--lr_scheduler_power", type=float, default=1,
                        help="Polynomial power for polynomial scheduler / polynomial", )
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--unet_lr', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--train_unet', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    # step 8. training
    parser.add_argument("--use_noise_scheduler", action='store_true')
    parser.add_argument('--min_timestep', type=int, default=0)
    parser.add_argument('--max_timestep', type=int, default=500)
    parser.add_argument("--save_model_as", type=str, default="safetensors",
               choices=[None, "ckpt", "pt", "safetensors"], help="format to save the model (default is .safetensors)",)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    parser.add_argument("--dataset_ex", action='store_true')
    parser.add_argument("--gen_batchwise_attn", action='store_true')
    # [0]
    parser.add_argument("--do_object_detection", action='store_true')
    parser.add_argument("--do_normal_sample", action='store_true')
    parser.add_argument("--do_anomal_sample", action='store_true')
    parser.add_argument("--do_background_masked_sample", action='store_true')
    parser.add_argument("--do_rotate_anomal_sample", action='store_true')
    # [1]
    parser.add_argument("--do_dist_loss", action='store_true')
    parser.add_argument("--mahalanobis_only_object", action='store_true')
    parser.add_argument("--mahalanobis_normalize", action='store_true')
    parser.add_argument("--dist_loss_weight", type=float, default=1.0)
    # [2]
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--do_cls_train", action='store_true')
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)
    parser.add_argument("--anomal_weight", type=float, default=1.0)
    parser.add_argument('--normal_weight', type=float, default=1.0)
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--original_normalized_score", action='store_true')
    # [3]
    parser.add_argument("--do_map_loss", action='store_true')
    parser.add_argument("--use_focal_loss", action='store_true')
    # [4]
    parser.add_argument("--test_noise_predicting_task_loss", action='store_true')
    parser.add_argument("--dist_loss_with_max", action='store_true')
    # -----------------------------------------------------------------------------------------------------------------
    parser.add_argument("--anomal_min_perlin_scale", type=int, default=0)
    parser.add_argument("--anomal_max_perlin_scale", type=int, default=3)
    parser.add_argument("--anomal_min_beta_scale", type=float, default=0.5)
    parser.add_argument("--anomal_max_beta_scale", type=float, default=0.8)
    parser.add_argument("--back_min_perlin_scale", type=int, default=0)
    parser.add_argument("--back_max_perlin_scale", type=int, default=3)
    parser.add_argument("--back_min_beta_scale", type=float, default=0.6)
    parser.add_argument("--back_max_beta_scale", type=float, default=0.9)
    parser.add_argument("--do_rot_augment", action='store_true')
    parser.add_argument("--anomal_trg_beta", type=float)
    parser.add_argument("--back_trg_beta", type=float)
    parser.add_argument("--on_desktop", action='store_true')
    parser.add_argument("--all_positional_embedder", action='store_true')
    parser.add_argument("--all_self_cross_positional_embedder", action='store_true')
    parser.add_argument("--patch_positional_self_embedder", action='store_true')
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--use_pretrained_vae", action='store_true')
    parser.add_argument("--retrain", action='store_true')
    # -----------------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    passing_normalize_argument(args)
    passing_mvtec_argument(args)
    main(args)

