import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
import torch
import os
from model.query_transformer import QueryTransformer
from attention_store import AttentionStore
from attention_store.normal_activator import NormalActivator
from model.diffusion_model import transform_models_if_DDP
from model.unet import unet_passing_argument
from utils import get_epoch_ckpt_name, save_model, prepare_dtype, arg_as_list
from utils.attention_control import passing_argument, register_attention_control
from utils.accelerator_utils import prepare_accelerator
from utils.optimizer_utils import get_optimizer, get_scheduler_fix
from utils.model_utils import pe_model_save, te_model_save
from utils.utils_loss import FocalLoss
from data.prepare_dataset import call_dataset
from model import call_model_package
from attention_store.normal_activator import passing_normalize_argument
from data.mvtec import passing_mvtec_argument
from model.global_local_segmentation import SegmentationSubNetwork
from torch import nn


def resize_query_features(query):
    # 8, 64, 160
    head_num, pix_num, dim = query.shape
    res = int(pix_num ** 0.5)  # 8
    # 8,160,8,8
    query_map = query.view(head_num, res, res, dim).permute(0, 3, 1, 2).contiguous()  # 1, channel, res, res
    resized_query_map = nn.functional.interpolate(query_map, size=(64, 64), mode='bilinear')  # head, 160, 64,  64
    resized_query = resized_query_map.permute(0, 2, 3, 1).contiguous().squeeze()  # head, 64, 64, 160
    resized_query = resized_query.view(head_num, 64 * 64,
                                       dim)  # #view(head_num, -1, dim).squeeze()  # head, pix_num, 160
    # resized_query = resized_query.view(64 * 64,dim)  # #view(head_num, -1, dim).squeeze()  # 1, pix_num, dim
    return resized_query


def reshape_batch_dim_to_heads(tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = 8
    res = int(seq_len ** 0.5)
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    tensor = tensor.reshape(batch_size // head_size, res, res, dim * head_size).permute(0,3,1,2)
    return tensor

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
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader = call_dataset(args)

    print(f'\n step 3. preparing accelerator')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f'\n step 4. model ')
    weight_dtype, save_dtype = prepare_dtype(args)
    l_text_encoder, l_vae, l_unet, l_network, l_position_embedder = call_model_package(args, weight_dtype, accelerator, True)
    g_text_encoder, g_vae, g_unet, g_network, g_position_embedder = call_model_package(args, weight_dtype, accelerator,False)

    class UNetUpBlock(nn.Module):
        def __init__(self, in_size, out_size):
            super(UNetUpBlock, self).__init__()

            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(out_size, out_size, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(out_size, out_size, kernel_size=1), )
            # self.conv_block = UNetConvBlock(out_size, out_size, padding, batch_norm)
            self.dim = out_size

        def forward(self, x):
            #
            h, pix_num, d = x.shape
            res = int(pix_num ** 0.5)
            x = x.reshape(h, res, res, d).permute(0,3,1,2)
            out = self.up(x)  # head, dim, res, res
            out = out.permute(0, 2, 3, 1)  # head, res, res, dim
            import einops
            out = einops.rearrange(out, 'h a b d -> h (a b) d')
            return out  # head, pix_num, dim

    gquery_transformer = UNetUpBlock(in_size=160, out_size=280)
    segmentation_net = SegmentationSubNetwork(in_channels=4480,
                                              out_channels=1,
                                              base_channels=64)

    print(f'\n step 5. optimizer')
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    trainable_params = g_network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    trainable_params.append({"params": g_position_embedder.parameters(), "lr": args.learning_rate})
    trainable_params.append({"params": gquery_transformer.parameters(), "lr": args.learning_rate})
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)

    print(f'\n step 6. lr')
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 7. loss function')
    loss_focal = FocalLoss()
    loss_l2 = torch.nn.modules.loss.MSELoss(reduction='none')
    normal_activator = NormalActivator(loss_focal, loss_l2, args.use_focal_loss)

    print(f'\n step 8. model to device')
    g_unet, g_text_encoder, g_network, optimizer, train_dataloader, lr_scheduler, g_position_embedder, gquery_transformer, segmentation_net = accelerator.prepare(
        g_unet, g_text_encoder, g_network, optimizer, train_dataloader, lr_scheduler, g_position_embedder, gquery_transformer,segmentation_net)

    g_text_encoders = transform_models_if_DDP([g_text_encoder])
    g_unet, g_network = transform_models_if_DDP([g_unet, g_network])
    if args.gradient_checkpointing:
        g_unet.train()
        g_position_embedder.train()
        for t_enc in g_text_encoders:
            t_enc.train()
            if args.train_text_encoder:
                t_enc.text_model.embeddings.requires_grad_(True)
        if not args.train_text_encoder:  # train U-Net only
            g_unet.parameters().__next__().requires_grad_(True)
    else:
        g_unet.eval()
        for t_enc in g_text_encoders:
            t_enc.eval()
    del t_enc
    g_network.prepare_grad_etc(g_text_encoder, g_unet)
    g_vae.to(accelerator.device, dtype=weight_dtype)

    l_unet = l_unet.to(accelerator.device, dtype=weight_dtype)
    l_unet.eval()
    l_text_encoder = l_text_encoder.to(accelerator.device, dtype=weight_dtype)
    l_text_encoder.eval()
    l_vae = l_vae.to(accelerator.device, dtype=weight_dtype)
    l_vae.eval()
    l_position_embedder.to(accelerator.device, dtype=weight_dtype)
    l_position_embedder.eval()
    l_network.to(accelerator.device, dtype=weight_dtype)
    l_network.eval()

    print(f'\n step 9. registering saving tensor')
    g_controller = AttentionStore()
    register_attention_control(g_unet, g_controller)
    l_controller = AttentionStore()
    register_attention_control(l_unet, l_controller)

    print(f'\n step 9. Training !')
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []

    for epoch in range(args.start_epoch, args.max_train_epochs):

        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            device = accelerator.device
            loss_dict = {}
            with torch.set_grad_enabled(True):
                encoder_hidden_states = l_text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
            """ Train Only With Normal Data (normal feature matching, anomal feature unmatching..?) """
            if args.do_normal_sample :
                with torch.no_grad():
                    latents = l_vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample() * args.vae_scale_factor
                    l_unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,noise_type=l_position_embedder,)
                    l_query_dict, l_key_dict = l_controller.query_dict, l_controller.key_dict
                    l_controller.reset()
                    l_query_list = []
                    for layer in args.trg_layer_list :
                        if 'mid' not in layer :
                            l_query_list.append(resize_query_features(l_query_dict[layer][0].squeeze()))
                    local_query = torch.cat(l_query_list, dim=-1)  # 8, 64*64, 280
                with torch.set_grad_enabled(True):
                    g_unet(latents,0,encoder_hidden_states,trg_layer_list=args.trg_layer_list, noise_type=g_position_embedder)
                g_query_dict, g_key_dict = g_controller.query_dict, g_controller.key_dict
                g_controller.reset()
                for layer in args.trg_layer_list :
                    if 'mid' in layer :
                        g_query = g_query_dict[layer][0].squeeze()
                global_query = gquery_transformer(g_query) # g_query = 8, 64, 160 -> 8, 64*64, 280
                # matching loss
                matching_loss = loss_l2(local_query.float(), global_query.float()) # [8, 64*64, 280]
                # matching throug segmentation
                """
                local_map  = reshape_batch_dim_to_heads(local_query)  # [1,64,64,2240]
                global_map = reshape_batch_dim_to_heads(global_query) # [1,64,64,2240]
                anomal_map = segmentation_net(torch.cat([local_map,
                                                         global_map], dim = 1))
                trg_anomal_map = torch.zeros(1,1,64,64)
                anomal_map_loss = loss_l2(anomal_map.float(),
                                          trg_anomal_map.float().to(anomal_map.device)) # [1,1,64,64]
                """

            # -------------------------------------------------------------------------------------------------------- #
            """
            if args.do_anormal_sample :
                with torch.no_grad():
                    latents = l_vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample() * args.vae_scale_factor
                    l_unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,noise_type=l_position_embedder,)
                    l_query_dict, l_key_dict = l_controller.query_dict, l_controller.key_dict
                    l_controller.reset()
                    l_query_list = []
                    for layer in args.trg_layer_list :
                        if 'mid' not in layer :
                            l_query_list.append(resize_query_features(l_query_dict[layer][0].squeeze()))
                    local_query = torch.cat(l_query_list, dim=-1)  # 8, 64*64, 280
                with torch.set_grad_enabled(True):
                    g_unet(latents,0,encoder_hidden_states,trg_layer_list=args.trg_layer_list, noise_type=g_position_embedder)
                g_query_dict, g_key_dict = g_controller.query_dict, g_controller.key_dict
                g_controller.reset()
                for layer in args.trg_layer_list :
                    if 'mid' in layer :
                        g_query = g_query_dict[layer][0].squeeze()
                global_query = gquery_transformer(g_query) # g_query = 8, 64, 160 -> 8, 64*64, 280
                # matching loss
                matching_loss = loss_l2(local_query.float(), global_query.float()) # [8, 64*64, 280]
                # matching throug segmentation
                local_map  = reshape_batch_dim_to_heads(local_query)  # [1,64,64,2240]
                global_map = reshape_batch_dim_to_heads(global_query) # [1,64,64,2240]
                anomal_map = segmentation_net(torch.cat([local_map,
                                                         global_map], dim = 1))
                trg_anomal_map = torch.zeros(1,1,64,64)
                anomal_map_loss = loss_l2(anomal_map.float(),
                                          trg_anomal_map.float().to(anomal_map.device)) # [1,1,64,64]
            """
            loss = matching_loss.mean()  #+ anomal_map_loss.mean()
            loss = loss.to(weight_dtype)
            current_loss = loss.detach().item()
            if epoch == args.start_epoch:
                loss_list.append(current_loss)
            else:
                epoch_loss_total -= loss_list[step]
                loss_list[step] = current_loss
            epoch_loss_total += current_loss
            avr_loss = epoch_loss_total / len(loss_list)
            loss_dict['avr_loss'] = avr_loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if is_main_process:
                progress_bar.set_postfix(**loss_dict)
            normal_activator.reset()
            if global_step >= args.max_train_steps:
                break
        # ----------------------------------------------------------------------------------------------------------- #
        # [6] epoch final
        accelerator.wait_for_everyone()
        if is_main_process:
            ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
            save_model(args, ckpt_name, accelerator.unwrap_model(g_network), save_dtype)
            if g_position_embedder is not None:
                position_embedder_base_save_dir = os.path.join(args.output_dir, 'position_embedder')
                os.makedirs(position_embedder_base_save_dir, exist_ok=True)
                p_save_dir = os.path.join(position_embedder_base_save_dir,
                                          f'position_embedder_{epoch + 1}.safetensors')
                pe_model_save(accelerator.unwrap_model(g_position_embedder), save_dtype, p_save_dir)
            # saving query transformer
            query_transformer_save_dir = os.path.join(args.output_dir, 'query_transformer')
            os.makedirs(query_transformer_save_dir, exist_ok = True)
            qt_save_dir = os.path.join(query_transformer_save_dir,f'query_transformer_{epoch + 1}.safetensors')
            def qt_model_save(model, save_dtype, save_dir):
                state_dict = model.state_dict()
                for key in list(state_dict.keys()):
                    v = state_dict[key]
                    v = v.detach().clone().to("cpu").to(save_dtype)
                    state_dict[key] = v
                _, file = os.path.split(save_dir)
                if os.path.splitext(file)[1] == ".safetensors":
                    from safetensors.torch import save_file
                    save_file(state_dict, save_dir)
                else:
                    torch.save(state_dict, save_dir)
            qt_model_save(accelerator.unwrap_model(gquery_transformer), save_dtype, qt_save_dir)
    accelerator.end_training()

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
    # -----------------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    passing_normalize_argument(args)
    passing_mvtec_argument(args)
    main(args)