import os
import argparse, torch
from model.lora import LoRANetwork,LoRAInfModule
from attention_store import AttentionStore
from utils.attention_control import passing_argument
from model.unet import unet_passing_argument
from utils.attention_control import register_attention_control
from accelerate import Accelerator
from model.tokenizer import load_tokenizer
from utils import prepare_dtype
from utils.model_utils import get_input_ids
from PIL import Image
import numpy as np
from model.diffusion_model import load_target_model
from model.pe import AllPositionalEmbedding
from safetensors.torch import load_file
from attention_store.normal_activator import NormalActivator
from attention_store.normal_activator import passing_normalize_argument
from torch import nn


def resize_query_features(query):
    # pix_num, dim = query.shape
    head_num, pix_num, dim = query.shape
    res = int(pix_num ** 0.5)  # 8
    # query_map = query.view(res, res, dim).permute(2,0,1).contiguous().unsqueeze(0)           # 1, channel, res, res
    query_map = query.view(head_num, res, res, dim).permute(0, 3, 1, 2).contiguous()  # 1, channel, res, res
    resized_query_map = nn.functional.interpolate(query_map, size=(64, 64), mode='bilinear')  # 1, channel, 64,  64
    resized_query = resized_query_map.permute(0, 2, 3, 1).contiguous().squeeze()  # head, 64, 64, channel
    resized_query = resized_query.view(head_num, 64 * 64,
                                       dim)  # #view(head_num, -1, dim).squeeze()  # head, pix_num, dim
    # resized_query = resized_query.view(64 * 64,dim)  # #view(head_num, -1, dim).squeeze()  # 1, pix_num, dim
    return resized_query

def prev_step(model_output,
              timestep: int,
              sample,
              scheduler):
    #timestep, prev_timestep = timestep, max( timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 0)
    timestep, prev_timestep = 1,0
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    beta_prod_t = 1 - alpha_prod_t
    prev_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    prev_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * prev_original_sample + prev_sample_direction
    return prev_sample

from diffusers import DDIMScheduler
scheduler = DDIMScheduler(num_train_timesteps=1000,
                                      beta_start=0.00085,
                                      beta_end=0.012,
                                      beta_schedule="scaled_linear")

def inference(latent,
              tokenizer, text_encoder, unet, controller, normal_activator, position_embedder,
              args, org_h, org_w, thred, global_conv_net):
    # [1] text
    input_ids, attention_mask = get_input_ids(tokenizer, args.prompt)
    encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device))["last_hidden_state"]
    # [2] unet
    if args.use_position_embedder and args.use_global_conv:
        unet(latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
             noise_type=[position_embedder, global_conv_net])
    elif args.use_position_embedder and not args.use_global_conv:
        unet(latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
             noise_type=position_embedder,)
    else:
        unet(latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, )



    query_dict, key_dict, attn_dict = controller.query_dict, controller.key_dict, controller.attn_dict
    controller.reset()
    attn_list, origin_query_list, query_list, key_list = [], [], [], []
    for layer in args.trg_layer_list:
        query = query_dict[layer][0].squeeze()  # head, pix_num, dim
        origin_query_list.append(query)
        query_list.append(resize_query_features(query))  # head, pix_num, dim
        key_list.append(key_dict[layer][0])  # head, pix_num, dim
        # attn_list.append(attn_dict[layer][0])
    # [1] local
    local_query = torch.cat(query_list, dim=-1)  # head, pix_num, long_dim
    local_key = torch.cat(key_list, dim=-1).squeeze()  # head, 77, long_dim
    attention_scores = torch.baddbmm(
        torch.empty(local_query.shape[0], local_query.shape[1], local_key.shape[1], dtype=query.dtype,
                    device=query.device),
        local_query, local_key.transpose(-1, -2),
        beta=0, )
    attn_score = attention_scores.softmax(dim=-1)[:, :, :2]

    cls_score, trigger_score = attn_score.chunk(2, dim=-1)  # [head,pixel], [head,pixel]
    cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # [head,pixel], [head,pixel]
    cls_map, trigger_map = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
    pix_num = trigger_map.shape[0]
    res = int(pix_num ** 0.5)
    cls_map = cls_map.unsqueeze(0).view(res, res)
    cls_map_pil = Image.fromarray((255 * cls_map).cpu().detach().numpy().astype(np.uint8)).resize((org_h, org_w))
    normal_map = torch.where(trigger_map > thred, 1, trigger_map).squeeze()
    normal_map = normal_map.unsqueeze(0).view(res, res)
    normal_map_pil = Image.fromarray(
        normal_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))
    anomal_np = ((1 - normal_map) * 255).cpu().detach().numpy().astype(np.uint8)
    anomaly_map_pil = Image.fromarray(anomal_np).resize((org_h, org_w))

    return cls_map_pil, normal_map_pil, anomaly_map_pil



def main(args):

    print(f'\n step 1. accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with,
                              project_dir='log')

    print(f'\n step 2. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype,
                                                   accelerator)

    position_embedder = None
    if args.use_position_embedder:
        if args.all_positional_embedder :
            position_embedder = AllPositionalEmbedding()

    global_conv_net = None
    if args.use_global_conv :
        from model.overlapping_conv import AllGCN
        global_conv_net = AllGCN()


    print(f'\n step 2. accelerator and device')
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 3. inference')
    models = os.listdir(args.network_folder)
    network = LoRANetwork(text_encoder=text_encoder,
                          unet=unet,
                          lora_dim=args.network_dim,
                          alpha=args.network_alpha,
                          module_class=LoRAInfModule)
    network.apply_to(text_encoder, unet, True, True)
    raw_state_dict = network.state_dict()
    raw_state_dict_orig = raw_state_dict.copy()

    normal_activator = NormalActivator(None, None, args.use_focal_loss)

    for model in models:

        network_model_dir = os.path.join(args.network_folder, model)
        lora_name, ext = os.path.splitext(model)
        lora_epoch = int(lora_name.split('-')[-1])

        # [1] loead pe
        if args.use_position_embedder:
            parent = os.path.split(args.network_folder)[0]
            pe_base_dir = os.path.join(parent, f'position_embedder')
            pretrained_pe_dir = os.path.join(pe_base_dir, f'position_embedder_{lora_epoch}.safetensors')
            position_embedder_state_dict = load_file(pretrained_pe_dir)
            position_embedder.load_state_dict(position_embedder_state_dict)
            position_embedder.to(accelerator.device, dtype=weight_dtype)

        if args.use_global_conv:
            global_net_pretrained_dir = os.path.join(os.path.join(parent, f'global_convolution_network'), f'global_convolution_net_{lora_epoch}.safetensors')
            global_conv_net.load_state_dict(load_file(global_net_pretrained_dir))
            global_conv_net.to(accelerator.device, dtype=weight_dtype)

        # [2] load network
        anomal_detecting_state_dict = load_file(network_model_dir)
        for k in anomal_detecting_state_dict.keys():
            raw_state_dict[k] = anomal_detecting_state_dict[k]
        network.load_state_dict(raw_state_dict)
        network.to(accelerator.device, dtype=weight_dtype)

        # [3] files
        parent, _ = os.path.split(args.network_folder)
        if args.do_train_check :
            recon_base_folder = os.path.join(parent, 'reconstruction_with_train_data')
        else :
            recon_base_folder = os.path.join(parent, 'reconstruction_with_test_data')
        os.makedirs(recon_base_folder, exist_ok=True)
        lora_base_folder = os.path.join(recon_base_folder, f'lora_epoch_{lora_epoch}')
        os.makedirs(lora_base_folder, exist_ok=True)

        # [4] collector
        controller = AttentionStore()
        register_attention_control(unet, controller)

        for thred in args.threds :
            thred_folder = os.path.join(lora_base_folder, f'thred_{thred}')
            os.makedirs(thred_folder, exist_ok=True)

            check_base_folder = os.path.join(thred_folder, f'my_check')
            os.makedirs(check_base_folder, exist_ok=True)
            answer_base_folder = os.path.join(thred_folder, f'scoring/{args.obj_name}/test')
            os.makedirs(answer_base_folder, exist_ok=True)

            # [1] test path
            test_img_folder = args.data_path
            if args.do_train_check :
                parent, test = os.path.split(args.data_path)
                test_img_folder = os.path.join(parent, 'train')
            parent, test_folder = os.path.split(test_img_folder)

            anomal_folders = os.listdir(test_img_folder)
            for anomal_folder in anomal_folders:
                answer_anomal_folder = os.path.join(answer_base_folder, anomal_folder)
                os.makedirs(answer_anomal_folder, exist_ok=True)
                save_base_folder = os.path.join(check_base_folder, anomal_folder)
                os.makedirs(save_base_folder, exist_ok=True)


                anomal_folder_dir = os.path.join(test_img_folder, anomal_folder)
                rgb_folder = os.path.join(anomal_folder_dir, 'rgb')
                gt_folder = os.path.join(anomal_folder_dir, 'gt')
                if args.object_crop:
                    object_mask_folder = os.path.join(anomal_folder_dir, 'object_mask')
                rgb_imgs = os.listdir(rgb_folder)

                for rgb_img in rgb_imgs:

                    name, ext = os.path.splitext(rgb_img)
                    rgb_img_dir = os.path.join(rgb_folder, rgb_img)
                    pil_img = Image.open(rgb_img_dir).convert('RGB')
                    org_h, org_w = pil_img.size

                    # [1] read object mask
                    input_img = pil_img
                    trg_h, trg_w = input_img.size
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            img = np.array(input_img.resize((512, 512))) # [512,512,3]
                            #latent = image2latent(img, vae, weight_dtype)
                            image = torch.from_numpy(img).float() / 127.5 - 1
                            image = image.permute(2, 0, 1).unsqueeze(0).to(vae.device, weight_dtype) # [1,3,512,512]
                            with torch.no_grad():
                                if args.patch_positional_self_embedder:
                                    latent = position_embedder.patch_embed(image.to(dtype=weight_dtype))
                                else:
                                    latent = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample() * 0.18215
                            cls_map_pil, normal_map_pil, anomaly_map_pil = inference(latent,
                                                                                     tokenizer, text_encoder, unet,
                                                                                     controller, normal_activator,
                                                                                     position_embedder,
                                                                                     args,
                                                                                     trg_h, trg_w,
                                                                                     thred, global_conv_net)
                            cls_map_pil.save(os.path.join(save_base_folder, f'{name}_cls.png'))
                            normal_map_pil.save(os.path.join(save_base_folder, f'{name}_normal.png'))
                            anomaly_map_pil.save( os.path.join(save_base_folder, f'{name}_anomal.png'))
                            anomaly_map_pil.save(os.path.join(answer_anomal_folder, f'{name}.tiff'))
                    controller.reset()
                    normal_activator.reset()
                    # [2] gt save
                    if 'good' not in anomal_folder:
                        gt_img_save_dir = os.path.join(save_base_folder, f'{name}_gt.png')
                        Image.open(os.path.join(gt_folder, rgb_img)).resize((org_h, org_w)).save(gt_img_save_dir)
                    # [3] original save
                    Image.open(rgb_img_dir).convert('RGB').save(os.path.join(save_base_folder, rgb_img))
            # ---------------------------------------------------------------------------------------------------------
            # [2] train path
            """
            if not args.object_crop:
                train_img_folder = os.path.join(parent, 'train')

                save_base_folder = os.path.join(check_base_folder, f'train_good')
                os.makedirs(save_base_folder, exist_ok=True)

                normal_folder_dir = os.path.join(train_img_folder, 'good')
                rgb_folder = os.path.join(normal_folder_dir, 'rgb')

                if args.object_crop:
                    object_mask_folder = os.path.join(anomal_folder_dir, 'object_mask')

                rgb_imgs = os.listdir(rgb_folder)
                for rgb_img in rgb_imgs:

                    name, ext = os.path.splitext(rgb_img)
                    rgb_img_dir = os.path.join(rgb_folder, rgb_img)
                    pil_img = Image.open(rgb_img_dir).convert('RGB')
                    org_h, org_w = pil_img.size

                    # [1] read object mask
                    if args.object_crop :
                        object_mask_pil = Image.open(os.path.join(object_mask_folder, rgb_img)).convert('L')
                        h_start, h_end, w_start, w_end = generate_object_point(object_mask_pil)
                        input_img = pil_img.crop((w_start, h_start, w_end, h_end))
                    else :
                        input_img = pil_img
                    trg_h, trg_w = input_img.size
                    if accelerator.is_main_process:

                        with torch.no_grad():
                            img = np.array(input_img.resize((512, 512)))
                            vae_latent = image2latent(img, vae, weight_dtype)
                            cls_map_pil, normal_map_pil, anomaly_map_pil = inference(vae_latent,
                                                                                     tokenizer, text_encoder, unet,
                                                                                     controller, normal_activator,
                                                                                     position_embedder,
                                                                                     args,
                                                                                     trg_h, trg_w,
                                                                                     thred)
                            cls_map_pil.save(os.path.join(save_base_folder, f'{name}_cls.png'))
                            normal_map_pil.save(os.path.join(save_base_folder, f'{name}_normal.png'))
                            anomaly_map_pil.save( os.path.join(save_base_folder, f'{name}_anomal.png'))
                            anomaly_map_pil.save(os.path.join(answer_anomal_folder, f'{name}.tiff'))
                        Image.open(rgb_img_dir).convert('RGB').save(os.path.join(save_base_folder, rgb_img))
            """
        print(f'Model To Original')
        for k in raw_state_dict_orig.keys():
            raw_state_dict[k] = raw_state_dict_orig[k]
        network.load_state_dict(raw_state_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int, default=64)
    parser.add_argument('--network_alpha', type=float, default=4)
    parser.add_argument('--network_folder', type=str)
    parser.add_argument("--lowram", action="store_true", )
    # step 4. dataset and dataloader
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument('--data_path', type=str,
                        default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bagel')
    # step 6
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--prompt", type=str, default="bagel", )
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--latent_res", type=int, default=64)
    parser.add_argument("--single_layer", action='store_true')
    parser.add_argument("--use_noise_scheduler", action='store_true')
    parser.add_argument('--min_timestep', type=int, default=0)
    parser.add_argument('--max_timestep', type=int, default=500)
    # step 8. test
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--threds", type=arg_as_list,default=[0.85,])
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--do_normalized_score", action='store_true')
    parser.add_argument("--d_dim", default=320, type=int)
    parser.add_argument("--thred", default=0.5, type=float)
    parser.add_argument("--image_classification_layer", type=str)
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--gen_batchwise_attn", action='store_true')
    parser.add_argument("--object_crop", action='store_true')
    parser.add_argument("--use_multi_position_embedder", action='store_true')
    parser.add_argument("--all_positional_embedder", action='store_true')
    parser.add_argument("--all_positional_self_cross_embedder", action='store_true')
    parser.add_argument("--patch_positional_self_embedder", action='store_true')
    parser.add_argument("--all_self_cross_positional_embedder", action='store_true')
    parser.add_argument("--use_global_conv", action='store_true')
    parser.add_argument("--do_train_check", action='store_true')
    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    passing_normalize_argument(args)
    main(args)