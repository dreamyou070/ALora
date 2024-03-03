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
from utils.image_utils import load_image, image2latent
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file
from attention_store.normal_activator import NormalActivator
from attention_store.normal_activator import passing_normalize_argument
from torch import nn
from transformers import CLIPVisionModel, AutoProcessor
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline
from model.unet import UNet2DConditionModel
import json

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

scheduler = DDIMScheduler(num_train_timesteps=1000,
                          beta_start=0.00085,
                          beta_end=0.012,
                          beta_schedule="scaled_linear")

def inference(latent, tokenizer, text_encoder, unet, controller, normal_activator, position_embedder,
              args, org_h, org_w, thred, query_transformer):
    # [1] text
    input_ids, attention_mask = get_input_ids(tokenizer, args.prompt)
    encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device))["last_hidden_state"]
    # [2] unet
    unet(latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=position_embedder)
    query_dict, key_dict = controller.query_dict, controller.key_dict
    controller.reset()
    for layer in args.trg_layer_list:
        if 'mid' in layer:
            query = query_dict[layer][0].squeeze()  # 8, 64, 160
            key = key_dict[layer][0].squeeze()  # 8, 77, 160
    global_query = query_transformer(query)  # g_query = 8, 64, 160 -> 8, 64*64, 280 (feature generating with only global context)
    attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                                     query, key.transpose(-1, -2), beta=0, )  # [head, 64, 77]
    global_attn = attention_scores.softmax(dim=-1)[:, :, 1:65]  # [head, 64, 64]

    head = global_attn.shape[0]
    attn = [torch.diagonal(global_attn[i]) for i in range(head)]
    global_anomal_map = torch.stack(attn, dim=0).mean(dim=0).view(8, 8).unsqueeze(0).unsqueeze(0)
    global_anomal_map = nn.functional.interpolate(global_anomal_map, size=(64, 64), mode='bilinear').squeeze() # (64,64)
    res = 64
    normal_map = torch.where(global_anomal_map > thred, 1, global_anomal_map).squeeze()
    #normal_map = normal_map.unsqueeze(0).view(res, res)
    normal_map_pil = Image.fromarray(normal_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))
    anomal_np = ((1 - normal_map) * 255).cpu().detach().numpy().astype(np.uint8)
    anomaly_map_pil = Image.fromarray(anomal_np).resize((org_h, org_w))
    return normal_map_pil, anomaly_map_pil

def main(args):

    print(f'\n step 1. accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with,
                              project_dir='log')

    print(f'\n step 2. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    # [1] vae
    vae_base_dir = r'/home/dreamyou070/AnomalLora_OriginCode/result/MVTec/transistor/vae_train/train_vae_20240302_6_distill_recon'
    vae_config_dir = os.path.join(vae_base_dir, 'vae_config.json')
    with open(vae_config_dir, 'r') as f:
        vae_config_dict = json.load(f)
    vae = AutoencoderKL.from_config(pretrained_model_name_or_path=vae_config_dict)
    vae.load_state_dict(load_file(os.path.join(vae_base_dir, f'vae_models/vae_91.safetensors')))
    # [3] unet
    unet_config_dir = os.path.join(r'/home/dreamyou070/AnomalLora_OriginCode/result/MVTec/transistor/unet_train/train_unet_background_sample',
                                   'unet_config.json')
    with open(unet_config_dir, 'r') as f:
        unet_config_dict = json.load(f)
    unet = UNet2DConditionModel(**unet_config_dict)

    print(f'\n step 2. accelerator and device')
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 3. inference')
    s_file = os.path.join(r'/home/dreamyou070/AnomalLora_OriginCode/result/MVTec/transistor/unet_train/train_unet_background_sample',
                                   'unet_scale_factor.txt')
    with open(s_file, 'r') as f :
        content = f.readlines()
    scaling_factor = float(content[0])
    unet_models = os.listdir(os.path.join(args.output_dir, 'unet_models'))
    clip_model = model.to(accelerator.device)

    for unet_model in unet_models:

        unet_name, ext = os.path.splitext(unet_model)
        unet_epoch = int(unet_name.split('_')[-1])

        unet_model_dir = os.path.join(args.output_dir, f'unet_models/{unet_model}')
        unet.load_state_dict(load_file(unet_model_dir))
        unet.requires_grad_(False)
        unet.to(accelerator.device, dtype=weight_dtype)

        # [3] files
        recon_base_folder = os.path.join(args.output_dir, 'reconstruction')
        os.makedirs(recon_base_folder, exist_ok=True)

        unet_base_folder = os.path.join(recon_base_folder, f'unet_epoch_{unet_epoch}')
        os.makedirs(unet_base_folder, exist_ok=True)
        check_base_folder = os.path.join(unet_base_folder, f'my_check')
        os.makedirs(check_base_folder, exist_ok=True)
        answer_base_folder = os.path.join(unet_base_folder, f'scoring/{args.obj_name}/test')
        os.makedirs(answer_base_folder, exist_ok=True)

        # [1] test path
        test_img_folder = args.data_path
        anomal_folders = os.listdir(test_img_folder)

        for anomal_folder in anomal_folders:
            answer_anomal_folder = os.path.join(answer_base_folder, anomal_folder)
            os.makedirs(answer_anomal_folder, exist_ok=True)
            save_base_folder = os.path.join(check_base_folder, anomal_folder)
            os.makedirs(save_base_folder, exist_ok=True)


            anomal_folder_dir = os.path.join(test_img_folder, anomal_folder)
            rgb_folder = os.path.join(anomal_folder_dir, 'rgb')
            gt_folder = os.path.join(anomal_folder_dir, 'gt')
            rgb_imgs = os.listdir(rgb_folder)

            for rgb_img in rgb_imgs:
                # [1] image condition
                name, ext = os.path.splitext(rgb_img)
                rgb_img_dir = os.path.join(rgb_folder, rgb_img)
                pil_img = Image.open(rgb_img_dir).convert('RGB')
                np_img = np.array(pil_img)
                with torch.no_grad() :
                    inputs = processor(images=np_img, return_tensors="pt").to(accelerator.device)
                    img_condition = clip_model(**inputs).last_hidden_state  # 1, 50, 768
                    # [2] sampling
                    latent = torch.randn(1,4,64,64).to(accelerator.device)
                    num_inference_steps = 50
                    scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
                    timesteps = scheduler.timesteps
                    for i, t in enumerate(timesteps):
                        noise_pred = unet(latent, t, encoder_hidden_states=img_condition).sample
                        latent = scheduler.step(noise_pred, t, latent, return_dict=False)[0]
                    # latent to image
                    image = vae.decode(latent / scaling_factor, return_dict=False)[0]
                    print(f'vae out. image : {type(image)}')
                    np_image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                    np_image = (np_image * 255).round().astype("uint8")
                    pil_image = Image.fromarray(np_image[:, :, :3])
                    #pil_image.save()




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
    parser.add_argument("--all_positional_embedder", action='store_true')
    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    passing_normalize_argument(args)
    main(args)