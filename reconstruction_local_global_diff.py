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
from model.diffusion_model import load_target_model
from model.pe import PositionalEmbedding
from safetensors.torch import load_file
from attention_store.normal_activator import NormalActivator
from attention_store.normal_activator import passing_normalize_argument
from torch import nn
from model.pe import PositionalEmbedding, MultiPositionalEmbedding, AllPositionalEmbedding, Patch_MultiPositionalEmbedding, AllSelfCrossPositionalEmbedding
from model import call_model_package
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


def main(args):

    print(f'\n step 1. accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with,
                              project_dir='log')

    print(f'\n step 2. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    tokenizer = load_tokenizer(args)
    print(f' (2.1) local model')
    # [1] call local model
    l_text_encoder, l_vae, l_unet, l_network, l_position_embedder = call_model_package(args, weight_dtype, accelerator, True)
    l_vae.requires_grad_(False)
    l_vae.to(accelerator.device, dtype=weight_dtype)
    l_unet.requires_grad_(False)
    l_unet.to(accelerator.device, dtype=weight_dtype)
    l_text_encoder.requires_grad_(False)
    l_text_encoder.to(accelerator.device, dtype=weight_dtype)
    l_controller = AttentionStore()
    register_attention_control(l_unet, l_controller)

    print(f' (2.2) global model')
    g_text_encoder, g_vae, g_unet, g_network, g_position_embedder = call_model_package(args, weight_dtype, accelerator, False)
    g_vae.requires_grad_(False)
    g_vae.to(accelerator.device, dtype=weight_dtype)
    g_unet.requires_grad_(False)
    g_unet.to(accelerator.device, dtype=weight_dtype)
    g_text_encoder.requires_grad_(False)
    g_text_encoder.to(accelerator.device, dtype=weight_dtype)
    g_controller = AttentionStore()
    register_attention_control(g_unet, g_controller)

    print(f'\n step 3. call experiment network dirs')
    models = os.listdir(args.network_folder)
    g_network.apply_to(g_text_encoder, g_unet, True, True)
    raw_state_dict = g_network.state_dict()
    raw_state_dict_orig = raw_state_dict.copy()

    for model in models:
        # [3.1] global network
        network_model_dir = os.path.join(args.network_folder, model)
        lora_name, ext = os.path.splitext(model)
        lora_epoch = int(lora_name.split('-')[-1])

        # [3.2] global pe
        pe_base_dir = os.path.join(os.path.split(args.network_folder)[0], f'position_embedder')
        g_position_embedder.load_state_dict(load_file(os.path.join(pe_base_dir, f'position_embedder_{lora_epoch}.safetensors')))
        g_position_embedder.to(accelerator.device, dtype=weight_dtype)

        # [3.3] load network
        anomal_detecting_state_dict = load_file(network_model_dir)
        for k in anomal_detecting_state_dict.keys():
            raw_state_dict[k] = anomal_detecting_state_dict[k]
        g_network.load_state_dict(raw_state_dict)
        g_network.to(accelerator.device, dtype=weight_dtype)

        # [3.4] files
        parent, _ = os.path.split(args.network_folder)
        recon_base_folder = os.path.join(parent, 'reconstruction')
        os.makedirs(recon_base_folder, exist_ok=True)
        lora_base_folder = os.path.join(recon_base_folder, f'lora_epoch_{lora_epoch}')
        os.makedirs(lora_base_folder, exist_ok=True)

        for thred in args.threds :
            thred_folder = os.path.join(lora_base_folder, f'thred_{thred}')
            os.makedirs(thred_folder, exist_ok=True)
            check_base_folder = os.path.join(thred_folder, f'my_check')
            os.makedirs(check_base_folder, exist_ok=True)
            answer_base_folder = os.path.join(thred_folder, f'scoring/{args.obj_name}/test')
            os.makedirs(answer_base_folder, exist_ok=True)

            # [1] test path
            test_img_folder = args.data_path
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
                            img = np.array(input_img.resize((512, 512)))
                            # [5.1] local extracting feature
                            latent = image2latent(img, l_vae, weight_dtype)
                            # (2) text embedding
                            input_ids, attention_mask = get_input_ids(tokenizer, args.prompt)
                            encoder_hidden_states = l_text_encoder(input_ids.to(l_text_encoder.device))["last_hidden_state"]
                            # (3) extract local features
                            l_unet(latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=l_position_embedder, )
                            l_query_dict, l_key_dict = l_controller.query_dict, l_controller.key_dict
                            l_controller.reset()
                            l_query_list = []
                            for layer in args.trg_layer_list:
                                if 'mid' not in layer:
                                    l_query_list.append(
                                        resize_query_features(l_query_dict[layer][0].squeeze()))  # feature selecting
                            local_query = torch.cat(l_query_list, dim=-1)  # 8, 64*64, 280
                            # (4) extract global features
                            with torch.set_grad_enabled(True):
                                g_unet(latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,noise_type=g_position_embedder)
                            g_query_dict, g_key_dict = g_controller.query_dict, g_controller.key_dict
                            g_controller.reset()
                            g_query_list = []
                            for layer in args.trg_layer_list:
                                if 'mid' not in layer:
                                    g_query_list.append(resize_query_features(g_query_dict[layer][0].squeeze()))  # feature selecting
                            global_query = torch.cat(g_query_list, dim=-1)  # 8, 64*64, 280
                            # (5) feature comparing
                            latent_diff = abs(local_query.float() - global_query.float())
                            latent_diff = latent_diff.mean(dim=0).mean(dim=-1)
                            latent_diff = (latent_diff.max() + 0.0001) - latent_diff
                            anormality = 1 - (latent_diff / latent_diff.max()) # [4096]
                            anomal_map = torch.where(anormality > thred, 1, anormality).squeeze()
                            anomal_map = anomal_map.view(64, 64)
                            # normal_map = normal_map.unsqueeze(0).view(res, res)
                            anomal_map_pil = Image.fromarray(anomal_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))

                            nomal_np = ((1 - anomal_map) * 255).cpu().detach().numpy().astype(np.uint8)
                            nomal_map_pil = Image.fromarray(nomal_np).resize((org_h, org_w))
                            nomal_map_pil.save(os.path.join(save_base_folder, f'{name}_normal.png'))

                            anomal_map_pil.save( os.path.join(save_base_folder, f'{name}_anomal.png'))
                            anomal_map_pil.save(os.path.join(answer_anomal_folder, f'{name}.tiff'))
                    g_controller.reset()
                    l_controller.reset()
                    # [2] gt save
                    if 'good' not in anomal_folder:
                        gt_img_save_dir = os.path.join(save_base_folder, f'{name}_gt.png')
                        Image.open(os.path.join(gt_folder, rgb_img)).resize((org_h, org_w)).save(gt_img_save_dir)
                    # [3] original save
                    Image.open(rgb_img_dir).convert('RGB').save(os.path.join(save_base_folder, rgb_img))
        print(f'Model To Original')
        for k in raw_state_dict_orig.keys():
            raw_state_dict[k] = raw_state_dict_orig[k]
        g_network.load_state_dict(raw_state_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int, default=64)
    parser.add_argument('--network_alpha', type=float, default=4)
    parser.add_argument('--network_folder', type=str)
    parser.add_argument("--network_weights", type=str, default=None,)
    parser.add_argument("--network_dropout", type=float, default=None, )
    parser.add_argument("--network_args", type=str, default=None, nargs="*", )

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