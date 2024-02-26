from model.lora import create_network
from model.pe import PositionalEmbedding
from model.diffusion_model import load_target_model
import os
from safetensors.torch import load_file
from model.unet import TimestepEmbedding


def call_model_package(args, weight_dtype, accelerator):

    # [1] diffusion
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    vae.to(dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)
    vae.eval()
    # [2] lora network
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value
    network = create_network(1.0, args.network_dim, args.network_alpha,
                             vae, text_encoder, unet, neuron_dropout=args.network_dropout, **net_kwargs, )
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        print(f'Loaded weights from {args.network_weights}: {info}')
    network.to(weight_dtype)

    # [3] PE
    position_embedder = PositionalEmbedding(max_len=args.latent_res * args.latent_res, d_model=args.d_dim)
    if args.network_weights is not None:
        models_folder,  lora_file = os.path.split(args.network_weights)
        base_folder = os.path.split(models_folder)[0]
        lora_name, _ = os.path.splitext(lora_file)
        lora_epoch = int(lora_name.split("-")[-1])
        pe_name = f"position_embedder_{lora_epoch}.safetensors"
        position_embedder_path = os.path.join(base_folder, f"position_embedder/{pe_name}")
        position_embedder_state_dict = load_file(position_embedder_path)
        position_embedder.load_state_dict(position_embedder_state_dict)
        print(f'Position Embedding Loading Weights from {position_embedder_path}')
    position_embedder.to(weight_dtype)

    # [4] text time embedding
    text_time_embedding = None
    if args.use_text_time_embedding:
        text_time_embedding = TimestepEmbedding(320, 768)
        text_time_embedding.to(weight_dtype)


    return text_encoder, vae, unet, network, position_embedder, text_time_embedding

