# !/bin/bash

port_number=50003
bench_mark="MVTec"
obj_name='transistor'
caption='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="from_trained_pe_local_global_from_pretrained_vae"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction_local_global_diff.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 --network_folder "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/models" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}" --prompt "${caption}" \
 --latent_res 64 \
 --network_weights "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/local_all_crossattn_pe/models/epoch-000009.safetensors" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --local_pretrained_network_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/local_no_pe/models/epoch-000016.safetensors" \
 --local_position_embedder_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/local_all_crossattn_pe/position_embedder/position_embedder_9.safetensors" \
 --all_positional_embedder \
 --threds [0.5]
