# !/bin/bash

port_number=50333
bench_mark="Tuft"
obj_name='teeth_crop'
caption='teeth'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="only_anomal_pe_cropped_image"

# position_embedding_layer="down_blocks_0_attentions_0_transformer_blocks_0_attn1"
# --d_dim 320 --use_position_embedder --position_embedding_layer ${position_embedding_layer} \
# --use_position_embedder --all_positional_embedder \

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction_local.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 --network_folder "../../result/${bench_mark}/${layer_name}/${sub_folder}/${file_name}/models" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --do_train_check \
 --obj_name "${obj_name}" --prompt "${caption}" \
 --latent_res 64 \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --threds [0.5] \
 --use_position_embedder --all_positional_embedder