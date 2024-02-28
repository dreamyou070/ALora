# !/bin/bash

port_number=50002
bench_mark="MVTec"
obj_name='transistor'
caption='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="test_global_3_back_normal"

position_embedding_layer="down_blocks_0_attentions_0_transformer_blocks_0_attn1"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 --network_folder "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/models" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}" --prompt "${caption}" \
 --latent_res 64 --trg_layer_list "['mid_block_attentions_0_transformer_blocks_0_attn2']" \
 --d_dim 320 --use_position_embedder --position_embedding_layer ${position_embedding_layer} \
 --threds [0.5]