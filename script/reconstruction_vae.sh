# !/bin/bash

port_number=50003
bench_mark="MVTec"
obj_name='transistor'
caption='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="train_from_vae_nomal_sample"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction_vae.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 --network_folder "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/models" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}" --prompt "${caption}"