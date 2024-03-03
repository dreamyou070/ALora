# !/bin/bash

port_number=50005
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
layer_name='vae_train'
sub_folder="mid_up_16_32_64"
file_name="train_vae_20240302_6_distill_recon"

anomal_source_path="../../../MyData/anomal_source"
# --anomal_source_path "${anomal_source_path}" \

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../train_unet.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${file_name}" \
 --start_epoch 30 --max_train_epochs 100 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}"