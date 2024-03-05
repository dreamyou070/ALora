# !/bin/bash

port_number=50005
bench_mark="Tuft"
obj_name='teeth_crop_onlynomal'
trigger_word='teeth'
layer_name='vae_train'
#sub_folder="mid_up_16_32_64"
file_name="train_vae_reconstruction_nomal_data"

anomal_source_path="../../../MyData/anomal_source"
# --anomal_source_path "${anomal_source_path}" \

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_config \
 --main_process_port $port_number ../train_vae.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${layer_name}/${file_name}" \
 --start_epoch 0 --max_train_epochs 150 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" --anomal_only_on_object \
 --anomal_min_perlin_scale 0 \
 --anomal_max_perlin_scale 6 \
 --anomal_min_beta_scale 0.5 \
 --anomal_max_beta_scale 0.8 \
 --back_min_perlin_scale 0 \
 --back_max_perlin_scale 6 \
 --back_min_beta_scale 0.6 \
 --back_max_beta_scale 0.9 \
 --back_trg_beta 0 \
 --use_pretrained_vae --answer_test
