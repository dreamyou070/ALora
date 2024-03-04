# !/bin/bash

port_number=50000
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
layer_name='layer_3'

anomal_source_path="../../../MyData/anomal_source"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../data_check.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/data_check_20240304_0.3_0.7_partial_anomal_random_rot" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --do_object_detection \
 --anomal_only_on_object \
 --anomal_source_path "${anomal_source_path}" \
 --anomal_min_perlin_scale 0 \
 --anomal_max_perlin_scale 6 \
 --anomal_min_beta_scale 0.3 \
 --anomal_max_beta_scale 0.7 \
 --back_trg_beta 0 \
 --do_background_masked_sample \
 --do_anomal_sample