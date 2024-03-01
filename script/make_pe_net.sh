# !/bin/bash

# scratch vae / trained vae
# matching loss : only normal

port_number=50000
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="from_trained_pe_local_global_from_scratch_vae"

anomal_source_path="../../../MyData/anomal_source"
network_weights="../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/local_all_crossattn_pe/models/epoch-000009.safetensors" \


accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../make_pe_net.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --network_weights ${network_weights} \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" --anomal_only_on_object