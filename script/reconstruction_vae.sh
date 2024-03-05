# !/bin/bash

port_number=50003
bench_mark="Tuft"
obj_name='teeth_crop'
caption='teeth'
file_name="train_vae_reconstruction_nomal_data"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction_vae.py \
 --output_dir "../../result/${bench_mark}/${obj_name}/vae_train/${file_name}" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}" --prompt "${caption}"