# !/bin/bash

port_number=50003
bench_mark="MVTec"
obj_name='transistor'
caption='transistor'
sub_folder="train_unet_background_sample"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction_clip_condition.py \
 --output_dir "../../result/${obj_name}/unet_train/${sub_folder}/"
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}" --prompt "${caption}"