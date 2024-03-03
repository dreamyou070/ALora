# !/bin/bash

port_number=50003
bench_mark="MVTec"
obj_name='transistor'
caption='transistor'
file_name="train_vae_20240302_6_distill_recon"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruc_vae.py \
 --output_dir "../../result/${bench_mark}/${obj_name}/vae_train/${file_name}" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}"