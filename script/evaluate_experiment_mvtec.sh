#! /bin/bash

bench_mark="MVTec"
obj_name='transistor'
caption='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="train_local_scaled_query"

dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
base_dir="../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/reconstruction"

output_dir="metrics"

python ../evaluation/evaluation_code_MVTec/evaluate_experiment_2.py \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --base_dir "${base_dir}" \
     --dataset_base_dir "${dataset_dir}" \
     --evaluated_objects "${obj_name}" \
     --pro_integration_limit 0.3