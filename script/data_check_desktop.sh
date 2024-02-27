# !/bin/bash

port_number=50000
bench_mark="MVTec"
obj_name='screw'
trigger_word='screw'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="test_3"

anomal_source_path="../anomal_source"

python ../data_check_desktop.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/data_check" \
 --data_path "../${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" --anomal_only_on_object \
 --anomal_source_path "${anomal_source_path}" \
 --anomal_min_perlin_scale 0 \
 --anomal_max_perlin_scale 6 \
 --anomal_min_beta_scale 0.5 \
 --anomal_max_beta_scale 0.8 \
 --back_min_perlin_scale 0 \
 --back_max_perlin_scale 6 \
 --back_min_beta_scale 0.6 \
 --back_max_beta_scale 0.9 \
 --back_trg_beta 0 \
 --do_anomal_sample --do_background_masked_sample \
 --do_rot_augment --on_desktop --do_object_detection