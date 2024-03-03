# !/bin/bash

port_number=50055
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="test_20240303_with_gt_answer_check"

anomal_source_path="../../../MyData/anomal_source"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_config \
 --main_process_port $port_number ../train_answer_test.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" --do_object_detection --anomal_only_on_object \
 --anomal_source_path "${anomal_source_path}" \
 --anomal_min_perlin_scale 0 \
 --anomal_max_perlin_scale 6 \
 --anomal_min_beta_scale 0.5 \
 --anomal_max_beta_scale 0.8 \
 --back_trg_beta 0 \
 --do_background_masked_sample --do_object_detection --do_anomal_sample --answer_test \
 --use_position_embedder \
 --do_map_loss \
 --trg_layer_list "['up_blocks_1_attentions_1_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_1_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_1_transformer_blocks_0_attn2']" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1