# !/bin/bash

port_number=54444
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
layer_name='layer_3'
sub_folder="down_16_32_64"
file_name="test_20240303_down_selfattn_partial_back_random_rot"

anomal_source_path="../../../MyData/anomal_source"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_config \
 --main_process_port $port_number ../train_self.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 100 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" --do_object_detection --anomal_only_on_object \
 --anomal_source_path "${anomal_source_path}" \
 --anomal_min_perlin_scale 0 \
 --anomal_max_perlin_scale 6 \
 --anomal_min_beta_scale 0.3 \
 --anomal_max_beta_scale 0.7 \
 --back_trg_beta 0 \
 --do_anomal_sample --do_background_masked_sample --do_random_rot_sample \
 --use_position_embedder \
 --do_map_loss \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn1',
                    'down_blocks_1_attentions_1_transformer_blocks_0_attn1',
                    'down_blocks_2_attentions_1_transformer_blocks_0_attn1',]" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1