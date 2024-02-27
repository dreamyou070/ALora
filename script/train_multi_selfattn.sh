# !/bin/bash

port_number=50000
bench_mark="MVTec"
obj_name='screw'
trigger_word='screw'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="test_4_selfattn"

anomal_source_path="../../../MyData/anomal_source"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_multi_selfattn.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
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
 --do_rot_augment \
 --anomal_p 0.02 \
 --do_anomal_sample --do_background_masked_sample --do_object_detection \
 --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' --d_dim 320 --latent_res 64 \
 --do_attn_loss --do_map_loss \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn1',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn1'
                    'up_blocks_1_attentions_2_transformer_blocks_0_attn1']" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1