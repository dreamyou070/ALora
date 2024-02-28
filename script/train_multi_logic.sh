# !/bin/bash

port_number=50000
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="rotating_2"

anomal_source_path="../../../MyData/anomal_source"
network_weights="../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/models/epoch-000015.safetensors"
accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../train_multi_logic.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --network_weights ${network_weights} \
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
 --anomal_p 0.04 \
 --do_anomal_sample --do_background_masked_sample --do_object_detection \
 --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' --d_dim 320 --latent_res 64 \
 --do_attn_loss --do_map_loss \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2',
                    'down_blocks_1_attentions_1_transformer_blocks_0_attn2',
                    'down_blocks_2_attentions_1_transformer_blocks_0_attn2',]" \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1