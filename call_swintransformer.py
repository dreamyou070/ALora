from model.swin_transformer import SwinTransformer
import os
import torch
from transformers import SwinConfig, SwinModel

# [1] model config
swin_base = '/home/dreamyou070/pretrained_model/swin_large_patch4_window12_384'
swin_config_dir = os.path.join(swin_base, 'swin_large_patch4_window12_384_22kto1k_finetune.yaml')
swin_model = SwinTransformer.from_pretrained(swin_config_dir)

# [2] load pretrained model
swin_state_dict_dir = os.path.join(swin_base, 'swin_large_patch4_window12_384_22k.pth')
swin_model.load_state_dict(torch.load(swin_state_dict_dir))

