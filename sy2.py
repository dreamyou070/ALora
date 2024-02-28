import torch
from torch import nn
from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

config_encoder = ViTConfig()
config_decoder = BertConfig()

config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("microsoft/swin-base-patch4-window7-224-in22k",
                                                                  "google-bert/bert-base-uncased")
model.save_pretrained("/home/dreamyou070/pretrained_model/vit-bert")