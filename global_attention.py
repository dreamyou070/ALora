from transformers import AutoImageProcessor, ViTModel
import torch

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k",
                                                     cache_dir="/home/dreamyou070/pretrained_model/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k",
                                 cache_dir="/home/dreamyou070/pretrained_model/vit-base-patch16-224-in21k")