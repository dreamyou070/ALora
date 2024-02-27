from transformers import AutoImageProcessor, ViTModel
import torch

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k",)
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k",)
model.save_pretrained(r"/home/dreamyou070/pretrained_model")
image_processor.save_pretrained(r"/home/dreamyou070/pretrained_model")