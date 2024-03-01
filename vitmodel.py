from transformers import ViTImageProcessor, ViTForImageClassification

vit_dir = r'/home/dreamyou070/.cache/huggingface/hub/models--google--vit-base-patch16-224/snapshots/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3'
processor = ViTImageProcessor.from_pretrained(vit_dir)
model = ViTForImageClassification.from_pretrained(vit_dir)
