from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
from PIL import Image

vit_dir = r'/home/dreamyou070/.cache/huggingface/hub/models--google--vit-base-patch16-224/snapshots/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3'
processor = ViTImageProcessor.from_pretrained(vit_dir)
#model = ViTForImageClassification.from_pretrained(vit_dir)

img_dir = r'/home/dreamyou070/MyData/anomal_source/dtd_images/banded/banded_0002.jpg'
image = Image.open(img_dir).convert('RGB')
inputs = processor(images=image, return_tensors="pt").data['pixel_values'] # batch, channel(=3), H(=224), W(=224)
print(f'input pixel values : {inputs.shape}')

vitmodel = ViTModel.from_pretrained(vit_dir)
output = vitmodel(inputs).last_hidden_state
print(f'output : {output.shape}')