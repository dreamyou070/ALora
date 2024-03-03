from PIL import Image
import requests
from transformers import CLIPVisionModel, AutoProcessor

device = 'cuda'
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

img_dir = '/home/dreamyou070/MyData/anomaly_detection/MVTec/transistor/train/good/rgb/000.png'
image = Image.open(img_dir).convert('RGB')
inputs = processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled CLS states
print(f'clip last hidden state : {last_hidden_state.shape}')
print(f'clip pooled output : {pooled_output.shape}')