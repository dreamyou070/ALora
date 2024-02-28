from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from transformers import ViTModel
#feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
#model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
vit_model = ViTModel.freom_pretrained('vit-base-patch16-384')
