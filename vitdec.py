from LG_ES_Transformer.AnoViT.timm import create_model
from LG_ES_Transformer.AnoViT.model import Decoder_r
from torch import nn
model = create_model('vit_base_patch16_384',
                     pretrained=True)
decmodel = nn.Sequential(model, Decoder_r(args))
decmodel.to(device)