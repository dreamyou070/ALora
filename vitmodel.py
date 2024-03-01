from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
from PIL import Image
import torch.nn as nn
import torch
import collections.abc

print(latent_diff)

#latent_diff = latent_diff / latent_diff.max() # only anomal = 1
#print(latent_diff) # [64*64]

"""
def save_tensors(module: nn.Module, features, name: str):
     Process and save activations in the module. 
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def main() :

    print(f' step 1. call model')
    vit_dir = r'/home/dreamyou070/.cache/huggingface/hub/models--google--vit-base-patch16-224/snapshots/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3'
    processor = ViTImageProcessor.from_pretrained(vit_dir)
    vitmodel = ViTModel.from_pretrained(vit_dir)
    vit_embedder = vitmodel.embeddings
    vit_encoder = vitmodel.encoder
    encoder_layers = vit_encoder.layer

    print(f' step 2. register hook')
    hooking_layers = []
    for layer in encoder_layers :
        layer.register_forward_hook(save_out_hook)
        hooking_layers.append(layer)

    print(f' step 3. call image')
    img_dir = r'/home/dreamyou070/MyData/anomal_source/dtd_images/banded/banded_0002.jpg'
    image = Image.open(img_dir).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").data['pixel_values'] # batch, channel(=3), H(=224), W(=224)
    output = vitmodel(inputs).last_hidden_state # [1batch, 1 + 196 (=14*14), 768 dim]

    extracted_features = torch.randn(8,64*64, 280)


    print(f' step 4. extract feature')
    activations = []
    for h_layer in hooking_layers :
        hooked_output = h_layer.activations[0]
        print(f'hooked output : {hooked_output.shape}')
        activations.append(hooked_output)
        h_layer.activations = None

if __name__ == '__main__' :
    main()
"""