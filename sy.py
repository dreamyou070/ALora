from torch.nn import L1Loss
import torch
l1_loss = L1Loss()
z = torch.randn(1,4,64,64)
z_t= torch.randn(1,4,64,64)
latent_matching_loss = l1_loss(z, z_t)
print(f'latent_matching_loss : {latent_matching_loss.shape}')