import os
import torch
import random
from random import sample

original_latent = torch.randn(1,4,64,64)
original_latent_vector = original_latent.flatten(2)

idx = torch.tensor(sample(range(0, 64*64), 100)).sort()
print(idx)
#random_index = random.select(0,64*64, 100)
#print(random_index)
#index_select = torch.index_select(original_latent_vector, dim = -1)
#print(index_select)
#embedding_vectors = torch.index_select(original_latent_vector, 2, idx)