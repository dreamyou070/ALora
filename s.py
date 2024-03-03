import torch
min_timestep = 0
max_timestep = 100
b_size = 1
timesteps = torch.randint(min_timestep, max_timestep, (b_size,), )
print(timesteps)
a = torch.tensor([15])
print(a)