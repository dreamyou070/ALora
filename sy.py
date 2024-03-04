import torch
a = torch.randn([3,4])
b = torch.randn([3,4,5])
b = b.to(dtype=a.dtype)
print(b)