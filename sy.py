import torch
import einops

attn_score = torch.randn(8, 16,16)
averizer = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
head, pixel_num, pixe_num = attn_score.shape
res = int(pixel_num ** 0.5)
context_score = torch.zeros(head, pixel_num)
for pixel_idx in range(pixel_num):
    pixelwise_attnmap = attn_score[:, :,pixel_idx].reshape(-1, res, res)     # head, 4,4
    pixelwise_attnmap = averizer(pixelwise_attnmap.unsqueeze(1)).squeeze(1)  # head, 4,4
    pixelwise_attnmap = pixelwise_attnmap.view(-1, res*res)                 # head, 16
    score = pixelwise_attnmap[:, pixel_idx]
    context_score[:, pixel_idx] = score
context_score = context_score.view(head, res, res).unsqueeze(0) # 1, head, res, rs
resized_context_score = torch.nn.functional.interpolate(input = context_score,
                                                        size=(64,64),
                                                        mode='bilinear').squeeze() # head, 64, 64
print(resized_context_score.shape)

resized_self_attn_scores = [torch.randn(8,64,64),torch.randn(8,64,64)]
concat_attn_score = torch.cat(resized_self_attn_scores, dim=0)     # head*num, 64, 64
concat_attn_score = einops.rearrange(concat_attn_score, 'h p c -> h (p c)') # head*num, 64*64
print(concat_attn_score.shape)
