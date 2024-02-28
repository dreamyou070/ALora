import torch
attn_score = torch.randn(64,2)
cls_score, trigger_score = attn_score.chunk(2, dim=-1)
cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()      # head, pix_num

cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
print(f'cls_score : {cls_score.shape}')