import torch
from torch import nn



def resize_query_features(query):
    pix_num, dim = query.shape
    res = int(pix_num ** 0.5)
    query_map = query.view(res, res, dim).permute(2, 0, 1).contiguous().unsqueeze(0)  # 1, channel, res, res

    resized_query_map = nn.functional.interpolate(query_map, size=(64, 64), mode='bilinear')  # 1, channel, 64,  64
    print(f'resized_query_map : {resized_query_map.shape}')
    resized_query = resized_query_map.permute(0, 2, 3, 1).contiguous().squeeze()  # 64, 64, channel
    resized_query = resized_query.view(pix_num, dim)
    return resized_query

query = torch.randn(64*64, 320)
a = resize_query_features(query)
print(query)