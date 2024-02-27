import torch
import torch.nn as nn
class SelfEmbedding(nn.Module):

    def __init__(self, ):
        super(SelfEmbedding, self).__init__()

        self.res_16_self_k = self.generate_self_emb(res=16)  # name = parameter argument name
        self.res_16_self_v = self.generate_self_emb(res=16)
        self.res_32_self_k = self.generate_self_emb(res=32)
        self.res_32_self_v = self.generate_self_emb(res=32)
        self.res_64_self_k = self.generate_self_emb(res=64)
        self.res_64_self_v = self.generate_self_emb(res=64)

    def generate_self_emb(self, res):
        dim = int((64 * 320) / res)
        emb = torch.nn.Parameter(torch.randn(res, dim),
                                 requires_grad=True)
        return emb

    def forward(self, layer_name, x):

        if 'to_k' in layer_name:
            if 'up_blocks_3_attentions_2' in layer_name:
                return self.res_64_self_k
            elif 'up_blocks_2_attentions_2' in layer_name:
                return self.res_32_self_k
            elif 'up_blocks_1_attentions_2' in layer_name:
                return self.res_16_self_k

        elif 'to_v' in layer_name:
            if 'up_blocks_3_attentions_2' in layer_name:
                return self.res_64_self_v
            elif 'up_blocks_2_attentions_2' in layer_name:
                return self.res_32_self_v
            elif 'up_blocks_1_attentions_2' in layer_name:
                return self.res_16_self_v
        else :
            return x