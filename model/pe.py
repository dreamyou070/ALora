import torch.nn as nn
import torch
import einops
class PositionalEmbedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)
        pe = self.positional_encodings.expand(b_size, -1, -1)
        x = x + pe
        if start_dim == 4:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)
        return x

class MultiPositionalEmbedding(nn.Module):

    def __init__(self,
                 max_lens: int = [64*64,32*32,16*16],
                 d_models: int = [320,640,1280]):
        super().__init__()
        self.positional_encodings = {}
        for d_model, max_len in zip(d_models, max_lens) :
            pe = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)
            self.positional_encodings[int(max_len ** 0.5)] = pe

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)

        pe_layer = self.positional_encodings[res]
        #pe = self.positional_encodings.expand(b_size, -1, -1)
        pe = pe_layer.expand(b_size, -1, -1)
        x = x + pe.to(x.device)
        if start_dim == 4:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)
        return x