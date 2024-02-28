import torch
from torch import nn
import numpy.random as npr
import math


class PPE_MLP(nn.Module):
    def __init__(self, freq_num=20, freq_max=None, out_channel=768, mlp_layer=3):
        import math

        super().__init__()
        self.freq_num = freq_num  # 20
        self.freq_max = freq_max  #
        self.out_channel = out_channel  # 768
        self.mlp_layer = mlp_layer  # 3
        self.twopi = 2 * math.pi  # 2 * pi

        mlp = []
        in_channel = freq_num * 4  # 80

        for idx in range(mlp_layer):
            # 3 of linear layers
            linear = nn.Linear(in_channel, out_channel, bias=True)
            nn.init.xavier_normal_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            mlp.append(linear)
            if idx != mlp_layer - 1:
                mlp.append(nn.SiLU())
            in_channel = out_channel
        self.mlp = nn.Sequential(*mlp)
        nn.init.constant_(self.mlp[-1].weight, 0)

    def forward(self, x, mask=None):
        assert mask is None, "Mask not implemented"
        h, w = x.shape[-2:]
        minlen = min(h, w)
        h_embed, w_embed = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        if self.training:
            import numpy.random as npr
            pertube_h, pertube_w = npr.uniform(-0.5, 0.5), npr.uniform(-0.5, 0.5)
        else:
            pertube_h, pertube_w = 0, 0
        h_embed = (h_embed + 0.5 - h / 2 + pertube_h) / (minlen) * self.twopi
        w_embed = (w_embed + 0.5 - w / 2 + pertube_w) / (minlen) * self.twopi
        h_embed, w_embed = h_embed.to(x.device).to(x.dtype), w_embed.to(x.device).to(x.dtype)
        dim_t = torch.linspace(0, 1, self.freq_num, dtype=torch.float32, device=x.device)
        freq_max = self.freq_max if self.freq_max is not None else minlen / 2  # 8
        dim_t = freq_max ** dim_t.to(x.dtype)
        pos_h = h_embed[:, :, None] * dim_t
        pos_w = w_embed[:, :, None] * dim_t
        pos = torch.cat((pos_h.sin(), pos_h.cos(), pos_w.sin(), pos_w.cos()), dim=-1)  # 4 배
        pos = self.mlp(pos)
        pos = pos.permute(2, 0, 1)[None]  # dim, res, res
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

pe_layer = PPE_MLP(freq_num=20,
                       freq_max=None,
                       out_channel=768,
                       mlp_layer=3)

query_list = [torch.randn(1,1280,8,8),torch.randn(1,1280,16,16),torch.randn(1,640,32,32),torch.randn(1,320,64,64)]