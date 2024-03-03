from torch.nn import L1Loss
import torch
from torch import nn

layer_names_res_dim = {'down_blocks_0_attentions_0_transformer_blocks_0_attn2': (64, 320),
                       'down_blocks_0_attentions_1_transformer_blocks_0_attn2': (64, 320),

                       'down_blocks_1_attentions_0_transformer_blocks_0_attn2': (32, 640),
                       'down_blocks_1_attentions_1_transformer_blocks_0_attn2': (32, 640),

                       'down_blocks_2_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                       'down_blocks_2_attentions_1_transformer_blocks_0_attn2': (16, 1280),

                       'mid_block_attentions_0_transformer_blocks_0_attn2': (8, 1280),

                       'up_blocks_1_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                       'up_blocks_1_attentions_1_transformer_blocks_0_attn2': (16, 1280),
                       'up_blocks_1_attentions_2_transformer_blocks_0_attn2': (16, 1280),

                       'up_blocks_2_attentions_0_transformer_blocks_0_attn2': (32, 640),
                       'up_blocks_2_attentions_1_transformer_blocks_0_attn2': (32, 640),
                       'up_blocks_2_attentions_2_transformer_blocks_0_attn2': (32, 640),

                       'up_blocks_3_attentions_0_transformer_blocks_0_attn2': (64, 320),
                       'up_blocks_3_attentions_1_transformer_blocks_0_attn2': (64, 320),
                       'up_blocks_3_attentions_2_transformer_blocks_0_attn2': (64, 320), }

# only channel is changed,
class GCN(nn.Module):
    def __init__(self, c, out_c, k=(7, 7)):  # out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c,
                                 kernel_size=(k[0], 1),                     # kernel_size = (7,1)
                                 padding=(int((int(k[0] - 1) / 2)), 0))          # (3,0)
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding=(0, int((k[0] - 1) / 2)))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k[1]), padding=(0, int((k[1] - 1) / 2)))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding=(int((k[1] - 1) / 2), 0))

    def forward(self, x):

        x_l = self.conv_l1(x.to(self.conv_l1.device))
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class AllGCN(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.layer_dict = layer_names_res_dim
        self.gcn = {}
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]
            k_size = int(res/2) - 1
            self.gcn[layer_name] = GCN(c=dim, out_c=dim,
                                       k = (k_size, k_size))

    def forward(self, x: torch.Tensor, layer_name):
        b, pix_num, dim = x.shape
        res = int(pix_num** 0.5)
        x2d = x.permute(0, 2, 1).view(b, dim, res, res)

        if layer_name in self.gcn.keys() :
            global_conv = self.gcn[layer_name]
            out = global_conv(x2d)
            out = out.permute(0, 2, 3, 1)
            out1d = out.view(1, pix_num, dim)
            return out1d
        else :
            return x