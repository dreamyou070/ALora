from torch.nn import L1Loss
import torch
from torch import nn

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
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x

out_c = 21
c = 320
x = torch.randn(1,320, 8,8)
gcn = GCN(c=c, out_c=out_c, k = (7,7))
conv_l1 = gcn.conv_l1
out = gcn(x)
#print(out.shape)
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
for layer_name in layer_names_res_dim.keys() :
    res, dim = layer_names_res_dim[layer_name]
    pix_num = res * res
    hidden_states = torch.randn(1,res*res,dim)
    hidden_states2d = hidden_states.permute(0,2,1).view(1, dim, res,res)
    #print(f'hidden_states2d = {hidden_states2d.shape}')
    kernel_size = int(res/2)-1
    gcn = GCN(c=dim, out_c=dim, k=(kernel_size,kernel_size))
    out = gcn(hidden_states2d.to(dtype=gcn.weight_dtype))
    out = out.permute(0,2,3,1)
    out1d = out.view(1,pix_num,dim)
    print(f'layer_name = {layer_name} | output = {out1d.shape}')