import os
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim,
                                   2 * dim,
                                   bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

def sen2img(torch) :
    b,pix_num,dim = torch.shape
    pix_size = int(pix_num ** 0.5)
    torch = torch.view(b,pix_size,pix_size,dim).permute(0,3,1,2)
    return torch

feat_1 = torch.randn(1,64*64,320)
feat_1 = sen2img(feat_1)

feat_2 = torch.randn(1,32*32, 320*2)
feat_2 = sen2img(feat_2)

feat_3 = torch.randn(1,16*16, 320*4)
feat_3 = sen2img(feat_3)

feat_1_patch_embedder = PatchEmbed(img_size=64, patch_size=4, in_chans=320, embed_dim=320)
feat_1_patch = feat_1_patch_embedder(feat_1)

feat_2_patch_embedder = PatchEmbed(img_size=32, patch_size=2, in_chans=320*2, embed_dim=320*2)
feat_2_patch = feat_2_patch_embedder(feat_2)

feat_3_patch_embedder = PatchEmbed(img_size=16, patch_size=1, in_chans=320*4, embed_dim=320*4)
feat_3_patch = feat_3_patch_embedder(feat_3)

print(feat_1_patch.shape) # torch.Size([1, 256, 320])
print(feat_2_patch.shape) # torch.Size([1, 256, 640])
print(feat_3_patch.shape) # torch.Size([1, 256, 1280])

merger_1 = PatchMerging(input_resolution=(16,16), dim=320)
feat_1_patch = merger_1(feat_1_patch)

merger_2 = PatchMerging(input_resolution=(16,16), dim=320*2)
feat_2_patch = merger_2(feat_2_patch)

merger_3 = PatchMerging(input_resolution=(16,16), dim=320*4)
feat_3_patch = merger_3(feat_3_patch)

print(feat_1_patch.shape) # torch.Size([1, 64, 192])
print(feat_2_patch.shape) # torch.Size([1, 64, 384])
print(feat_3_patch.shape) # torch.Size([1, 64, 768])



