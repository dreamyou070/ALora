import torch.nn as nn
import torch
import einops

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


class SinglePositionalEmbedding(nn.Module):

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

class AllPositionalEmbedding(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.layer_dict = layer_names_res_dim
        self.positional_encodings = {}
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]
            self.positional_encodings[layer_name] = SinglePositionalEmbedding(max_len = res*res, d_model = dim)

    def forward(self, x: torch.Tensor, layer_name):
        position_embedder = self.positional_encodings[layer_name]
        output = position_embedder(x)
        return output



class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self):
        super().__init__()
        image_size = (512,512)
        patch_size = (8,8)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        #self.num_channels = num_channels
        self.num_channels = 3
        self.num_patches = num_patches
        hidden_size = 4
        self.projection = nn.Conv2d(self.num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError("Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                             f" Expected {self.num_channels} but got {num_channels}.")
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(f"Input image size ({height}*{width}) doesn't match model"
                                 f" ({self.image_size[0]}*{self.image_size[1]}).")
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTEmbeddings(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.patch_embeddings = ViTPatchEmbeddings()
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, 4))

    def forward(self,pixel_values: torch.Tensor,) -> torch.Tensor:

        # [1] patch embedding the raw image
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        # [2] positional encoding
        embeddings = embeddings + self.position_embeddings
        b, pix_num, d = embeddings.shape
        res = int(pix_num ** 0.5)
        embeddings = embeddings.view(b, res, res, d).permute(0, 3, 1, 2)
        return embeddings


# layer_name down_blocks_1_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_1_transformer_blocks_0_attn2
# layer_name up_blocks_1_attentions_0_transformer_blocks_0_attn1
# layer_name mid_block_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_0_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_0_transformer_blocks_0_attn1
# layer_name up_blocks_1_attentions_0_transformer_blocks_0_attn2
# layer_name mid_block_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_2_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_0_attentions_1_transformer_blocks_0_attn2
# layer_name mid_block_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_2_attentions_0_transformer_blocks_0_attn2
# layer_name mid_block_attentions_0_transformer_blocks_0_attn2
# layer_name up_blocks_1_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn1
# layer_name up_blocks_1_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn2
# layer_name up_blocks_1_attentions_2_transformer_blocks_0_attn1
# layer_name up_blocks_1_attentions_0_transformer_blocks_0_attn1
# layer_name up_blocks_1_attentions_2_transformer_blocks_0_attn2