import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
def c2_xavier_fill(module):
    # Caffe2 implementation of XavierFill in fact
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def with_pos_embed(x, pos):
    return x if pos is None else x + pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class SelfAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self,
                     qkv,
                     qk_pos=None,
                     mask=None, ):
        h = qkv
        qk = with_pos_embed(qkv, qk_pos).transpose(0, 1)
        v = qkv.transpose(0, 1)
        h1 = self.self_attn(qk, qk, v, attn_mask=mask)[0]
        h1 = h1.transpose(0, 1)
        h = h + self.dropout(h1)
        h = self.norm(h)
        return h

    def forward_pre(self, tgt,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None):
        # deprecated
        assert False
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


class CrossAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self,
                     q,
                     kv,
                     q_pos=None,
                     k_pos=None,
                     mask=None, ):
        h = q
        q = with_pos_embed(q, q_pos).transpose(0, 1)
        k = with_pos_embed(kv, k_pos).transpose(0, 1)
        v = kv.transpose(0, 1)
        h1 = self.multihead_attn(q, k, v, attn_mask=mask)[0]
        h1 = h1.transpose(0, 1)
        h = h + self.dropout(h1)
        h = self.norm(h)
        return h

    def forward_pre(self, tgt, memory,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=None):
        # Deprecated
        assert False
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


class PPE_MLP(nn.Module):
    def __init__(self, freq_num=20, freq_max=None, out_channel=768, mlp_layer=3):
        import math

        super().__init__()
        self.freq_num = freq_num # 20
        self.freq_max = freq_max #
        self.out_channel = out_channel # 768
        self.mlp_layer = mlp_layer # 3
        self.twopi = 2 * math.pi # 2 * pi

        mlp = []
        in_channel = freq_num * 4 # 80

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
        freq_max = self.freq_max if self.freq_max is not None else minlen / 2 # 8
        dim_t = freq_max ** dim_t.to(x.dtype)
        pos_h = h_embed[:, :, None] * dim_t
        pos_w = w_embed[:, :, None] * dim_t
        pos = torch.cat((pos_h.sin(), pos_h.cos(), pos_w.sin(), pos_w.cos()), dim=-1) # 4 배
        pos = self.mlp(pos)
        pos = pos.permute(2, 0, 1)[None] # dim, res, res
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


class FeedForwardLayer(nn.Module):
    def __init__(self, channels, hidden_channels=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, x):
        h = x
        h1 = self.linear2(self.dropout(self.activation(self.linear1(h))))
        h = h + self.dropout(h1)
        h = self.norm(h)
        return h

    def forward_pre(self, x):
        xn = self.norm(x)
        h = x
        h1 = self.linear2(self.dropout(self.activation(self.linear1(xn))))
        h = h + self.dropout(h1)
        return h

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


class QueryTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_queries = [8*8, 64*64],
                 nheads = 8,
                 num_layers = 9,
                 feedforward_dim = 2048,
                 mask_dim = 256,
                 pre_norm = False,
                 num_feature_levels = 3,
                 enforce_input_project = False,
                 with_fea2d_pos = True):

        super().__init__()

        if with_fea2d_pos:
            self.pe_layer = PPE_MLP(freq_num=20,
                                    freq_max=None,
                                    out_channel=hidden_dim,
                                    mlp_layer=3)
        else:
            self.pe_layer = None
        if in_channels!=hidden_dim or enforce_input_project:
            self.input_proj = nn.ModuleList()
            for j in range(num_feature_levels):
                self.input_proj.append(nn.Conv2d(in_channels[j],
                                                 hidden_dim,
                                                 kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
        else:
            self.input_proj = None

        self.num_heads = nheads
        self.num_layers = num_layers
        self.transformer_selfatt_layers = nn.ModuleList()
        self.transformer_crossatt_layers = nn.ModuleList()
        self.transformer_feedforward_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_selfatt_layers.append(SelfAttentionLayer(channels=hidden_dim,
                                                                      nhead=nheads,
                                                                      dropout=0.0,
                                                                      normalize_before=pre_norm, ))
            self.transformer_crossatt_layers.append(CrossAttentionLayer(channels=hidden_dim,
                                                                        nhead=nheads,
                                                                        dropout=0.0,
                                                                        normalize_before=pre_norm, ))
            self.transformer_feedforward_layers.append(
                FeedForwardLayer(
                    channels=hidden_dim,
                    hidden_channels=feedforward_dim,
                    dropout=0.0,
                    normalize_before=pre_norm, ))

        self.num_queries = num_queries
        # -----------------------------------------------------------------------------
        # global query, local query
        num_gq, num_lq = self.num_queries # 4, 144
        self.init_query = nn.Embedding(num_gq+num_lq, hidden_dim)
        self.query_pos_embedding = nn.Embedding(num_gq+num_lq, hidden_dim)

        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)  # 4,768

    def forward(self, x):

        # [1] check how many feature to use
        assert len(x) == self.num_feature_levels

        # [2]
        fea2d = []
        fea2d_pos = []
        size_list = []
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:]) # pix_num, dim
            # [1] feature2d position
            if self.pe_layer is not None:

                pi = self.pe_layer(x[i], None).flatten(2)
                pi = pi.transpose(1, 2)
            else:
                pi = None
            fea2d_pos.append(pi)

            # [2] feature in 2d
            xi = self.input_proj[i](x[i]) if self.input_proj is not None else x[i]
            xi = xi.flatten(2) + self.level_embed.weight[i][None, :, None]
            xi = xi.transpose(1, 2) # 1,1,768
            fea2d.append(xi) # feature 2d

        bs, _, _ = fea2d[0].shape #
        num_gq, num_lq = self.num_queries
        # [1] start from scratch query
        gquery = self.init_query.weight[:num_gq].unsqueeze(0).repeat(bs, 1, 1)
        lquery = self.init_query.weight[num_gq:].unsqueeze(0).repeat(bs, 1, 1)
        gquery_pos = self.query_pos_embedding.weight[:num_gq].unsqueeze(0).repeat(bs, 1, 1)
        lquery_pos = self.query_pos_embedding.weight[num_gq:].unsqueeze(0).repeat(bs, 1, 1)

        # [2] transformer layers
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # [2.1] cross attention with ony local query
            qout = self.transformer_crossatt_layers[i](q = lquery,
                                                       kv = fea2d[level_index],
                                                       q_pos = lquery_pos,
                                                       k_pos = fea2d_pos[level_index],
                                                       mask = None,)
            lquery = qout
            # [2.2] self attention and FF with both global and local queries
            qout = self.transformer_selfatt_layers[i](qkv = torch.cat([gquery, lquery], dim=1),
                                                      qk_pos = torch.cat([gquery_pos, lquery_pos], dim=1),) # 768 dim
            qout = self.transformer_feedforward_layers[i](qout)
            # [2.3] original shape
            gquery = qout[:, :num_gq] # batch, len, dim
            lquery = qout[:, num_gq:] # batch, len, dim
        #output = torch.cat([gquery, lquery], dim=1)
        return gquery, lquery

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
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # conv2d : 1280dim -> 768dim
        # b,768dim,res,res
        # b,768dim,pix_len
        # b,pix_len, 768dim
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



class GlobalQueryTransformer(nn.Module):

    def __init__(self,
                 hidden_dim=768,
                 num_feature_levels = 4,
                 with_fea2d_pos=True):

        super().__init__()

        # [1] semantic tokens
        if with_fea2d_pos:
            self.pe_layer = PPE_MLP(freq_num=20,
                                    freq_max=None,
                                    out_channel=hidden_dim,
                                    mlp_layer=3)
        else:
            self.pe_layer = None

        self.num_feature_levels = num_feature_levels
        self.patch_embeddings = nn.ModuleList()
        base_channels = [1280, 1280, 640, 320]
        for j in range(num_feature_levels) :
            self.patch_embeddings.append(PatchEmbed(patch_size = 2 ** j,
                                                    img_size = 8*(2 ** j),
                                                    in_chans=base_channels[j],
                                                    embed_dim = hidden_dim))
    def forward(self, x):

        # [1] check how many feature to use
        # len(x) = 4
        assert len(x) == self.num_feature_levels

        # [2] patch embedding
        pi = 0
        for i in range(self.num_feature_levels):
            xi = x[i] # pix_num, dim
            #pix_num, dim = xi.shape
            head, pix_num, dim = xi.shape
            res = int(pix_num ** 0.5)
            xi = xi.transpose(-1,-2) # head, dim, pix_num
            #xi = xi.view(-1, res,res).unsqueeze(0) # 1,    dim, res, res
            xi = xi.view(head, -1, res, res)        # head, dim, res, res
            x[i] = xi
            pi += self.patch_embeddings[i](xi) # 1, 8**2, dim
        si = self.pe_layer(x[0]).permute(0,2,3,1) # 1, res, res, dim
        import einops
        si = einops.rearrange(si, 'p a b c -> p (a b) c')
        return (pi,pi+si)