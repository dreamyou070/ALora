from torch import nn

class QueryTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_queries = [8, 144],
                 nheads = 8,
                 num_layers = 9,
                 feedforward_dim = 2048,
                 mask_dim = 256,
                 pre_norm = False,
                 num_feature_levels = 3,
                 enforce_input_project = False,
                 with_fea2d_pos = True):

        super().__init__()

        self.pe_layer = None
        if in_channels!=hidden_dim or enforce_input_project:
            self.input_proj = nn.ModuleList()
            for _ in range(num_feature_levels):
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
        else:
            self.input_proj = None

        self.num_heads = nheads
        self.num_layers = num_layers
        self.transformer_selfatt_layers = nn.ModuleList()
        self.transformer_crossatt_layers = nn.ModuleList()
        self.transformer_feedforward_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_selfatt_layers.append(SelfAttentionLayer(
                    channels=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm, ))

            self.transformer_crossatt_layers.append(
                CrossAttentionLayer(
                    channels=hidden_dim,
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
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)

    def forward(self, x):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        fea2d = []
        fea2d_pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            if self.pe_layer is not None:
                pi = self.pe_layer(x[i], None).flatten(2)
                pi = pi.transpose(1, 2)
            else:
                pi = None
            xi = self.input_proj[i](x[i]) if self.input_proj is not None else x[i]
            xi = xi.flatten(2) + self.level_embed.weight[i][None, :, None]
            xi = xi.transpose(1, 2)
            fea2d.append(xi)
            fea2d_pos.append(pi)

        bs, _, _ = fea2d[0].shape
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
                                                      qk_pos = torch.cat([gquery_pos, lquery_pos], dim=1),)
            qout = self.transformer_feedforward_layers[i](qout)
            # [2.3] original shape
            gquery = qout[:, :num_gq]
            lquery = qout[:, num_gq:]

        output = torch.cat([gquery, lquery], dim=1)

        return output