import math
import os
from typing import Dict, List, Optional
import torch
from torch import nn
import einops
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

class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self,lora_name,
                 org_module: torch.nn.Module,
                 multiplier=1.0,
                 lora_dim=4,
                 alpha=1,
                 dropout=None,
                 rank_dropout=None,module_dropout=None,):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.is_linear = False
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.is_linear = True
        self.in_dim = in_dim
        self.out_dim = out_dim

        common_dim = gcd(in_dim, out_dim)
        self.common_dim = common_dim
        down_dim = int(in_dim // common_dim)
        up_dim = int(out_dim // common_dim)

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.org_weight = org_module.weight.detach().clone() #####################################################
        self.org_module_ref = [org_module]  ########################################################################

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        #del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale



class LoRAInfModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self,lora_name,
                 org_module: torch.nn.Module,
                 multiplier=1.0,
                 lora_dim=4,
                 alpha=1,
                 dropout=None,
                 rank_dropout=None,module_dropout=None,):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.is_linear = False
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.is_linear = True
        self.in_dim = in_dim
        self.out_dim = out_dim

        common_dim = gcd(in_dim, out_dim)
        self.common_dim = common_dim
        down_dim = int(in_dim // common_dim)
        up_dim = int(out_dim // common_dim)

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.org_weight = org_module.weight.detach().clone() #####################################################
        self.org_module_ref = [org_module]  ########################################################################

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        #del self.org_module

    def restore(self):
        self.org_module.forward = self.org_forward

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale



# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(multiplier, file, block_wise,
                                vae, text_encoder, unet, weights_sd=None,
                                for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim

    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    # LoRAInfModule for inference, LoRAModule for training
    module_class = LoRAInfModule if for_inference else LoRAModule
    network = LoRANetwork(text_encoder, unet,
                          block_wise=block_wise,
                          multiplier=multiplier,
                          modules_dim=modules_dim,
                          modules_alpha=modules_alpha,
                          # module_class = LoRAInfModule
                          module_class=module_class,
                          **kwargs )

    # block lr
    down_lr_weight, mid_lr_weight, up_lr_weight = parse_block_lr_kwargs(kwargs)
    if up_lr_weight is not None or mid_lr_weight is not None or down_lr_weight is not None:
        network.set_block_lr_weight(up_lr_weight, mid_lr_weight, down_lr_weight)

    return network, weights_sd


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

class PENetwork(torch.nn.Module):

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    def __init__(self, unet,) -> None:

        super().__init__()
        def create_modules(is_unet: bool,text_encoder_idx: Optional[int],  # None, 1, 2
                           root_module: torch.nn.Module,
                           target_replace_modules: List[torch.nn.Module],) -> List[LoRAModule]:
            prefix = (self.LORA_PREFIX_UNET if is_unet else (self.LORA_PREFIX_TEXT_ENCODER if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)))
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:

                                        loras.append(lora)
            return loras, skipped

        self.unet_pe = create_modules(True, None, unet, target_modules)
        skipped = skipped_te + skipped_un

        for name, module in unet.named_modules() :

            def forward(x)
                x = x + pe
                x = module(x)

def register_attention_control(unet: nn.Module,controller: AttentionStore):

    def ca_forward(self, layer_name):


        def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None,**model_kwargs):

            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

            """ cross self rechecking necessary """
            if argument.use_position_embedder :
                if not argument.use_multi_position_embedder and not argument.all_positional_embedder and not argument.patch_positional_self_embedder and not argument.all_self_cross_positional_embedder :
                    if layer_name == argument.position_embedding_layer :
                        hidden_states_pos = noise_type(hidden_states)
                        hidden_states = hidden_states_pos
                elif argument.use_multi_position_embedder :
                    if layer_name in argument.trg_layer_list :
                        hidden_states_pos = noise_type(hidden_states)
                        hidden_states = hidden_states_pos
                elif argument.all_positional_embedder :
                    if is_cross_attention :
                        hidden_states_pos = noise_type(hidden_states, layer_name)
                        hidden_states = hidden_states_pos
                elif argument.all_self_cross_positional_embedder :
                    #print(f'self and cross all positin embedding')
                    hidden_states_pos = noise_type(hidden_states, layer_name)
                    hidden_states = hidden_states_pos

                elif argument.patch_positional_self_embedder and is_cross_attention :
                    hidden_states_pos = noise_type(hidden_states, layer_name)
                    hidden_states = hidden_states_pos

            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)

            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self.upcast_attention:
                query = query.float()
                key = key.float()

            """ Second Trial """
            if trg_layer_list is not None and layer_name in trg_layer_list :
                controller.save_query((query * self.scale), layer_name) # query = batch, seq_len, dim
                controller.save_key(key, layer_name)

            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,)

            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)

            if trg_layer_list is not None and layer_name in trg_layer_list :
                trg_attn = attention_probs[:,:,:2]
                controller.save_attn(trg_attn, layer_name)


            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)

            return hidden_states

        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count


