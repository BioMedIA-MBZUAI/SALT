# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
import sys
sys.path.append("/home/abdelrahman.elsayed/med-cvpr/AllinonSAM/prompt_adapted_SAM2")
from prompt_adapted_SAM2.modeling.Adapters.SALT import SALTLinear
from prompt_adapted_SAM2.modeling.Adapters.LoRA import LoRALinear
from prompt_adapted_SAM2.modeling.Adapters.SVD import SVDLinear

from prompt_adapted_SAM2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from prompt_adapted_SAM2.modeling.sam2_utils import DropPath, MLP2, MLP

###### ADAPTERS PARAMETERS #########
# don't use none there is a problem with it
global_adapter = "salt"
####### SALT #######
r_lora_g = 256
ratio_salt_g = 0.1
######## LoRA ########
r_g = 256
######## SVD ########
fraction_trainable_g = 1
    
    
# Function to help you choose your needed Adapter
def get_adapter_linear(in_features: int, 
                       out_features: int, 
                       adapter: str = None, 
                       **kwargs) -> nn.Module:
    """
    Returns a linear layer using the specified adapter.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        adapter (str, optional): One of "salt", "lora", "svd", or None.
            If None or "none", returns a plain linear adapter.
        **kwargs: Additional keyword arguments for the adapter (e.g., r_lora, ratio_salt).
    
    Returns:
        nn.Module: An instance of the chosen linear adapter.
    """
    if adapter.lower() == "salt":
        return SALTLinear(in_features, out_features, **kwargs)
    elif adapter.lower() == "lora":
        return LoRALinear(in_features, out_features, **kwargs)
    elif adapter.lower() == "svd":
        return SVDLinear(in_features, out_features, **kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter}")


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
        adapter: str = None  # adapter type: "salt", "lora", "svd", or None
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.adapter = adapter
        # self.qkv = nn.Linear(dim, dim_out * 3)
        # self.proj = nn.Linear(dim_out, dim_out)s
        # Create qkv and proj layers using the adapter factory.
        if adapter == "salt": 
            self.qkv = get_adapter_linear(dim, dim_out * 3, adapter, r_lora=r_lora_g, ratio_salt=ratio_salt_g) #Adapter Modification
            self.proj = get_adapter_linear(dim_out, dim_out, adapter, r_lora=r_lora_g, ratio_salt=ratio_salt_g)#Adapter Modification
        elif adapter == "lora":
            self.qkv = get_adapter_linear(dim, dim_out * 3, adapter, r=r_g) #Adapter Modification
            self.proj = get_adapter_linear(dim_out, dim_out, adapter, r=r_g)#Adapter Modification 
        elif adapter=="svd":
            self.qkv = get_adapter_linear(dim, dim_out * 3, adapter) #Adapter Modification
            self.proj = get_adapter_linear(dim_out, dim_out, adapter)#Adapter Modification 
        else:
            print("Local Attention , using normal linear layers")
            self.qkv = nn.Linear(dim, dim_out * 3)
            self.proj = nn.Linear(dim_out, dim_out)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        
        # print(f"The X shape: {x.shape}")
        # print(f"The dim shape: {self.dim}")
        # print(f"The dim_out shape: {self.dim_out}")

        # qkv with shape (B, H * W, 3, nHead, C)
        if self.adapter != "none":
            qkv,reg_loss1 = self.qkv(x)
        else:
            qkv = self.qkv(x)
            reg_loss1=0
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)
        if self.adapter != "none":
             x,reg_loss2 = self.proj(x)
        else:
            x = self.proj(x)
            reg_loss2 = 0

        return x , (reg_loss1+reg_loss2)


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        adapter: str = None  # New parameter for the adapter type
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )
        # apply it to the global/local attention only
        # if self.window_size == 0: #global
        #     adapter = "none"
        # Sanity Check
        self.adapter= adapter
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
            adapter = adapter
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        # adapter = "none"
        if adapter == "salt":
            self.mlp = MLP2(
                dim_out,
                int(dim_out * mlp_ratio),
                dim_out,
                num_layers=2,
                activation=act_layer,
                adapter = adapter, r_lora=r_lora_g, ratio_salt=ratio_salt_g #Adapter Modification
            )
        elif adapter == "lora":
             self.mlp = MLP2(
                dim_out,
                int(dim_out * mlp_ratio),
                dim_out,
                num_layers=2,
                activation=act_layer,
                adapter = adapter, r=r_g #Adapter Modification
            )
        elif adapter == "svd":
            self.mlp = MLP2(
                dim_out,
                int(dim_out * mlp_ratio),
                dim_out,
                num_layers=2,
                activation=act_layer,
                adapter = adapter , fraction_trainable=fraction_trainable_g #Adapter Modification
            )
        else:
            # we use normal MLP
            self.mlp = MLP(
                dim_out,
                int(dim_out * mlp_ratio),
                dim_out,
                num_layers=2,
                activation=act_layer,
            )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x,reg_loss = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        # print(x)
        # If global /local attention
        # if self.window_size != 0: # local
        if self.adapter != "none": # for MLP
            mlp_output , mlp_reg_loss = self.mlp(self.norm2(x))
            x = x + self.drop_path(mlp_output)
            reg_loss += mlp_reg_loss
        else:
            mlp_output = self.mlp(self.norm2(x))
            mlp_reg_loss=0
            x = x + self.drop_path(mlp_output)
            reg_loss += mlp_reg_loss
            
        return x,reg_loss


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
        adapter="salt",
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            # Filterting by block indices
            in_later_stages = i >= 0 
            # global Attention
            # in_later_stages = i in global_att_blocks
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                adapter=global_adapter if in_later_stages else "none", # Temprorarily set it to global adapter
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))
        # just checking for the global attention blocks
        for i, block in enumerate(self.blocks):
            print(f"Block {i} → window_size: {block.window_size}, global_att_blocks: {self.global_att_blocks}, is_global: {block.window_size == 0} , Adapter Type: {block.adapter}")


    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        # Accumlate the reg_loss
        total_reg_loss = 0.0
        
        outputs = []
        for i, blk in enumerate(self.blocks):
            x,reg_loss = blk(x)
        
            total_reg_loss += reg_loss
            
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs , total_reg_loss

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
