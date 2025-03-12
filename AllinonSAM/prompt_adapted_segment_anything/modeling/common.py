# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type
from .svd_layers import SVDLinear
from .SALT_layers_please_work import SALTLinear
# from .SALT_layers_3 import SALTLinear , SALTConv2d
from .lora_layers import LoRAConv2D, LoRALinear
# from .SALT_layer_dynamic import SALTLinear , SALTConv2d
from .DoRA import DoRALinear, DoRAConv2d
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        mlp_transform=False,
        use_lora = False
    ) -> None:
        super().__init__()
        if use_lora:
            self.lin1 = LoRALinear(embedding_dim, mlp_dim)
            self.lin2 = LoRALinear(mlp_dim, embedding_dim)
        else:
            self.lin1 = SVDLinear(embedding_dim, mlp_dim, mlp_transform=mlp_transform)
            self.lin2 = SVDLinear(mlp_dim, embedding_dim, mlp_transform=mlp_transform)
            # rank_value = 50
            # # print("\nEmbedding dim in MLP Block is" ,embedding_dim)
            # # print("\n no need for MLP transform" , mlp_transform)
            # self.lin1 = SALTLinear(embedding_dim, mlp_dim, rank=rank_value , r_lora=256)
            # self.lin2 = SALTLinear(mlp_dim, embedding_dim, rank=rank_value , r_lora=256)
            
            # self.lin1 = DoRALinear(embedding_dim, mlp_dim , r=64)
            # self.lin2 = DoRALinear(mlp_dim, embedding_dim, r=64)
            # dynamic
            # self.lin1 = SALTLinear(embedding_dim, mlp_dim , r_lora=256)
            # self.lin2 = SALTLinear(mlp_dim, embedding_dim , r_lora=256)
        self.act = act()

    def forward(self, x: torch.Tensor, output_loss=True) -> torch.Tensor:
        out, reg_loss1 = self.lin1(x)
        out, reg_loss2 = self.lin2(self.act(out))
        if output_loss:
            return out, (reg_loss1+reg_loss2)
        else:
            return out 

class MLPBlock2(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin1(x)
        out = self.lin2(self.act(out))
        return out 


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
