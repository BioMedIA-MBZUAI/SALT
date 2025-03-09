import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from typing import Type


class SVDLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, mlp_transform=False, fraction_trainable=1) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.weight.requires_grad = False
        self.done_svd = False
        self.mlp_transform = mlp_transform
        if mlp_transform:
            self.trainable_mlp = MLPBlock2(
                embedding_dim=self.S.shape[0],
                mlp_dim=256
            )
        else:
            S_len = (self.S.shape[0])
            # self.trainable_scale = nn.Parameter(torch.ones(int(S_len*1)))
            self.trainable_scale = nn.Parameter(torch.ones(int(S_len*fraction_trainable)))
            # self.trainable_shift = nn.Parameter(torch.zeros(int(S_len*0)))
            self.trainable_shift = nn.Parameter(torch.zeros(int(S_len*fraction_trainable)))
            self.frozen_scale = torch.ones(S_len-self.trainable_scale.shape[0])
            self.frozen_shift = torch.ones(S_len - self.trainable_shift.shape[0])
        self.reset_parameters()

    def perform_svd(self):
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.done_svd = True

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'trainable_shift'):
            nn.init.zeros_(self.trainable_shift)
        if hasattr(self, 'trainable_scale'):
            nn.init.ones_(self.trainable_scale)

    def forward(self, input: torch.Tensor):
        if not self.done_svd:
            self.perform_svd()
        if self.mlp_transform:
            s_new = (self.trainable_mlp((self.S.to(input.device)).flatten())).reshape(self.S.shape)
            weight_updated = self.U.to(input.device, dtype=input.dtype) @ torch.diag(F.relu(s_new)).to(input.device) @ self.Vt.to(device=input.device, dtype=input.dtype)
            reg_loss = torch.norm(s_new - self.S)
        else:
            scale = torch.cat([self.trainable_scale,self.frozen_scale.to(input.device)])
            shift = torch.cat([self.trainable_shift, self.frozen_shift.to(input.device)])
            weight_updated = self.U.to(input.device, dtype=input.dtype) @ torch.diag(F.relu(scale.to(input.device, dtype=input.dtype)*self.S.to(input.device, dtype=input.dtype) + shift)) @ self.Vt.to(device=input.device, dtype=input.dtype)
            reg_loss = torch.norm(1 - self.trainable_scale) + torch.norm(self.trainable_shift)
        return F.linear(input, weight_updated, self.bias), reg_loss

#adapted from https://github.com/phymhan/SVDiff
class SVDConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        scale: float = 1.0,
        mlp_transform: bool = False,
        fraction_trainable=1,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        assert type(kernel_size) is int
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        # initialize to 0 for smooth tuning 
        
        self.weight.requires_grad = False
        self.done_svd = False
        self.mlp_transform = mlp_transform
        if mlp_transform:
            self.trainable_mlp = MLPBlock2(
                embedding_dim=self.S.shape[0],
                mlp_dim=256
            )
        else:
            S_len = (self.S.shape[0])
            # self.trainable_scale = nn.Parameter(torch.ones(int(S_len*1)))
            self.trainable_scale = nn.Parameter(torch.ones(int(S_len*fraction_trainable)))
            # self.trainable_shift = nn.Parameter(torch.zeros(int(S_len*0)))
            self.trainable_shift = nn.Parameter(torch.zeros(int(S_len*fraction_trainable)))
            self.frozen_scale = torch.ones(S_len-self.trainable_scale.shape[0])
            self.frozen_shift = torch.ones(S_len - self.trainable_shift.shape[0])
        self.reset_parameters()

    def perform_svd(self):
        # shape
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = True        
        
    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'trainable_shift'):
            nn.init.zeros_(self.trainable_shift)
        if hasattr(self, 'trainable_scale'):
            nn.init.ones_(self.trainable_scale)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        
        if self.mlp_transform:
            s_new = (self.trainable_mlp((self.S.to(x.device)).flatten())).reshape(self.S.shape)
            weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(s_new)).to(x.device) @ self.Vt.to(device=x.device, dtype=x.dtype)
            reg_loss = torch.norm(s_new - self.S)
        
        else:
            scale = torch.cat([self.trainable_scale,self.frozen_scale.to(x.device)])
            shift = torch.cat([self.trainable_shift, self.frozen_shift.to(x.device)])
            weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(scale.to(x.device, dtype=x.dtype)*self.S.to(x.device, dtype=x.dtype) + shift)) @ self.Vt.to(device=x.device, dtype=x.dtype)
            reg_loss = torch.norm(1 - self.trainable_scale) + torch.norm(self.trainable_shift)

        weight_updated = rearrange(weight_updated, 'co (cin h w) -> co cin h w', cin=self.weight.size(1), h=self.weight.size(2), w=self.weight.size(3))
        
        return F.conv2d(x, weight_updated, self.bias, self.stride, self.padding, self.dilation, self.groups), reg_loss


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
