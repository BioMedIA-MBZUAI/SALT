import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from typing import Type


"""
This Version of SALT uses:
    - W = U (\Sigma . A + B) + XY
    - we uses normal LoRA
"""

class SALTLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int,  # rank for truncated SVD
        r_lora: int,  # rank for LoRA
        bias: bool = True, 
        device=None, 
        dtype=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # Perform full SVD initially
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.weight.requires_grad = False
        self.done_svd = False
        max_possible_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0])
        print("\nThe max possible rank is " , max_possible_rank)
        # Initialize A and B for singular value transformation
        self.trainable_A = nn.Parameter(torch.ones(rank))
        self.trainable_B = nn.Parameter(torch.zeros(rank))
        self.A_frozen = torch.ones(max_possible_rank-self.trainable_A.shape[0])
        self.B_frozen = torch.ones(max_possible_rank-self.trainable_B.shape[0])
        
        # Initialize LoRA parameters
        self.trainable_lora_X = nn.Parameter(torch.randn(out_features, r_lora) * 0.01)
        self.trainable_lora_Y = nn.Parameter(torch.randn(r_lora, in_features) * 0.01)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'trainable_A'):
            nn.init.ones_(self.trainable_A)
        if hasattr(self, 'trainable_B'):
            nn.init.zeros_(self.trainable_B)
        if hasattr(self, 'trainable_lora_X'):
            nn.init.normal_(self.trainable_lora_X, std=0.01)
        if hasattr(self, 'trainable_lora_Y'):
            nn.init.normal_(self.trainable_lora_Y, std=0.01)
    # No clue why they are using this
    def perform_svd(self):
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.done_svd = True
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.done_svd:
            self.perform_svd()
        # Transform singular values: A·Σ_r + B
        # We first cat the trainable + the frozen parameters
        A_total = torch.cat([self.trainable_A,self.A_frozen.to(input.device)])
        B_total = torch.cat([self.trainable_B,self.B_frozen.to(input.device)])
        transformed_S = A_total * self.S + B_total
        
        # Compute truncated SVD part: U_r(A·Σ_r + B)V_r^T
        weight_svd = self.U @ torch.diag(F.relu(transformed_S)) @ self.Vt
        
        # Add LoRA part: X·Y
        weight_lora = self.trainable_lora_X @ self.trainable_lora_Y
        
        # Combine both parts
        weight_updated = weight_svd + weight_lora
        
        # Compute regularization loss
        reg_loss = (
            torch.norm(1 - self.trainable_A) +  # A should be close to 1
            torch.norm(self.trainable_B) +      # B should be close to 0
            torch.norm(self.trainable_lora_X) * torch.norm(self.trainable_lora_Y)  # LoRA regularization
        )
        
        return F.linear(input, weight_updated, self.bias), reg_loss

class SALTConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int,  # rank for truncated SVD
        r_lora: int,  # rank for LoRA
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert isinstance(kernel_size, int)
        
        # Reshape weight and perform SVD
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = False
        # Ensure rank is not larger than the minimum dimension
        max_possible_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0])
        print("\nThe max possible rank is " , max_possible_rank)
        self.actual_rank = min(rank, max_possible_rank)
        
        # Initialize A and B for singular value transformation with correct size
        self.trainable_A = nn.Parameter(torch.ones(self.actual_rank))
        self.trainable_B = nn.Parameter(torch.zeros(self.actual_rank))
        self.A_frozen = torch.ones(max_possible_rank-self.actual_rank)
        self.B_frozen = torch.ones(max_possible_rank-self.actual_rank)
        # Initialize LoRA parameters
        total_kernel_size = in_channels * kernel_size * kernel_size
        self.trainable_lora_X = nn.Parameter(torch.randn(out_channels, r_lora) * 0.01)
        self.trainable_lora_Y = nn.Parameter(torch.randn(r_lora, total_kernel_size) * 0.01)
        
        # Freeze original weights
        self.weight.requires_grad = False
        
        # Save shapes for reshaping
        self.weight_shape = self.weight.shape
        self.reset_parameters()
    # No clue why they are using this
    def perform_svd(self):
        # shape
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = True        

    def reset_parameters(self) -> None:
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'trainable_A'):
            nn.init.ones_(self.trainable_A)
        if hasattr(self, 'trainable_B'):
            nn.init.zeros_(self.trainable_B)
        if hasattr(self, 'trainable_lora_X'):
            nn.init.normal_(self.trainable_lora_X, std=0.01)
        if hasattr(self, 'trainable_lora_Y'):
            nn.init.normal_(self.trainable_lora_Y, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.done_svd:
            self.perform_svd()

        A_total = torch.cat([self.trainable_A,self.A_frozen.to(x.device)])
        B_total = torch.cat([self.trainable_B,self.B_frozen.to(x.device)])
        transformed_S = A_total * self.S + B_total
        
        # Compute truncated SVD part: U_r(A·Σ_r + B)V_r^T
        weight_svd = self.U @ torch.diag(F.relu(transformed_S)) @ self.Vt
        
        # Add LoRA part: X·Y
        weight_lora = self.trainable_lora_X @ self.trainable_lora_Y
        
        # Combine both parts
        weight_updated = weight_svd + weight_lora
        
        # Reshape back to conv2d weight shape
        weight_updated = rearrange(
            weight_updated, 
            'co (cin h w) -> co cin h w', 
            cin=self.weight_shape[1], 
            h=self.weight_shape[2], 
            w=self.weight_shape[3]
        )
        
        # Compute regularization loss
        reg_loss = (
            torch.norm(1 - self.trainable_A) +  # A should be close to 1
            torch.norm(self.trainable_B) +      # B should be close to 0
            torch.norm(self.trainable_lora_X) * torch.norm(self.trainable_lora_Y)  # LoRA regularization
        )
        
        return F.conv2d(
            x, weight_updated, self.bias, 
            self.stride, self.padding, 
            self.dilation, self.groups
        ), reg_loss