import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
"""
This Version of SALT uses:
    - W = U (\Sigma . A + B) + XY
    - we uses srLoRA
"""
class SALTLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int,  # rank for truncated SVD
        lora_rank: int,  # rank for rsLoRA
        alpha: float = 32.0,  # scaling factor for rsLoRA
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
        print("\nThe max possible rank is", max_possible_rank)
        
        # Initialize A and B for singular value transformation
        
        self.A = nn.Parameter(torch.ones(rank))
        self.B = nn.Parameter(torch.zeros(rank))
        self.A_frozen = torch.ones(max_possible_rank - self.A.shape[0])
        self.B_frozen = torch.ones(max_possible_rank - self.B.shape[0])
        
        # Initialize rsLoRA parameters with the new scaling
        rs_lora_scaling = alpha / (lora_rank ** 0.5)
        self.lora_X = nn.Parameter(torch.randn(out_features, lora_rank) * rs_lora_scaling)
        self.lora_Y = nn.Parameter(torch.randn(lora_rank, in_features) * rs_lora_scaling)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'A'):
            nn.init.ones_(self.A)
        if hasattr(self, 'B'):
            nn.init.zeros_(self.B)
        if hasattr(self, 'lora_X'):
            nn.init.normal_(self.lora_X, std=0.01)
        if hasattr(self, 'lora_Y'):
            nn.init.normal_(self.lora_Y, std=0.01)

    def perform_svd(self):
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.done_svd = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.done_svd:
            self.perform_svd()
        
        # Transform singular values: A·Σ_r + B
        A_total = torch.cat([self.A, self.A_frozen.to(input.device)])
        B_total = torch.cat([self.B, self.B_frozen.to(input.device)])
        transformed_S = A_total * self.S + B_total
        
        # Compute truncated SVD part: U_r(A·Σ_r + B)V_r^T
        weight_svd = self.U @ torch.diag(F.relu(transformed_S)) @ self.Vt
        
        # Add rsLoRA part: X·Y
        weight_rslora = self.lora_X @ self.lora_Y
        
        # Combine both parts
        weight_updated = weight_svd + weight_rslora
        
        # Compute regularization loss
        reg_loss = (
            torch.norm(1 - self.A) +
            torch.norm(self.B) +
            torch.norm(self.lora_X) * torch.norm(self.lora_Y)
        )
        
        return F.linear(input, weight_updated, self.bias), reg_loss

class SALTConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int,  # rank for truncated SVD
        lora_rank: int,  # rank for rsLoRA
        alpha: float = 1.0,  # scaling factor for rsLoRA
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert isinstance(kernel_size, int)
        
        # Reshape weight and perform SVD
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = False
        
        max_possible_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0])
        print("\nThe max possible rank is", max_possible_rank)
        self.actual_rank = min(rank, max_possible_rank)
        
        # Initialize A and B for singular value transformation
        self.A = nn.Parameter(torch.ones(self.actual_rank))
        self.B = nn.Parameter(torch.zeros(self.actual_rank))
        self.A_frozen = torch.ones(max_possible_rank - self.actual_rank)
        self.B_frozen = torch.ones(max_possible_rank - self.actual_rank)
        
        # Initialize rsLoRA parameters with scaling
        total_kernel_size = in_channels * kernel_size * kernel_size
        rs_lora_scaling = alpha / (lora_rank ** 0.5)
        self.lora_X = nn.Parameter(torch.randn(out_channels, lora_rank) * rs_lora_scaling)
        self.lora_Y = nn.Parameter(torch.randn(lora_rank, total_kernel_size) * rs_lora_scaling)
        
        # Freeze original weights
        self.weight.requires_grad = False
        
        # Save shapes for reshaping
        self.weight_shape = self.weight.shape
        self.reset_parameters()

    def perform_svd(self):
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = True        

    def reset_parameters(self) -> None:
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'A'):
            nn.init.ones_(self.A)
        if hasattr(self, 'B'):
            nn.init.zeros_(self.B)
        if hasattr(self, 'lora_X'):
            nn.init.normal_(self.lora_X, std=0.01)
        if hasattr(self, 'lora_Y'):
            nn.init.normal_(self.lora_Y, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.done_svd:
            self.perform_svd()

        A_total = torch.cat([self.A, self.A_frozen.to(x.device)])
        B_total = torch.cat([self.B, self.B_frozen.to(x.device)])
        transformed_S = A_total * self.S + B_total
        
        # Compute truncated SVD part: U_r(A·Σ_r + B)V_r^T
        weight_svd = self.U @ torch.diag(F.relu(transformed_S)) @ self.Vt
        
        # Add rsLoRA part: X·Y
        weight_rslora = self.lora_X @ self.lora_Y
        
        # Combine both parts
        weight_updated = weight_svd + weight_rslora
        
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
            torch.norm(1 - self.A) +
            torch.norm(self.B) +
            torch.norm(self.lora_X) * torch.norm(self.lora_Y)
        )
        
        return F.conv2d(
            x, weight_updated, self.bias, 
            self.stride, self.padding, 
            self.dilation, self.groups
        ), reg_loss
