import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from typing import Type, Tuple, Optional

class SALTLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int,  # truncation rank for SVD
        r_lora: int = 8,  # LoRA rank
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seed: int = 42
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        torch.manual_seed(seed)
        
        # Initialize parameters for SVD
        self.weight.requires_grad = False
        self.done_svd = False
        self.U, self.S, self.Vt = self._initialize_svd()
        
        max_possible_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0]) # 
        # print("\nThe max possible rank is", max_possible_rank)
        print(f"\nLayer size: {in_features}x{out_features}")
        print(f"Max possible rank: {max_possible_rank}")
        print(f"Using rank: {rank}, r_lora: {r_lora}")
        
        # Count parameters
        scale_shift_params = rank * 2
        lora_params = (max_possible_rank - rank) * r_lora * 2
        total_params = scale_shift_params + lora_params
        print(f"Scale/shift parameters: {scale_shift_params}")
        print(f"LoRA parameters: {lora_params}")
        print(f"Total trainable parameters: {total_params}")

        # Truncation rank for SVD
        self.rank = rank
        
        # Initialize scaling and shifting parameters for top singular values
        self.trainable_scale_A = nn.Parameter(torch.ones(rank))
        self.trainable_shift_B = nn.Parameter(torch.zeros(rank))
        
        # Initialize LoRA matrices for remaining singular values
        remaining_rank = max_possible_rank - rank
        self.trainable_X = nn.Parameter(torch.randn(remaining_rank, r_lora) * 0.01)
        self.trainable_Y = nn.Parameter(torch.randn(r_lora, remaining_rank) * 0.01)
        self._verify_parameters()
        self.reset_parameters()

    def _verify_parameters(self):
        """Print trainable parameter information"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nVerifying SALTLinear parameters:")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {total_params}")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape} (trainable: {param.requires_grad})")

    def _initialize_svd(self):
        """Initializes SVD decomposition on the weight matrix."""
        return torch.linalg.svd(self.weight, full_matrices=False)

    def perform_svd(self) -> None:
        """Updates truncated SVD decomposition on the weight matrix."""
        self.U, self.S, self.Vt = self._initialize_svd()
        self.done_svd = True

    def get_modified_singular_values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes modified singular values using both scaling/shifting and LoRA adaptation.
        Returns:
            Tuple containing:
                - Modified singular values tensor
                - Scale/shift modification term
                - LoRA adaptation term
        """
        # Create diagonal matrix of original singular values
        S_diag = torch.diag(self.S)
        
        # Apply scaling and shifting to top rank singular values
        top_s = self.S[:self.rank]
        modified_top_s = self.trainable_scale_A * top_s + self.trainable_shift_B
        
        # Compute LoRA term for remaining singular values
        loRA_term = self.trainable_X @ self.trainable_Y
        
        # Create the combined singular value matrix
        new_s = S_diag.clone()
        new_s[:self.rank, :self.rank] = torch.diag(modified_top_s)
        new_s[self.rank:, self.rank:] += loRA_term
        
        scale_shift_term = torch.zeros_like(S_diag)
        scale_shift_term[:self.rank, :self.rank] = torch.diag(modified_top_s) - torch.diag(top_s)
        
        return new_s, scale_shift_term, loRA_term

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with both scaling/shifting and LoRA modifications.
        
        Args:
            input: Input tensor
            
        Returns:
            Tuple containing:
                - Output tensor after linear transformation
                - Combined regularization loss
        """
        if not self.done_svd:
            self.perform_svd()

        new_s, scale_shift_term, LoRA_term = self.get_modified_singular_values()
        s_new = F.relu(new_s.to(input.device))

        # Reconstruct weight matrix using modified components
        weight_updated = self.U @ s_new @ self.Vt
        
        # Compute regularization losses
        scale_shift_reg = torch.norm(scale_shift_term)
        lora_reg = torch.norm(LoRA_term)
        reg_loss = scale_shift_reg + lora_reg

        return F.linear(input, weight_updated, self.bias), reg_loss


class SALTConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        rank: int,  # truncation rank for SVD
        r_lora: int = 8,  # LoRA rank
        seed: int = 42,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        torch.manual_seed(seed)
        
        self.done_svd = False
        self.weight.requires_grad = False

        # Reshape weight and perform initial truncated SVD
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = self._initialize_svd(weight_reshaped)

        max_possible_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0])
        print("\nThe max possible rank is", max_possible_rank)
         # Count parameters
        scale_shift_params = rank * 2
        lora_params = (max_possible_rank - rank) * r_lora * 2
        total_params = scale_shift_params + lora_params
        print(f"Scale/shift parameters: {scale_shift_params}")
        print(f"LoRA parameters: {lora_params}")
        print(f"Total trainable parameters: {total_params}")


        self.rank = rank
        
        # Initialize scaling and shifting parameters for top singular values
        self.trainable_scale_A = nn.Parameter(torch.ones(rank))
        self.trainable_shift_B = nn.Parameter(torch.zeros(rank))
        
        # Initialize LoRA matrices for remaining singular values
        remaining_rank = max_possible_rank - rank
        self.trainable_X = nn.Parameter(torch.randn(remaining_rank, r_lora) * 0.01)
        self.trainable_Y = nn.Parameter(torch.randn(r_lora, remaining_rank) * 0.01)
        self._verify_parameters()
        self.reset_parameters()
    

    def _verify_parameters(self):
        """Print trainable parameter information"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nVerifying SALTConv2d parameters:")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {total_params}")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape} (trainable: {param.requires_grad})")


    def _initialize_svd(self, weight_reshaped):
        """Initializes SVD decomposition on the reshaped weight matrix."""
        return torch.linalg.svd(weight_reshaped, full_matrices=False)

    def perform_svd(self) -> None:
        """Updates truncated SVD decomposition on the reshaped weight matrix."""
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = self._initialize_svd(weight_reshaped)
        self.done_svd = True

    def get_modified_singular_values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes modified singular values using both scaling/shifting and LoRA adaptation.
        Returns:
            Tuple containing:
                - Modified singular values tensor
                - Scale/shift modification term
                - LoRA adaptation term
        """
        # Create diagonal matrix of original singular values
        S_diag = torch.diag(self.S)
        
        # Apply scaling and shifting to top rank singular values
        top_s = self.S[:self.rank]
        modified_top_s = self.trainable_scale_A * top_s + self.trainable_shift_B
        
        # Compute LoRA term for remaining singular values
        loRA_term = self.trainable_X @ self.trainable_Y
        
        # Create the combined singular value matrix
        new_s = S_diag.clone()
        new_s[:self.rank, :self.rank] = torch.diag(modified_top_s)
        new_s[self.rank:, self.rank:] += loRA_term
        
        scale_shift_term = torch.zeros_like(S_diag)
        scale_shift_term[:self.rank, :self.rank] = torch.diag(modified_top_s) - torch.diag(top_s)
        
        return new_s, scale_shift_term, loRA_term

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with both scaling/shifting and LoRA modifications.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing:
                - Output tensor after convolution
                - Combined regularization loss
        """
        if not self.done_svd:
            self.perform_svd()

        new_s, scale_shift_term, LoRA_term = self.get_modified_singular_values()
        s_new = F.relu(new_s.to(x.device))

        # Reconstruct weight matrix using modified components
        weight_updated = self.U @ s_new @ self.Vt
        
        # Reshape weight back to conv2d format
        weight_updated = rearrange(
            weight_updated, 
            'co (cin h w) -> co cin h w', 
            cin=self.weight.size(1), 
            h=self.weight.size(2), 
            w=self.weight.size(3)
        )
        
        # Compute regularization losses
        scale_shift_reg = torch.norm(scale_shift_term)
        lora_reg = torch.norm(LoRA_term)
        reg_loss = scale_shift_reg + lora_reg

        return F.conv2d(
            x, weight_updated, self.bias, 
            self.stride, self.padding, 
            self.dilation, self.groups
        ), reg_loss