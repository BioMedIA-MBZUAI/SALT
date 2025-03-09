import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from typing import Type, Tuple, Optional

class SALTLinear(nn.Linear):
    """
    A linear layer that combines truncated SVD decomposition with LoRA-style adaptation.
    Only keeps top r singular values and vectors, then adds LoRA adaptation.
    """
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
        
        max_possible_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0])
        print("\nThe max possible rank is", max_possible_rank)

        # Truncation rank for SVD
        self.rank = rank
        
        # Initialize LoRA matrices
        self.X = nn.Parameter(torch.randn(max_possible_rank, r_lora) * 0.01)
        self.Y = nn.Parameter(torch.randn(r_lora, max_possible_rank) * 0.01)

        self.reset_parameters()

    def _initialize_svd(self):
        """Initializes SVD decomposition on the weight matrix."""
        return torch.linalg.svd(self.weight, full_matrices=False)

    def perform_svd(self) -> None:
        """Updates truncated SVD decomposition on the weight matrix."""
        self.U, self.S, self.Vt = self._initialize_svd()
        self.done_svd = True

    def get_modified_singular_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes modified singular values using LoRA adaptation.
        Returns:
            Tuple containing:
                - Modified singular values tensor
                - LoRA adaptation term
        """
         # Compute the LoRA adaptation term
        loRA_term = self.X @ self.Y

        # Create a mask that matches the shape of loRA_term
        mask = torch.ones_like(loRA_term, device=self.X.device)
        # Example: Set the first `rank` rows of the mask to 0
        mask[:self.rank, :] = 0  # Adjust as needed

        # Apply mask to LoRA term
        masked_loRA_term = loRA_term * mask

        # Compute the modified singular values
        new_s = torch.diag(self.S) + masked_loRA_term
        return new_s, masked_loRA_term

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LoRA-modified truncated singular values.
        
        Args:
            input: Input tensor
            
        Returns:
            Tuple containing:
                - Output tensor after linear transformation
                - Regularization loss
        """
        if not self.done_svd:
            self.perform_svd()

        new_s, LoRA_term = self.get_modified_singular_values()
        s_new = F.relu(new_s.to(input.device))

        # Reconstruct weight matrix using truncated components
        weight_updated = self.U @ s_new @ self.Vt
        
        # Compute regularization loss
        reg_loss = torch.norm(LoRA_term)

        return F.linear(input, weight_updated, self.bias), reg_loss


class SALTConv2d(nn.Conv2d):
    """
    A 2D convolutional layer that combines truncated SVD decomposition with LoRA-style adaptation.
    The weight matrix is reshaped before applying truncated SVD and LoRA modifications.
    """
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

        self.rank = rank
        
        # Initialize LoRA matrices
        self.X = nn.Parameter(torch.randn(max_possible_rank, r_lora) * 0.01)
        self.Y = nn.Parameter(torch.randn(r_lora, max_possible_rank) * 0.01)

        self.reset_parameters()

    def _initialize_svd(self, weight_reshaped):
        """Initializes SVD decomposition on the reshaped weight matrix."""
        return torch.linalg.svd(weight_reshaped, full_matrices=False)

    def perform_svd(self) -> None:
        """Updates truncated SVD decomposition on the reshaped weight matrix."""
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = self._initialize_svd(weight_reshaped)
        self.done_svd = True

    def get_modified_singular_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes modified singular values using LoRA adaptation.
        Returns:
            Tuple containing:
                - Modified singular values tensor
                - LoRA adaptation term
        """
         # Compute the LoRA adaptation term
        loRA_term = self.X @ self.Y

        # Create a mask that matches the shape of loRA_term
        mask = torch.ones_like(loRA_term, device=self.X.device)
        # Example: Set the first `rank` rows of the mask to 0
        mask[:self.rank, :] = 0  # Adjust as needed

        # Apply mask to LoRA term
        masked_loRA_term = loRA_term * mask

        # Compute the modified singular values
        new_s = torch.diag(self.S) + masked_loRA_term
        return new_s, masked_loRA_term

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LoRA-modified truncated singular values.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing:
                - Output tensor after convolution
                - Regularization loss
        """
        if not self.done_svd:
            self.perform_svd()

        new_s, LoRA_term = self.get_modified_singular_values()
        s_new = F.relu(new_s.to(x.device))

        # Reconstruct weight matrix using truncated components
        weight_updated = self.U @ s_new @ self.Vt
        
        # Reshape weight back to conv2d format
        weight_updated = rearrange(
            weight_updated, 
            'co (cin h w) -> co cin h w', 
            cin=self.weight.size(1), 
            h=self.weight.size(2), 
            w=self.weight.size(3)
        )
        
        # Compute regularization loss
        reg_loss = torch.norm(LoRA_term)

        return F.conv2d(
            x, weight_updated, self.bias, 
            self.stride, self.padding, 
            self.dilation, self.groups
        ), reg_loss
