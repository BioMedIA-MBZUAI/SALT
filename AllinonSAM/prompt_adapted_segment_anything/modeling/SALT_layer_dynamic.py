import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from typing import Tuple, Optional

class SALTLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r_lora: int = 8,  
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seed: int = 42
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        torch.manual_seed(seed)

        # 1) Freeze the original weight
        self.weight.requires_grad = False
        self.done_svd = False
        
        # Initialize a step counter for logging
        self.step_counter = 0

        # 2) Compute the *full* SVD
        U, S, Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.full_rank = S.shape[0]

        # Store SVD as buffers
        self.register_buffer("U", U)
        self.register_buffer("S", S)
        self.register_buffer("Vt", Vt)

        # 3) Dynamic rank threshold
        self.trainable_rank_threshold = nn.Parameter(torch.tensor(0.2))
        self.current_rank = None

        # 4) Create trainable adapters
        self._init_adaptation_parameters(r_lora)

    def _init_adaptation_parameters(self, r_lora: int):
        self.trainable_scale_A = nn.Parameter(torch.ones(self.full_rank))
        self.trainable_shift_B = nn.Parameter(torch.zeros(self.full_rank))
        self.trainable_X = nn.Parameter(torch.randn(self.full_rank, r_lora) * 0.01)
        self.trainable_Y = nn.Parameter(torch.randn(r_lora, self.full_rank) * 0.01)

    def _compute_dynamic_rank(self) -> int:
        s_norm = self.S / self.S.max()
        thresh = torch.sigmoid(self.trainable_rank_threshold)
        rank = torch.sum(s_norm > thresh).item()
        rank = max(1, min(rank, self.full_rank))
        return rank

    def _get_modified_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.current_rank = self._compute_dynamic_rank()
        remaining_rank = self.full_rank - self.current_rank

        S_diag = torch.diag(self.S.clone())

        # Scale & shift top portion
        top_s = self.S[:self.current_rank]
        scaled = self.trainable_scale_A[:self.current_rank] * top_s
        shifted = scaled + self.trainable_shift_B[:self.current_rank]
        S_diag[:self.current_rank, :self.current_rank] = torch.diag(shifted)

        # LoRA on leftover
        if remaining_rank > 0:
            X_block = self.trainable_X[self.current_rank:, :]
            Y_block = self.trainable_Y[:, self.current_rank:]
            lora_term = X_block @ Y_block
            S_diag[self.current_rank:, self.current_rank:] += lora_term
        else:
            lora_term = torch.zeros(0, device=S_diag.device)

        return S_diag, lora_term

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Optionally re-SVD only once
        if not self.done_svd:
            self.done_svd = True

        # Increment step counter
        self.step_counter += 1

        # Get updated singular values
        modified_S, lora_term = self._get_modified_components()

        # Reconstruct weight
        weight = self.U @ F.relu(modified_S) @ self.Vt

        # Log the rank every 50 steps
        if self.step_counter % 50 == 0:
            print(f"[SALTLinear] Step {self.step_counter}: current_rank = {self.current_rank}/{self.full_rank}")

        # Regularization
        reg_loss = (
            torch.norm(self.trainable_scale_A, p=2) +
            torch.norm(self.trainable_shift_B, p=2) +
            torch.norm(lora_term, p=2) +
            0.1 * torch.sigmoid(self.trainable_rank_threshold)
        )

        return F.linear(x, weight, self.bias), reg_loss
    
    
class SALTConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        r_lora: int = 8, 
        bias: bool = True,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, bias=bias, **kwargs)
        torch.manual_seed(seed)

        # 1) Freeze original weight
        self.weight.requires_grad = False
        self.done_svd = False

        # Step counter for logging
        self.step_counter = 0

        # 2) Reshape and do full SVD
        weight_2d = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        U, S, Vt = torch.linalg.svd(weight_2d, full_matrices=False)
        self.full_rank = S.shape[0]

        # Store SVD as buffers
        self.register_buffer("U", U)
        self.register_buffer("S", S)
        self.register_buffer("Vt", Vt)

        # 3) Dynamic threshold param
        self.trainable_rank_threshold = nn.Parameter(torch.tensor(0.0))
        self.current_rank = None

        # 4) Create adapter parameters
        self._init_adaptation_parameters(r_lora)

    def _init_adaptation_parameters(self, r_lora: int):
        self.trainable_scale_A = nn.Parameter(torch.ones(self.full_rank))
        self.trainable_shift_B = nn.Parameter(torch.zeros(self.full_rank))
        self.trainable_X = nn.Parameter(torch.randn(self.full_rank, r_lora) * 0.01)
        self.trainable_Y = nn.Parameter(torch.randn(r_lora, self.full_rank) * 0.01)

    def _compute_dynamic_rank(self) -> int:
        s_norm = self.S / self.S.max()
        thresh = torch.sigmoid(self.trainable_rank_threshold)
        rank = torch.sum(s_norm > thresh).item()
        return max(1, min(rank, self.full_rank))

    def _get_modified_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.current_rank = self._compute_dynamic_rank()
        remaining_rank = self.full_rank - self.current_rank

        S_diag = torch.diag(self.S.clone())

        # Scale+shift top portion
        top_s = self.S[:self.current_rank]
        scaled = self.trainable_scale_A[:self.current_rank] * top_s
        shifted = scaled + self.trainable_shift_B[:self.current_rank]
        S_diag[:self.current_rank, :self.current_rank] = torch.diag(shifted)

        # LoRA leftover
        if remaining_rank > 0:
            X_block = self.trainable_X[self.current_rank:, :]
            Y_block = self.trainable_Y[:, self.current_rank:]
            lora_term = X_block @ Y_block
            S_diag[self.current_rank:, self.current_rank:] += lora_term
        else:
            lora_term = torch.zeros(0, device=S_diag.device)

        return S_diag, lora_term

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.done_svd:
            self.done_svd = True

        # Increment step counter
        self.step_counter += 1

        modified_S, lora_term = self._get_modified_components()

        weight_2d = self.U @ F.relu(modified_S) @ self.Vt

        # Reshape to conv
        weight_updated = rearrange(
            weight_2d, 
            'co (cin h w) -> co cin h w', 
            cin=self.weight.size(1), 
            h=self.weight.size(2), 
            w=self.weight.size(3)
        )

        # Optional logging every 50 steps
        if self.step_counter % 50 == 0:
            print(f"[SALTConv2d] Step {self.step_counter}: current_rank = {self.current_rank}/{self.full_rank}")

        reg_loss = (
            torch.norm(self.trainable_scale_A, p=2) +
            torch.norm(self.trainable_shift_B, p=2) +
            torch.norm(lora_term, p=2) +
            0.1 * torch.sigmoid(self.trainable_rank_threshold)
        )

        return F.conv2d(
            x, weight_updated, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        ), reg_loss
