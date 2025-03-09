import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional

class DoRALinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 8,  # LoRA rank
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seed: int = 42
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        torch.manual_seed(seed)
        
        # Decompose weight into magnitude (m) and direction (V)
        with torch.no_grad():
            weight = self.weight.data
            m = torch.norm(weight, p=2, dim=1, keepdim=False)  # (out_features,)
            V = weight / m.unsqueeze(1)  # (out_features, in_features)
        
        self.register_buffer('V', V)
        self.register_buffer('m_initial', m.clone())
        self.trainable_m = nn.Parameter(m)
        
        # Initialize LoRA parameters
        self.r = r
        self.trainable_X = nn.Parameter(torch.randn(r, in_features, device=device, dtype=dtype) * 0.01)
        self.trainable_Y = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        
        # Freeze original weight
        self.weight.requires_grad = False
        
        # Parameter logging
        mag_params = out_features
        lora_params = r * (in_features + out_features)
        print(f"\nDoRALinear: {in_features}x{out_features}, r={r}")
        print(f"magintude params: {mag_params}, LoRA params: {lora_params}")
        print(f"Total trainable params: {mag_params + lora_params}")
        
        self._verify_parameters()
        
    def _verify_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable}, Total: {total}")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape} (Trainable: {param.requires_grad})")

    # For Linear layer
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_V = self.trainable_Y @ self.trainable_X  # (out, in)
        V_adapted = self.V + delta_V
        
        # Compute norms with gradient detachment
        with torch.no_grad():  # or use .detach()
            norms = torch.norm(V_adapted, p=2, dim=1, keepdim=True) + 1e-6
            
        norms.detach()
        
        # Normalize using detached norms
        V_normalized = V_adapted / norms
        
        # Compute adapted weight
        W_adapted = self.trainable_m.unsqueeze(1) * V_normalized
        
        # Regularization loss
        reg_loss = torch.norm(self.trainable_m - self.m_initial, p=2) + torch.norm(delta_V, p='fro')
        
        return F.linear(x, W_adapted, self.bias), 0


class DoRAConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 8,  # LoRA rank
        seed: int = 42,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        torch.manual_seed(seed)
        
        # Reshape weight to 2D and decompose
        with torch.no_grad():
            weight = self.weight.data
            weight_2d = weight.view(out_channels, -1)  # (out, in*h*w)
            m = torch.norm(weight_2d, p=2, dim=1)  # (out,)
            V = weight_2d / m.unsqueeze(1)
        
        self.register_buffer('V', V)
        self.register_buffer('m_initial', m.clone())
        self.trainable_m = nn.Parameter(m)
        
        # Initialize LoRA parameters
        self.r = r
        flat_size = weight_2d.size(1)
        self.trainable_X = nn.Parameter(torch.randn(r, flat_size, device=weight.device) * 0.01)
        self.trainable_Y = nn.Parameter(torch.zeros(out_channels, r, device=weight.device))
        
        self.weight.requires_grad = False
        
        # Parameter logging
        mag_params = out_channels
        lora_params = r * (flat_size + out_channels)
        print(f"\nDoRAConv2d: {in_channels}x{out_channels}, kernel={kernel_size}, r={r}")
        print(f"Mag params: {mag_params}, LoRA params: {lora_params}")
        print(f"Total trainable params: {mag_params + lora_params}")
        
        self._verify_parameters()
        
    def _verify_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable}, Total: {total}")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape} (Trainable: {param.requires_grad})")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_V = self.trainable_Y @ self.trainable_X  # (out, in*h*w)
        V_adapted = self.V + delta_V
        
        # Normalize rows and reshape
        with torch.no_grad():  # or use .detach()
            norms = torch.norm(V_adapted, p=2, dim=1, keepdim=True) + 1e-6
            
        norms.detach()
        # Normalize using detached norms
        V_normalized = V_adapted / norms
        W_adapted = (self.trainable_m.unsqueeze(1) * V_normalized).view(self.weight.shape)
        
        # Regularization loss
        reg_loss = torch.norm(self.trainable_m - self.m_initial, p=2) + torch.norm(delta_V, p='fro')
        
        return F.conv2d(
            x, W_adapted, self.bias, 
            self.stride, self.padding, 
            self.dilation, self.groups
        ), 0 # in DoRA they are not using regularization loss