

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from prompt_adapted_SAM2.modeling.Adapters.LoRA import LoRALinear
from prompt_adapted_SAM2.modeling.Adapters.SVD import SVDLinear
from prompt_adapted_SAM2.modeling.Adapters.SALT import SALTLinear

def analyze_model_singular_values(model, model_name="SAM2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Extract the Hiera trunk from the image encoder
    hiera_trunk = model.image_encoder.trunk
    
    layer_analyses = {"global": {}, "local": {}}

    # Iterate through Hiera blocks to classify global vs. local layers
    for block_idx, block in enumerate(hiera_trunk.blocks):
        # Check if the block uses global attention (window_size=0)
        is_global = block.window_size == 0
        
        # Extract linear layers from MultiScaleAttention and MLP in the block
        for name, module in block.named_modules():
            if isinstance(module, (torch.nn.Linear, LoRALinear, SVDLinear, SALTLinear)):
                layer_type = "global" if is_global else "local"
                layer_key = f"block_{block_idx}_{name}"
                
                # Perform SVD on the weight matrix
                W = module.weight.detach().float().cpu().numpy()
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                
                # Store results
                layer_analyses[layer_type][layer_key] = {
                    "singular_values": S / S.max(),
                    "cumulative_energy": np.cumsum(S**2) / np.sum(S**2),
                    "effective_rank": np.sum(S / S.max() > 0.01),
                    "shape": W.shape
                }

    return layer_analyses


def visualize_analysis(layer_analyses, model_name):
    plt.figure(figsize=(20, 12))
    
    # Plot for Global Attention Layers
    plt.subplot(2, 3, 1)
    for layer_name, data in layer_analyses["global"].items():
        plt.semilogy(data['singular_values'], alpha=0.5)
    plt.title(f"{model_name} - Global Attention: Singular Values")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Magnitude (log scale)")
    
    plt.subplot(2, 3, 2)
    for layer_name, data in layer_analyses["global"].items():
        plt.plot(data['cumulative_energy'], alpha=0.5)
    plt.title(f"{model_name} - Global Attention: Cumulative Energy")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Fraction of Total Energy")
    
    plt.subplot(2, 3, 3)
    effective_ranks = [data['effective_rank'] for data in layer_analyses["global"].values()]
    plt.hist(effective_ranks, bins=20)
    plt.title(f"{model_name} - Global Attention: Effective Rank")
    plt.xlabel("Effective Rank (σ > 1% of σ_max)")
    plt.ylabel("Count")
    
    # Plot for Local Attention Layers
    plt.subplot(2, 3, 4)
    for layer_name, data in layer_analyses["local"].items():
        plt.semilogy(data['singular_values'], alpha=0.5)
    plt.title(f"{model_name} - Local Attention: Singular Values")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Magnitude (log scale)")
    
    plt.subplot(2, 3, 5)
    for layer_name, data in layer_analyses["local"].items():
        plt.plot(data['cumulative_energy'], alpha=0.5)
    plt.title(f"{model_name} - Local Attention: Cumulative Energy")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Fraction of Total Energy")
    
    plt.subplot(2, 3, 6)
    effective_ranks = [data['effective_rank'] for data in layer_analyses["local"].values()]
    plt.hist(effective_ranks, bins=20)
    plt.title(f"{model_name} - Local Attention: Effective Rank")
    plt.xlabel("Effective Rank (σ > 1% of σ_max)")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(f"SVD analysis_{model_name}.png")