import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_model_singular_values(model, model_name="SAM2"):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Dictionary to store layer analyses
    layer_analyses = {}

    # Collect all linear layers
    linear_layers = [module for module in model.modules() 
                    if isinstance(module, torch.nn.Linear)]

    for i, layer in enumerate(tqdm(linear_layers, desc="Analyzing layers")):
        # Convert weights to float32 for SVD compatibility
        W = layer.weight.detach().float().cpu().numpy()  # Added .float() here
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        
        # Convert back to original dtype if needed
        if layer.weight.dtype == torch.float16:
            S = S.astype(np.float16)
        
        # Rest of the analysis remains the same
        normalized_S = S / S.max()
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        
        layer_analyses[f"layer_{i}"] = {
            "shape": W.shape,
            "singular_values": normalized_S,
            "cumulative_energy": cumulative_energy,
            "effective_rank": np.sum(S / S.max() > 0.01)
        }

    return layer_analyses

def visualize_analysis(layer_analyses, model_name):
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Singular Value Spectra
    plt.subplot(2, 2, 1)
    for layer_name, data in layer_analyses.items():
        plt.semilogy(data['singular_values'], alpha=0.5)
    plt.title(f"{model_name} - Normalized Singular Values")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Magnitude (log scale)")
    
    # Plot 2: Cumulative Energy
    plt.subplot(2, 2, 2)
    for layer_name, data in layer_analyses.items():
        plt.plot(data['cumulative_energy'], alpha=0.5)
    plt.title(f"{model_name} - Cumulative Energy")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Fraction of Total Energy")
    
    # Plot 3: Effective Rank Distribution
    plt.subplot(2, 2, 3)
    effective_ranks = [data['effective_rank'] for data in layer_analyses.values()]
    plt.hist(effective_ranks, bins=20)
    plt.title(f"{model_name} - Effective Rank Distribution")
    plt.xlabel("Effective Rank (σ > 1% of σ_max)")
    plt.ylabel("Count")
    
    # Plot 4: Layer Dimensionality
    plt.subplot(2, 2, 4)
    in_dims = [data['shape'][1] for data in layer_analyses.values()]
    out_dims = [data['shape'][0] for data in layer_analyses.values()]
    plt.scatter(in_dims, out_dims, alpha=0.5)
    plt.title(f"{model_name} - Layer Dimensions")
    plt.xlabel("Input Features")
    plt.ylabel("Output Features")
    
    plt.tight_layout()
    plt.savefig("SVD analysis.png")
