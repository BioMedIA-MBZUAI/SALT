import torch
from model import *
import yaml

def load_pretrained_weights(custom_model, pretrained_path):
    # Load pretrained SAM2 state dict
    pretrained_dict = torch.load(pretrained_path)["model"]
    
    # Get custom model's state dict
    model_dict = custom_model.state_dict()
    
    # 1. Filter out unnecessary memory-related keys
    filtered_pretrained = {k: v for k, v in pretrained_dict.items() 
                          if not k.startswith(('memory_attention', 'memory_encoder', 'maskmem'))}

    
    # 3. Convert pretrained keys to custom model format
    aligned_dict = {}
    for pretrained_key, tensor in filtered_pretrained.items():
        # Handle key prefixes
        if 'image_encoder.trunk' in pretrained_key:
            new_key = pretrained_key.replace('model.', '')
        elif 'image_encoder.neck' in pretrained_key:
            new_key = pretrained_key.replace('model.', '')
            print(new_key)
        elif 'sam_prompt_encoder' in pretrained_key:
            new_key = pretrained_key.replace('sam_prompt_encoder', 'prompt_encoder')
        elif 'sam_mask_decoder' in pretrained_key:
            new_key = pretrained_key.replace('sam_mask_decoder', 'mask_decoder')
            if 'transformer.layers.1.mlp.layers.0' in new_key:
                new_key = new_key.replace("transformer.layers.1.mlp.layers.0", "transformer.layers.1.mlp.lin1")
            if 'transformer.layers.1.mlp.layers.1' in new_key:
                new_key = new_key.replace("transformer.layers.1.mlp.layers.1", "transformer.layers.1.mlp.lin2")
            if 'transformer.layers.0.mlp.layers.0' in new_key:
                new_key = new_key.replace("transformer.layers.0.mlp.layers.0", "transformer.layers.0.mlp.lin1")
            if 'transformer.layers.0.mlp.layers.1' in new_key:
                new_key = new_key.replace("transformer.layers.0.mlp.layers.1", "transformer.layers.0.mlp.lin2")     
        else:
            continue
            
        if new_key in model_dict:
            aligned_dict[new_key] = tensor
            
    # 4. Get matches between pretrained and custom model
    matched_dict = {k: v for k, v in aligned_dict.items() if k in model_dict}
    
    # 5. Load while preserving new text projection layer
    model_dict.update(matched_dict)
    custom_model.load_state_dict(model_dict, strict=False)
    
    # 6. Verify matches
    print(f"Loaded {len(matched_dict)}/{len(filtered_pretrained)} pretrained parameters")
    print(f"Missing keys: {[k for k in model_dict if k not in matched_dict]}")
    
    return custom_model


with open("/home/abdelrahman.elsayed/med-cvpr/AllinonSAM/model_svdtuning.yml", "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)



def save_model_state_dict_to_txt(model, filename):
    """
    Saves the model's state dict keys and tensor shapes to a text file.
    
    Args:
        model (nn.Module): The model whose state dict to save.
        filename (str): Path to the text file where to write the information.
    """
    state_dict = model.state_dict()
    with open(filename, "w") as f:
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                shape = list(tensor.shape)
            else:
                shape = type(tensor).__name__
            f.write(f"{key}: {shape}\n")
    print(f"Model state dict saved to {filename}")
    
model = Prompt_Adapted_SAM2(model_config)
custom_model = load_pretrained_weights(
    model, 
    "/home/abdelrahman.elsayed/sam2/checkpoints/sam2.1_hiera_large.pt"
)
save_model_state_dict_to_txt(custom_model, "custom_model_state_dict.txt")