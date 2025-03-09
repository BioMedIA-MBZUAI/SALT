import torch
import sys

def get_key_shapes(state_dict, prefix=""):
    """
    Recursively traverse a state_dict and collect the shape information
    for all tensor values.
    
    Args:
        state_dict (dict): Dictionary loaded from a checkpoint.
        prefix (str): A prefix for nested keys.
    
    Returns:
        dict: A dictionary where keys are the full keys (with prefix) and 
              values are the shape tuple or type name for non-tensor values.
    """
    key_shapes = {}
    for key, value in state_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, torch.Tensor):
            key_shapes[full_key] = tuple(value.shape)
        elif isinstance(value, dict):
            key_shapes.update(get_key_shapes(value, full_key))
        else:
            key_shapes[full_key] = type(value).__name__
    return key_shapes

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_key_shapes.py <checkpoint_path> <output_txt_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_txt_path = sys.argv[2]

    # Load the checkpoint to CPU for compatibility
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract shapes of all keys in the checkpoint
    key_shapes = get_key_shapes(checkpoint)
    
    # Write the keys and their shapes to a text file
    with open(output_txt_path, "w") as f:
        for key, shape in key_shapes.items():
            f.write(f"{key}: {shape}\n")

    print(f"Key shapes written to {output_txt_path}")
