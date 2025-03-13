# SALT: Parameter-Efficient Fine-Tuning via Singular Value Adaptation with Low-Rank Transformation

SALT is a novel Parameter-Efficient Fine-Tuning (PEFT) method designed to adapt large-scale foundation models—especially **Segment Anything Model (SAM)**—to domain-specific tasks such as **medical image segmentation**. With SALT, you can achieve **state-of-the-art** segmentation performance while training **only a small percentage of parameters**.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Datasets](#datasets)
4. [Repository Structure](#repository-structure)
5. [Usage](#usage)
   - [1. Configuration Files](#1-configuration-files)
   - [2. Training](#2-training)
6. [Command Examples](#command-examples)
7. [Performance Highlights](#performance-highlights)
8. [Citations](#citations)
9. [License](#license)

---

## Key Features

- :star2: **Hybrid PEFT**  
  Merges SVD-based parameter updates for top singular values with LoRA for the residual subspace.
- :chart_with_upwards_trend: **Parameter Efficiency**  
  Yields SOTA performance with as little as **3.9%** of total parameters being trainable.
- :microscope: **Robust Medical Adaptation**  
  Outperforms other PEFT methods (LoRA, S-SAM) by **2–5%** in Dice scores across challenging medical datasets.
- :wrench: **Easy Integration**  
  Provided as a drop-in module for existing SAM pipelines, requiring minimal code changes.

---

## Installation

1. **Clone this repository** :cloning:  
   ```bash
   git clone https://github.com/YourUsername/SALT.git
   cd SALT
   ```
   
2. **Set up a Conda environment** :snake:  
   ```bash
   conda env create -f SALT_env.yml
   conda activate SALT
   ```
   > **Note:** Verify or modify the packages in `SALT_env.yml` as required (e.g., PyTorch version, CUDA, etc.).
---

## Datasets

Below are the five medical imaging datasets highlighted in our experiments. Each dataset can be found on Hugging Face:

- 🟢: **[ROSE](https://huggingface.co/datasets/pythn/ROSE)** (Retinal OCT Angiography)  
- 🔵: **[ARCADE](https://huggingface.co/datasets/pythn/ARCADE)** (Coronary Artery Segmentation)  
- 🟠: **[DRIVE](https://huggingface.co/datasets/pythn/drive)** (Retinal Vessel Segmentation)  
- 🟡: **[DIAS](https://huggingface.co/datasets/pythn/DIAS)** (Dynamic Digital Subtraction Angiography)  
- 🔴: **[Xray-Angio](https://huggingface.co/datasets/pythn/DB)** (Occluded Vessel Segmentation)

Please organize each dataset as follows:

```
dataset_name/
├── images/               # Folder containing image files
├── masks/                # Folder containing segmentation masks
└── data_split.csv        # CSV specifying [imgs, split] columns for train/val/test
```

> **Tip:** Make sure the file paths in `data_split.csv` correspond exactly to those in the `images/` and `masks/` folders.

---

## Repository Structure

Here's a simplified overview of the repository and the purpose of each major file:

```
SALT/
├── main.py                # Entry point for training/testing SALT
├── model.py               # Model definitions (Prompt_Adapted_SAM, Prompt_Adapted_SAM2)
├── train.py               # Core training loop and helper functions
├── test.py                # Evaluation and metrics computation
├── datasets/
│   └── data.py            # Custom Dataset (GeneralDataset) and data loader logic
├── utils.py          # Dice, Focal, and other loss functions
├── data_transforms/
│   ├── data_transforms.py # Contains Data_Transform class
├── prompt_adapted_SAM2/
│   ├── .... # Contains the implementation for Prompt Adapted SAM 2 along with LoRA , SVD , SALT layers.
├── prompt_adapted_segment_anything/
│   ├── .... # Contains the implementation for Prompt Adapted SAM along with LoRA , SVD , SALT layers.
├── SALT_env.yml
└── config_data.yml # Contains example data set configuration
└── SALT_config.yml # Contains example model and training configuration
└── ... # Other complementary files for doing model analysis, submitting jobs,...etc
```

- **`main.py`**  
  Parses command-line arguments, loads datasets, sets up models, and invokes training or testing.

- **`model.py`**  
  Houses the `Prompt_Adapted_SAM` and `Prompt_Adapted_SAM2` classes, plus SALT/LoRA logic.

- **`train.py`**  
  Contains the main training function, optimization loops, checkpoint saving, etc.

- **`test.py`**  
  Handles inference and computing evaluation metrics (e.g., Dice, HD95).

- **`data/data.py`**  
  Implements the `GeneralDataset` for image-mask pair loading and augmentation.

- **`utils/losses.py`**  
  Provides a variety of segmentation loss functions (Dice, BCE, Focal, WeightedCE).

---

## Usage

### 1. Configuration Files

You’ll need two primary YAML files:

1. **Data Config (`config_data.yaml`)**  
   Defines your dataset root path, transforms, image size, label definitions, etc. Example:
   ```yaml
   data_transforms:
     a_min: 0
     a_max: 255
     img_size: 512
     ...
   data:
     root_path: '/path/to/ROSE'
     data_split_csv: '/path/to/ROSE/data_split.csv'
     label_list: [0, 1]
     label_names: ["Background", "Vein"]
     ...
   ```

2. **Model Config (`SALT_config.yaml`)**  
   Specifies model settings, optimizer, LoRA/SALT ranks, etc. Example:
   ```yaml
   sam:
     img_size: 512
     num_classes: 2
     sam_type: "base"

   training:
     optimizer: 'adamw'
     lr: 1e-4
     batch_size: 8
     num_epochs: 200
     ...
   ft:
     type: 'svd'    # Could be 'svd', 'lora', or 'salt'
     svd_rank_linear: 0
     svd_rank_conv2d: 0
     r_lora: 4
   ```

### 2. Training

Use the script `main.py` to train. Key flags include:
- `--data_config`: path to YAML for data
- `--model_config`: path to YAML for model and training hyperparameters
- `--pretrained_path`: path to your SAM checkpoint (if applicable)
- `--save_path`: where to store trained weights
- `--training_strategy`: one of `[salt, svdbiastuning, biastuning, lora]`

For example:
```bash
python main.py \
  --data_config configs/config_data.yaml \
  --model_config configs/SALT_config.yaml \
  --pretrained_path /path/to/sam_checkpoint.pth \
  --save_path checkpoints/salt_model.pth \
  --training_strategy salt \
  --device cuda:0
```
> **Note:** You may need to adjust `device` based on your system (e.g., `"cuda:1"` or `"cpu"`).

---

## Command Examples

1. **Training with SALT** :sparkles:
   ```bash
   python main.py \
     --data_config configs/config_data.yaml \
     --model_config configs/SALT_config.yaml \
     --pretrained_path /path/to/sam_checkpoint.pth \
     --save_path checkpoints/salt_model.pth \
     --training_strategy salt \
     --device cuda:0
   ```
2. **Training with LoRA** :tada:
   ```bash
   python main.py \
     --data_config configs/config_data.yaml \
     --model_config configs/SALT_config.yaml \
     --pretrained_path /path/to/sam_checkpoint.pth \
     --save_path checkpoints/lora_model.pth \
     --training_strategy lora \
     --device cuda:0
   ```

---

## Performance Highlights

Here’s a brief summary of SALT’s performance versus other PEFT methods:

| **Method**      | **Avg. Dice** | **Avg. HD95** | **Trainable Params** |
|-----------------|---------------|---------------|----------------------|
| LoRA (rank=256) | 0.70          | 25.94         | 14.08%              |
| S-SAM           | 0.71          | 30.12         | 0.40%               |
| **SALT (Ours)** | **0.74**      | **23.87**     | **3.90%**           |

🔥: **SALT** provides the best Dice and HD95 among these PEFT methods, striking a strong balance between accuracy and parameter efficiency.

---

## Citations

Coming Soon...

---

## License

This project is released under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software; see the LICENSE file for details.
