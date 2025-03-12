# SALT: Parameter-Efficient Fine-Tuning via Singular Value Adaptation with Low-Rank Transformation

SALT is a novel Parameter-Efficient Fine-Tuning (PEFT) approach designed to adapt foundation models effectively for precise medical image segmentation.

## Overview
SALT combines the strengths of Singular Value Decomposition (SVD)-based adaptation and Low-Rank Adaptation (LoRA). It selectively updates dominant singular values with trainable scale and shift parameters and applies low-rank updates to the residual singular subspace, resulting in robust and efficient fine-tuning.

## Key Highlights
- **Hybrid PEFT Approach:** Combines strengths of LoRA and SVD.
- **Superior Performance:** Outperforms existing PEFT methods by 2-5% Dice improvement.
- **Minimal Parameter Overhead:** Achieves state-of-the-art performance with only 3.9% trainable parameters.

## Evaluated Datasets
- **ARCADE:** Coronary artery segmentation from X-ray angiography
- **DIAS:** Intracranial artery segmentation in Dynamic Digital Subtraction Angiography (DSA)
- **ROSE:** Retinal OCT angiography segmentation
- **XRay-Angio:** Multiscale segmentation of occluded vessels
- **DRIVE:** Retinal vessel segmentation in RGB fundus images

## Results Summary

| Method | Avg. Dice Score | Avg. HD95 | Trainable Parameters (%) |
|--------|-----------------|-------------|--------------------------|
| LoRA   | 0.70            | 25.94       | 14.08%                  |
| S-SAM  | 0.71            | 30.12       | 0.40%                    |
| **SALT (ours)** | **0.74**    | **23.87**   | **3.90%**              |

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/SALT.git
cd SALT
```

Create SALT Environment:

```bash
conda env create -f SALT_env.yml
```
## Datasets access
You can download each dataset from Hugging Face:

- 🟢 **ROSE:** [Download here](https://huggingface.co/datasets/pythn/ROSE)
- 🔵 **ARCADE:** [Download here](https://huggingface.co/datasets/pythn/ARCADE)
- 🟠 **drive:** [Download here](https://huggingface.co/datasets/pythn/drive)
- 🟣 **DIAS:** [Download here](https://huggingface.co/datasets/pythn/DIAS)
- 🔴 **Xray-Angio:** [Download here](https://huggingface.co/datasets/pythn/DB)

Once downloaded, the dataset should have the following structure:

dataset_name/  
  │── images/        # Image files  
  │── masks/         # Corresponding masks  
  │── data_split.csv # Train/Val/Test splits  

## Usage

### Training



## Repository Structure



## Citation


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

