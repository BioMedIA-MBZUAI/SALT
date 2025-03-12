import torch

import sys
import copy
import os

from model import *
from utils import *
import yaml
from tqdm import tqdm
import wandb
from analyze import analyze_model_singular_values, visualize_analysis
import torch
import wandb
from PIL import Image
import numpy as np
from utils import (
    running_stats,
    dice_collated,
    compute_hd95,
    fractal_dimension,
    iou_coef,
    average_closest_distance,
)

from pathlib import Path

def print_model_parameters_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")

    # Print parameters by module
    print("\nParameters by module:")
    for name, module in model.named_children():
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params
        print(
            "*************************************************************************************************************"
        )
        print(f"  {name}:")
        print(f"    Total: {total_params:,}")
        print(f"    Trainable: {trainable_params:,}")
        print(f"    Frozen: {frozen_params:,}")
        print(
            "*******************************************************************************************"
        )

# Load configuration from data_config.yml
with open(
    "/home/abdelrahman.elsayed/med-cvpr/AllinonSAM/config_data.yml", "r"
) as data_config_file:
    data_config = yaml.safe_load(data_config_file)

# Load configuration from model_svdtuning.yml
with open(
    "/home/abdelrahman.elsayed/med-cvpr/AllinonSAM/SALT_config.yml", "r"
) as model_config_file:
    model_config = yaml.safe_load(model_config_file)


def train_dl(
    model,
    datasets,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    sav_path="./checkpoints/temp.pth",
    num_epochs=25,
    bs=32,
    device="cuda:0",
    retain_graph=False,
    neg2pos_ratio=-1,
    save_dir="./validation_images",
    reg_multiplier=0.01,
):
    torch.cuda.empty_cache()
    model = model.to(device)
    best_dice = 0
    best_loss = 10000
    best_hd95 = 1000000
    print_model_parameters_stats(model)

    run_name = f"{data_config['data']['root_path'].split('/')[-1]}_model_{model_config['arch']}_ft_{model_config['ft']['type']}_svd_{model_config['ft']['svd_rank_linear']}_svd_conv_{model_config['ft']['svd_rank_conv2d']}_loRA_{model_config['ft']['r_lora']}"
    
    wandb.init(
        project="Ablation SAM2",
        name=run_name,
        config={
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": bs,
            "num_epochs": num_epochs,
            "reg_multiplier": reg_multiplier,
        },
    )

    print("Training parameters: \n----------")
    print(
        "number of trainable parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("batch size: ", bs)
    print("num epochs: ", num_epochs)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        dataloaders = {}

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                if neg2pos_ratio > 0:
                    datasets[phase].generate_examples(neg2pos_ratio)
            else:
                model.eval()

            running_loss = 0.0
            running_dice = 0
            running_hd95 = 0.0
            count = 0
            dataloaders[phase] = torch.utils.data.DataLoader(
                datasets[phase], batch_size=bs, shuffle=True, num_workers=2, pin_memory=True
            )

            pbar = tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch}", leave=False)

            for batch_idx, (inputs, labels, text_idxs, text) in enumerate(pbar):
                count += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs, reg_loss = model(inputs, text) 
                    if len(outputs.shape) == 4:
                        outputs = torch.squeeze(outputs, dim=1)
                    loss = 0
                    seg_loss = 0
                    for c in criterion:
                        if "text" in c.__code__.co_varnames:
                            seg_loss += c(outputs, text, labels.float())
                        else:
                            seg_loss += c(outputs, labels.float())
                    loss += seg_loss
                    loss += reg_loss * reg_multiplier

                    if phase == "train":
                        loss.backward(retain_graph=True)
                        optimizer.step()

                with torch.no_grad():
                    preds = outputs >= 0.5
                    running_loss += loss.item() * inputs.size(0)
                    ri, ru = running_stats(labels, preds)
                    running_dice += dice_collated(ri, ru)

                    # Always compute HD95 for validation
                    if phase == "val":
                        hd95 = compute_hd95(preds, labels)
                        running_hd95 += hd95.item() * inputs.size(0)

                pbar.set_postfix({
                    "loss": loss.item(),
                    "dice": running_dice / count,
                    "hd95": running_hd95 / count if phase == "val" else "N/A"
                })

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_dice = running_dice / dataset_sizes[phase]
            
            # Handle validation metrics
            if phase == "val":
                epoch_hd95 = running_hd95 / dataset_sizes[phase]
                # Update best HD95
                if epoch_hd95 < best_hd95:
                    best_hd95 = epoch_hd95
                    wandb.run.summary["best_val_hd95"] = best_hd95

            print(f"{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}", 
                  f"HD95: {epoch_hd95:.4f}" if phase == "val" else "")

            # Log metrics to wandb
            log_data = {
                f"{phase}_loss": epoch_loss,
                f"{phase}_dice": epoch_dice,
                "epoch": epoch
            }
            if phase == "val":
                log_data[f"{phase}_hd95"] = epoch_hd95
            wandb.log(log_data)

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_dice = epoch_dice
                # torch.save(model.state_dict(), sav_path)
                wandb.run.summary.update({
                    "best_val_loss": best_loss,
                    "best_val_dice": best_dice
                })

    print(f"Best val loss: {best_loss:.4f}, best val dice: {best_dice:.4f}, best Hd95: {best_hd95:.4f}")
    model_save_path = f"{sav_path}/final_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    wandb.finish()
    return model
