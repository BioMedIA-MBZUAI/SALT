import argparse
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import *
from test import *
import pandas as pd
from train import *
import sys
import torch
import os

source_path = os.path.join("/home/abdelrahman.elsayed/SALT/datasets")
sys.path.append(source_path)
from data import GeneralDataset
# from crfseg import CRF
import itertools
from utils import CosineAnnealingWarmupScheduler

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_config",
        default="/home/abdelrahman.elsayed/SALT/config_data.yml",
        help="data config file path",
    )

    parser.add_argument(
        "--model_config",
        default="/home/abdelrahman.elsayed/SALT/SALT_config.yml",
        help="model config file path",
    )

    parser.add_argument("--pretrained_path", default=None, help="pretrained model path")

    parser.add_argument(
        "--save_path", default="checkpoints/temp.pth", help="pretrained model path"
    )
    parser.add_argument(
        "--training_strategy", default="salt", help="how to train the model"
    )

    parser.add_argument("--device", default="cuda:0", help="device to train on")

    args = parser.parse_args()

    return args


def main_onetime_functions(config):
    dataset_dict, dataset_sizes, label_dict = get_data(
        config,
        tr_folder_start=0,
        tr_folder_end=78000,
        val_folder_start=0,
        val_folder_end=104000,
        use_norm=False,
    )
    for x in dataset_dict:
        dataset_dict[x].one_time_generate_pos_neg_list_dicts(x)



def main_test(data_config, model_config, pretrained_path):
    test_start = 104
    test_end = 131
    test(
        data_config,
        model_config,
        pretrained_path,
        test_start,
        test_end,
        device="cuda:0",
    )


def lr_lambda(step):
    if step < model_config["training"]["warmup_steps"]:
        return step / model_config["training"]["warmup_steps"]  # Linear warm-up
    elif step < model_config["training"]["steps"][0]:
        return 1.0  # Maintain initial learning rate
    elif step < model_config["training"]["steps"][1]:
        return 1 / model_config["training"]["decay_factor"]  # First decay
    else:
        return 1 / (model_config["training"]["decay_factor"] ** 2)  # Second decay


def main_train(
    data_config,
    model_config,
    pretrained_path,
    save_path,
    training_strategy="biastuning",
    device="cuda:0",
):
    print(data_config)
    print(model_config)
    data_split_csv_path = data_config["data"]["data_split_csv"]
    data_split = pd.read_csv(data_split_csv_path)

    dataset_dict = {}
    dataloader_dict = {}

    use_norm = True
    no_text_mode = False

    for split in ["train", "val"]:
        split_data = data_split[data_split["split"] == split]["imgs"].tolist()

        dataset_dict[split] = GeneralDataset(
            config=data_config,
            file_list=split_data, 
            shuffle_list=True,
            is_train=(split == "train"),
            apply_norm=use_norm,
            no_text_mode=no_text_mode,
        )

        # Create DataLoader for each dataset
        dataloader_dict[split] = torch.utils.data.DataLoader(
            dataset_dict[split],
            batch_size=model_config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
        )

    # Get dataset sizes
    dataset_sizes = {split: len(dataset_dict[split]) for split in ["train", "val"]}

    # Create label dictionary
    label_dict = {
        name: i for i, name in enumerate(data_config["data"]["label_names"])
    }

    # Print dataset sizes
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Val dataset size: {dataset_sizes['val']}")

    # Get dataset sizes
    dataset_sizes = {split: len(dataset_dict[split]) for split in ["train", "val"]}

    # Create label dictionary
    label_dict = {
        name: i for i, name in enumerate(data_config["data"]["label_names"])
    }

    # Print dataset sizes
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Val dataset size: {dataset_sizes['val']}")
    # load model
    # change the img size in model config according to data config
    model_config["sam"]["img_size"] = data_config["data_transforms"]["img_size"]
    model_config["sam"]["num_classes"] = len(data_config["data"]["label_list"])
    if training_strategy == "lora":
        model_config["use_lora"] = True
    else:
        model_config["use_lora"] = False

    if training_strategy == "biastuning":
        model_config["decoder_training"] = "full"

    if model_config["arch"] == "Prompt Adapted SAM":
        model = Prompt_Adapted_SAM(
            model_config, label_dict, device, training_strategy=training_strategy
        )
    else:
        print(training_strategy)
        model = Prompt_Adapted_SAM2(
            model_config, label_dict, device, training_strategy=training_strategy
        )

    # load model weights
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))

    # freeze correct weights
    for p in model.parameters():
        # p.requires_grad=True
        p.requires_grad = False

    # unfreeze according to strategy:
    for name, p in model.named_parameters():
        if training_strategy == "salt":
            if "trainable" in name.lower():
                p.requires_grad = True
        elif training_strategy == "biastuning":
            if ("bias" in name.lower()) and ("clip" not in name.lower()):
                p.requires_grad = True
        elif training_strategy == "svdbiastuning":
            if "trainable" in name.lower():
                p.requires_grad = True
            if ("bias" in name.lower()) and ("clip" not in name.lower()):
                p.requires_grad = True
        elif training_strategy == "lora":
            if "trainable_lora" in name.lower():
                p.requires_grad = True

        if model_config["prompts"]["USE_TEXT_PROMPT"]:
            if "Text_Embedding_Affine" in name:
                p.requires_grad = True
        if model_config["prompts"]["USE_SLICE_NUM"]:
            if "slice" in name:
                p.requires_grad = True

        if model_config["decoder_training"] == "full":
            if ("decoder" in name.lower()) and ("clip" not in name.lower()):
                p.requires_grad = True
        elif model_config["decoder_training"] == "salt":
            if "trainable" in name.lower():
                p.requires_grad = True
        elif model_config["decoder_training"] == "none":
            if "decoder" in name.lower():
                p.requires_grad = False

        if "prompt_encoder" in name.lower():
            p.requires_grad = False
            # p.requires_grad = True

        # common parameters
        if "norm" in name.lower():
            p.requires_grad = True
        if "pos_embed" in name.lower():
            p.requires_grad = True
        if "clip" in name.lower():
            p.requires_grad = False

    # training parameters
    training_params = model_config["training"]
    if training_params["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(training_params["lr"]),
            weight_decay=float(training_params["weight_decay"]),
        )
    elif training_params["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(training_params["lr"]),
            weight_decay=float(training_params["weight_decay"]),
            momentum=0.9,
        )


    exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=training_params["schedule_step"],
                gamma=training_params["schedule_step_factor"],
        )
   
    criterion = []
    if "dice" in training_params["loss"]:
        criterion.append(dice_loss)
    if "focal" in training_params["loss"]:
        criterion.append(focal_loss)
    if "CE" in training_params["loss"]:
        criterion.append(nn.BCELoss())
    if "weighted CE" in training_params["loss"]:
        criterion.append(weighted_ce_loss)
    if criterion == []:
        criterion = [nn.BCELoss()]

    # retain_graph = False if model_config['decoder_training']=='none' else True
    retain_graph = False

    # train the model
    save_path = "./models" + data_config["data"]["root_path"].split("/")[-1]
    model = train_dl(
        model,
        dataset_dict,
        dataset_sizes,
        criterion,
        optimizer,
        exp_lr_scheduler,
        save_path,
        save_dir=f"./{args.training_strategy}/{model_config['ft']['type']}/{data_config['data']['root_path'].split('/')[-1]}",
        num_epochs=training_params["num_epochs"],
        bs=2, #DB
        device=device,
        retain_graph=retain_graph,
        neg2pos_ratio=data_config["data"]["negative_to_positive_ratio"],
        reg_multiplier=model_config["training"]["reg_multiplier"],
    )
        

if __name__ == "__main__":
    args = parse_args()
    with open(args.data_config, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    main_train(
        data_config,
        model_config,
        args.pretrained_path,
        args.save_path,
        args.training_strategy,
        device=args.device,
    )
