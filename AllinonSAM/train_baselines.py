import argparse
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_utils import *
from model import *
from test import *
from train import *
from baselines import UNet, UNext, medt_net
from vit_seg_modeling import VisionTransformer
import segmentation_models_pytorch as smp
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from axialnet import MedT
import sys
source_path = os.path.join("/home/abdelrahman.elsayed/sarim_code/datasets")
sys.path.append(source_path)
from arcade import ArcadeDataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_config', default='/home/abdelrahman.elsayed/med-cvpr/AllinonSAM/config_arcade.yml',
                        help='data config file path')

    parser.add_argument('--model_config', default='//home/abdelrahman.elsayed/med-cvpr/AllinonSAM/model_baseline.yml',
                        help='model config file path')

    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')

    parser.add_argument('--save_path', default='checkpoints/temp.pth',
                        help='pretrained model path')
    parser.add_argument('--training_strategy', default='biastuning', help='how to train the model')

    parser.add_argument('--device', default='cuda:0', help='device to train on')

    args = parser.parse_args()

    return args

def main_datautils(config, use_norm=True):
    selected_idxs = [0,12,42,79,100]
    print(config)
    dataset_dict, dataset_sizes, label_dict = get_data(config, tr_folder_start=0, tr_folder_end=78000, val_folder_start=0, val_folder_end=104000, use_norm=use_norm)
    print(len(dataset_dict['train']))
    for i in selected_idxs:
        temp = (dataset_dict['train'][i])
        print(temp[-1])
        print(temp[-2])
        print(temp[0].shape)
        print(temp[1].shape)
        plt.imshow(temp[0].permute(1,2,0), cmap='gray')
        plt.show()
        plt.imshow(temp[1], cmap='gray')
        plt.show()

def main_model(config):
    print(config)
    label_dict = {
        'liver':0,
        'tumor':1
    }
    model = Prompt_Adapted_SAM(config, label_dict)
    for name, p in model.named_parameters():
        print(name)
    return

def main_test(data_config, model_config, pretrained_path):
    test_start = 104
    test_end = 131
    test(data_config, model_config, pretrained_path, test_start, test_end, device='cuda:0')

def main_train(data_config, model_config, pretrained_path, save_path, training_strategy='biastuning', device='cuda:0'):
    #load data
    if data_config['data']['name']=='LITS':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=78, val_folder_start=78, val_folder_end=104)
    elif data_config['data']['name']=='IDRID':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=40, val_folder_start=40, val_folder_end=104)
        dataloader_dict = {}
        for x in ['train','val']:
            dataloader_dict[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=model_config['training']['batch_size'], shuffle=True, num_workers=4)
    elif data_config['data']['name']=='ENDOVIS':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=180, val_folder_start=180, val_folder_end=304, no_text_mode=True)
        dataloader_dict = {}
        for x in ['train','val']:
            dataloader_dict[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=model_config['training']['batch_size'], shuffle=True, num_workers=4)
    elif data_config['data']['name']=='ENDOVIS 18':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=18000, val_folder_start=0, val_folder_end=34444, no_text_mode=True)
        dataloader_dict = {}
        for x in ['train','val']:
            dataloader_dict[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=model_config['training']['batch_size'], shuffle=True, num_workers=4)
    elif data_config['data']['name']=='CHESTXDET':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=18000, val_folder_start=0, val_folder_end=34444, no_text_mode=True)
        dataloader_dict = {}
        for x in ['train','val']:
            dataloader_dict[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=model_config['training']['batch_size'], shuffle=True, num_workers=4)
    elif data_config['data']['name']=='CHOLEC 8K':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=18000, val_folder_start=0, val_folder_end=34444, no_text_mode=True)
        dataloader_dict = {}
        for x in ['train','val']:
            dataloader_dict[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=model_config['training']['batch_size'], shuffle=True, num_workers=4)
    elif data_config['data']['name']=='ULTRASOUND':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=18000, val_folder_start=0, val_folder_end=34444, no_text_mode=True)
        dataloader_dict = {}
        for x in ['train','val']:
            dataloader_dict[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=model_config['training']['batch_size'], shuffle=True, num_workers=4)
    elif data_config['data']['name']=='GLAS':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=18000, val_folder_start=0, val_folder_end=34444, no_text_mode=True)
        dataloader_dict = {}
        for x in ['train','val']:
            dataloader_dict[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=model_config['training']['batch_size'], shuffle=True, num_workers=4)
    elif data_config["data"]["name"] == "ArcadeDataset":
        print("HERE")
        data_split_csv_path = data_config["data"]["data_split_csv"]
        data_split = pd.read_csv(data_split_csv_path)

        dataset_dict = {}
        dataloader_dict = {}

        use_norm = True
        no_text_mode = False

        for split in ["train", "val"]:
            # Filter the CSV for the current split
            split_data = data_split[data_split["split"] == split]["imgs"].tolist()

            # Pass the filtered data to the dataset class (ArcadeDataset)
            dataset_dict[split] = ArcadeDataset(
                config=data_config,
                file_list=split_data,  # Pass file_list as (image_path, mask_path) tuples
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


    #load model
    #change the img size in model config according to data config
    in_channels = model_config['in_channels']
    out_channels = model_config['num_classes']
    img_size = model_config['img_size']
    if model_config['arch']=='Prompt Adapted SAM':
        model = Prompt_Adapted_SAM(model_config, label_dict, device, training_strategy=training_strategy)
    elif model_config['arch']=='UNet':
        model = smp.Unet(
                encoder_name="resnet18",      
                encoder_depth=1,               
                encoder_weights="imagenet",     
                in_channels=3,   
                decoder_channels=(16,),               
                classes=1,                      
                activation="sigmoid",
                decoder_use_batchnorm=True                
            )
        model = smp.Unet(
    encoder_name="resnet18",         
    encoder_depth=4,             
    encoder_weights=None,         
    in_channels=3,                # Input channels (e.g., RGB)
    decoder_channels=(256, 128, 64, 32),  # Decoder channels approximating the original design
    classes=1,                    # For binary segmentation
    activation="sigmoid",         # Sigmoid to map output to [0, 1]
    decoder_use_batchnorm=False    # Optional: Use BatchNorm for improved training stability
)
    elif model_config['arch']=='UNext':
        model = UNext(num_classes=out_channels, input_channels=in_channels, img_size=img_size)
    elif model_config['arch']=='MedT':
        #TODO
        model = MedT(img_size=img_size, num_classes=out_channels)
    elif model_config['arch']=='TransUNet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = out_channels
        config_vit.n_skip = 3
        # if args.vit_name.find('R50') != -1:
        #     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = VisionTransformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes)

    #load model weights
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))

    #training parameters
    print('number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    training_params = model_config['training']
    if training_params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=float(training_params['lr']), weight_decay=float(training_params['weight_decay']))
    elif training_params['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=float(training_params['lr']), weight_decay=float(training_params['weight_decay']), momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=training_params['schedule_step'], gamma=training_params['schedule_step_factor'])
    
    criterion = []
    if 'dice' in training_params['loss']:
        criterion.append(dice_loss)
    if 'focal' in training_params['loss']:
        criterion.append(multiclass_focal_loss)
    if 'weighted CE' in training_params['loss']:
        criterion.append(weighted_ce_loss)
    if criterion==[]:
        criterion = [nn.BCELoss()]
    
    #train the model
    if data_config['data']['name']=='LITS':
        model = train(model, dataset_dict['train'], dataset_dict['val'], criterion, optimizer, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config['data']['name']=='IDRID':
        model = train_dl(model, dataloader_dict, dataset_sizes, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config['data']['name']=='ENDOVIS':
        model = train_dl(model, dataloader_dict, dataset_sizes, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config['data']['name']=='ENDOVIS 18':
        model = train_dl(model, dataloader_dict, dataset_sizes, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config['data']['name']=='CHOLEC 8K':
        model = train_dl(model, dataloader_dict, dataset_sizes, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config['data']['name']=='ULTRASOUND':
        model = train_dl(model, dataloader_dict, dataset_sizes, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config['data']['name']=='CHESTXDET':
        model = train_dl(model, dataloader_dict, dataset_sizes, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config['data']['name']=='GLAS':
        model = train_dl(model, dataset_dict, dataset_sizes, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config["data"]["name"] == "ArcadeDataset":
        save_path = "./models" + data_config["data"]["root_path"].split("/")[-1]
        model = train_dl(
            model,
            dataset_dict,
            dataset_sizes,
            criterion,
            optimizer,
            exp_lr_scheduler,
            save_path,
            save_dir=f"./{args.training_strategy}/{data_config['data']['root_path'].split('/')[-1]}",
            num_epochs=training_params["num_epochs"],
            bs=2,
            device=device
        )


if __name__ == '__main__':
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # #for checking data_utils
    # main_datautils(data_config, use_norm=False)

    # #for checking model
    # main_model(model_config)

    # #for testing on the test dataset
    # main_test(data_config, model_config, args.pretrained_path)

    # for training the model
    main_train(data_config, model_config, args.pretrained_path, args.save_path, args.training_strategy, device=args.device)
