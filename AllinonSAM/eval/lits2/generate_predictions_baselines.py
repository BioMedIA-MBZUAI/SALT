import torch
import yaml
import sys
import copy
import os
sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/SVDSAM/")

from data_utils import *
from model import *
from utils import *
from baselines import UNet, UNext, medt_net
from vit_seg_modeling import VisionTransformer
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from axialnet import MedT

label_names = ['Liver','Tumor']
# visualize_li = [[1,0,0],[0,1,0],[1,0,0], [0,0,1], [0,0,1]]
label_dict = {}
# visualize_dict = {}
for i,ln in enumerate(label_names):
        label_dict[ln] = i
        # visualize_dict[ln] = visualize_li[i]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='config_tmp.yml',
                        help='data folder file path')

    parser.add_argument('--data_config', default='config_tmp.yml',
                        help='data config file path')

    parser.add_argument('--model_config', default='model_baseline.yml',
                        help='model config file path')

    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')

    parser.add_argument('--save_path', default='checkpoints/temp.pth',
                        help='pretrained model path')

    parser.add_argument('--gt_path', default='',
                        help='ground truth path')

    parser.add_argument('--device', default='cuda:0', help='device to train on')

    parser.add_argument('--codes', default='1,2,1,3,3', help='numeric label to save per instrument')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    codes = args.codes.split(',')
    codes = [int(c) for c in codes]

    label_dict = {
            'Liver': [[100,0,100]],
            'Kidney': [[255,255,0]],
            'Pancreas': [[0,0,255]],
            'Vessels': [[255,0,0]],
            'Adrenals': [[0,255,255]],
            'Gall Bladder': [[0,255,0]],
            'Bones': [[255,255,255]],
            'Spleen': [[255,0,255]]
        }


    #make folder to save visualizations
    os.makedirs(os.path.join(args.save_path,"preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_gt"),exist_ok=True)


    #load model
    #change the img size in model config according to data config
    in_channels = model_config['in_channels']
    out_channels = model_config['num_classes']
    img_size = model_config['img_size']
    if model_config['arch']=='Prompt Adapted SAM':
        model = Prompt_Adapted_SAM(model_config, label_dict, args.device, training_strategy='svdtuning')
    elif model_config['arch']=='UNet':
        model = UNet(in_channels=in_channels, out_channels=out_channels)
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

    model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    model = model.to(args.device)
    model = model.eval()

    #load data transform
    data_transform = LiTS2_Transform(config=data_config)

    #dice
    tumor_dices = []
    tumor_ious=[]
    liver_dices = []
    liver_ious=[]


    #load data
    root_path = "/media/ubuntu/New Volume/jay/LiTS2/archive"
    imgs_path = os.path.join(root_path, 'dataset_6/dataset_6')
    test_csv = pd.read_csv(os.path.join(root_path, 'lits_test.csv'))
    for i in range(len(test_csv)):
        if i%10!=0:
            continue
        img_path = (os.path.join(root_path,'dataset_6',test_csv['filepath'].iloc[i][18:]))
        image_name = test_csv['filepath'].iloc[i][28:]
        liver_mask_path = os.path.join(root_path,'dataset_6',test_csv['liver_maskpath'].iloc[i][18:])
        tumor_mask_path = os.path.join(root_path,'dataset_6',test_csv['tumor_maskpath'].iloc[i][18:])

        # print(img_path)
        img = torch.as_tensor(np.array(Image.open(img_path).convert("RGB")))
        img = img.permute(2,0,1)
        C,H,W = img.shape
        #make a dummy mask of shape 1XHXW

        try:
            liver_label = torch.Tensor(np.array(Image.open(liver_mask_path)))[:,:,0]
            tumor_label = torch.Tensor(np.array(Image.open(tumor_mask_path)))[:,:,0]
        except:
            liver_label = torch.zeros(H, W)
            tumor_label = torch.zeros(H, W)
            # label = np.array(Image.open(gt_path).convert("RGB"))
            # temp = np.zeros((H,W)).astype('uint8')
            # selected_color_list = label_dict[args.labels_of_interest]
            # for c in selected_color_list:
            #     temp = temp | (np.all(np.where(label==c,1,0),axis=2))

            # # plt.imshow(gold)
            # # plt.show()
            # mask = torch.Tensor(temp).unsqueeze(0)

        liver_label = liver_label.unsqueeze(0)
        liver_label = (liver_label>0)+0
        tumor_label = tumor_label.unsqueeze(0)
        tumor_label = (tumor_label>0)+0

        #convert all grayscale pixels due to resizing back to 0, 1
        img1, liver_label = data_transform(img, liver_label, is_train=False, apply_norm=True)
        liver_label = (liver_label>=0.5)+0
        # liver_label = liver_label[0]

        #convert all grayscale pixels due to resizing back to 0, 1
        _, tumor_label = data_transform(img, tumor_label, is_train=False, apply_norm=True)
        tumor_label = (tumor_label>=0.5)+0
        # tumor_label = tumor_label[0]

        #get image embeddings
        img = img1.unsqueeze(0).to(args.device)  #1XCXHXW
        final_label = torch.cat([liver_label,tumor_label], dim=0)
        masks,_ = model(img,'')
        masks_liver = masks[:,0,:,:].cpu()
        masks_tumor = masks[:,1,:,:].cpu()

        plt.imshow(((masks_liver>=0.5)[0]), cmap='gray')
        plt.savefig(os.path.join(args.save_path,'rescaled_preds', image_name[:-4] +'_liver.png'))
        plt.close()
        # plt.show()

        plt.imshow(((masks_tumor>=0.5)[0]), cmap='gray')
        plt.savefig(os.path.join(args.save_path,'rescaled_preds', image_name[:-4] +'_tumor.png'))
        plt.close()
        # plt.show()


        plt.imshow((liver_label[0]), cmap='gray')
        plt.savefig(os.path.join(args.save_path,'rescaled_gt', image_name[:-4] +'_liver.png'))
        plt.close()
        # plt.show()


        plt.imshow((tumor_label[0]), cmap='gray')
        plt.savefig(os.path.join(args.save_path,'rescaled_gt', image_name[:-4] +'_tumor.png'))
        plt.close()
        # plt.show()

        # print("dice: ",dice_coef(label, (masks>0.5)+0))
        # print(liver_label.shape)
        # print((((masks[0]>=0.5)+0).unsqueeze(0)).shape)
        liver_dices.append(dice_coef(liver_label, ((masks_liver[0]>=0.5)+0).unsqueeze(0)))
        tumor_dices.append(dice_coef(tumor_label, ((masks_tumor[0]>=0.5)+0).unsqueeze(0)))

        liver_ious.append(iou_coef(liver_label, ((masks_liver[0]>=0.5)+0).unsqueeze(0)))
        tumor_ious.append(iou_coef(tumor_label, ((masks_tumor[0]>=0.5)+0).unsqueeze(0)))
        # 1/0
        # break
    print("Liver DICE: ",torch.mean(torch.Tensor(liver_dices)))
    print("Liver IoU", torch.mean(torch.Tensor(liver_ious)))
    print("Tumor DICE", torch.mean(torch.Tensor(tumor_dices)))
    print("Tumor IoU", torch.mean(torch.Tensor(tumor_ious)))

if __name__ == '__main__':
    main()


        


