import torch
import yaml
import sys
import copy
import os
import h5py
sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/SVDSAM/")

from data_utils import *
from model import *
from utils import *

label_names = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Liver', 'Stomach', 'Aorta', 'Pancreas']

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

    parser.add_argument('--labels_of_interest', default='Left Prograsp Forceps,Maryland Bipolar Forceps,Right Prograsp Forceps,Left Large Needle Driver,Right Large Needle Driver', help='labels of interest')

    parser.add_argument('--codes', default='1,2,1,3,3', help='numeric label to save per instrument')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    labels_of_interest = args.labels_of_interest.split(',')
    codes = args.codes.split(',')
    codes = [int(c) for c in codes]

    label_dict = {
            "Spleen":1,
            "Right Kidney": 2,
            "Left Kidney": 3,
            "Gall Bladder": 4,
            "Liver": 5,
            "Stomach": 6,
            "Aorta": 7,
            "Pancreas": 8
        }


    #make folder to save visualizations
    os.makedirs(os.path.join(args.save_path,"preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_gt"),exist_ok=True)

    #load model
    model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device, training_strategy='svdtuning')
    # model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device, training_strategy='lora')

    #legacy model support
    sdict = torch.load(args.pretrained_path, map_location=args.device)
    # for key in list(sdict.keys()):
    #     if 'sam_encoder.neck' in key:
    #         if '0' in key:
    #             new_key = key.replace('0','conv1')
    #         if '1' in key:
    #             new_key = key.replace('1','ln1')
    #         if '2' in key:
    #             new_key = key.replace('2','conv2')
    #         if '3' in key:
    #             new_key = key.replace('3','ln2')
    #         sdict[new_key] = sdict[key]
    #         _ = sdict.pop(key)
    #     if 'mask_decoder' in key:
    #         if 'trainable' in key:
    #             _ = sdict.pop(key)   
    
    model.load_state_dict(sdict,strict=True)
    model = model.to(args.device)
    model = model.eval()

    #load data transform
    data_transform = BTCV_Transform(config=data_config)

    #dice
    dices = []
    ious=[]

    #load data
    for i,h5_name in enumerate(sorted(os.listdir(args.data_folder))):
        # if i%5!=0:
        #     continue
        h5_path = (os.path.join(args.data_folder,h5_name))
        data = h5py.File(h5_path)
        all_img, all_label = data['image'], data['label']

        for i in range(all_img.shape[0]):
            if i%5!=0:
                continue
            img = torch.as_tensor(all_img[i]).unsqueeze(0).repeat(3,1,1)
            label = torch.as_tensor(all_label[i])
            # print("image shape", img.shape)
            # print("label shape: ", label.shape)
            # 1/0
            # img = img.permute(2,0,1)
            C,H,W = img.shape
            #make a dummy mask of shape 1XHXW
            selected_color = label_dict[args.labels_of_interest]
            temp = (label==selected_color)+0

            # plt.imshow(gold)
            # plt.show()
            mask = torch.Tensor(temp).unsqueeze(0)
            img, mask = data_transform(img, mask, is_train=False, apply_norm=True)
            mask = (mask>=0.5)+0

            #get image embeddings
            img = img.unsqueeze(0).to(args.device)  #1XCXHXW
            img_embeds = model.get_image_embeddings(img)

            # generate masks for all labels of interest
            img_embeds_repeated = img_embeds.repeat(len(labels_of_interest),1,1,1)
            x_text = [t for t in labels_of_interest]
            masks = model.get_masks_for_multiple_labels(img_embeds_repeated, x_text).cpu()
            argmax_masks = torch.argmax(masks, dim=0)
            final_mask = torch.zeros(masks[0].shape)
            final_mask_rescaled = torch.zeros(masks[0].shape).unsqueeze(-1).repeat(1,1,3)
            #save masks
            for i in range(final_mask.shape[0]):
                for j in range(final_mask.shape[1]):
                    final_mask[i,j] = codes[argmax_masks[i,j]] if masks[argmax_masks[i,j],i,j]>=0.5 else 0
                    # final_mask_rescaled[i,j] = torch.Tensor(visualize_dict[(labels_of_interest[argmax_masks[i,j]])] if masks[argmax_masks[i,j],i,j]>=0.5 else [0,0,0])

            # save_im = Image.fromarray(final_mask.numpy())
            # save_im.save(os.path.join(args.save_path,'preds', img_name))

            # plt.imshow(final_mask_rescaled,cmap='gray')
            # plt.savefig(os.path.join(args.save_path,'rescaled_preds', img_name))
            # plt.close()

            # print("label shape: ", label.shape)
            # plt.imshow(label[0], cmap='gray')
            # plt.show()

            plt.imshow((masks[0]>=0.5), cmap='gray')
            plt.savefig(os.path.join(args.save_path,'rescaled_preds', h5_name[:h5_name.find('.')]+"_"+str(i)+".png"))
            plt.close()

            plt.imshow((mask[0]), cmap='gray')
            plt.savefig(os.path.join(args.save_path,'rescaled_gt', h5_name[:h5_name.find('.')]+"_"+str(i)+".png"))
            plt.close()

            # print("dice: ",dice_coef(label, (masks>0.5)+0))
            dices.append(dice_coef(mask, (masks>=0.5)+0))
            ious.append(iou_coef(mask, (masks>=0.5)+0))
            # break
    print(torch.mean(torch.Tensor(dices)))
    print(torch.mean(torch.Tensor(ious)))

if __name__ == '__main__':
    main()


        


