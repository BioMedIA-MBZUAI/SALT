import torch
import yaml
import sys
import copy
import os
sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/SVDSAM/")

from data_utils import *
from model import *
from utils import *

label_names = ['Liver', 'Tumor']
# visualize_li = [[1,0,0],[0,1,0],[1,0,0], [0,0,1], [0,0,1]]
label_dict = {}
# visualize_dict = {}
for i,ln in enumerate(label_names):
        label_dict[ln] = i
        # visualize_dict[ln] = visualize_li[i]

def parse_args():
    parser = argparse.ArgumentParser()

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


    #make folder to save visualizations
    os.makedirs(os.path.join(args.save_path,"preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_gt"),exist_ok=True)

    #load model
    model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device, training_strategy='svdtuning')
    # model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device, training_strategy='lora')
    
    #legacy model support
    if args.pretrained_path:
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
        # if i%10!=0:
        #     continue
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

        #get positive point prompts
        _,y_liver,x_liver = torch.where(liver_label==1)
        pos_prompts = torch.cat([x_liver.unsqueeze(1),y_liver.unsqueeze(1)],dim=1)

        #get negative point prompts
        _,y_liver_neg,x_liver_neg = torch.where(liver_label==0)
        neg_prompts = (torch.cat([x_liver_neg.unsqueeze(1),y_liver_neg.unsqueeze(1)],dim=1))

        if len(y_liver)>0:
            pos_point_idx = random.randint(0,y_liver.shape[0]-1)
            neg_point_idx = random.randint(0,y_liver_neg.shape[0]-1)
            points_liver = (pos_prompts[pos_point_idx].unsqueeze(0).unsqueeze(0).to(args.device), torch.Tensor([1]).unsqueeze(0).to(args.device))
        else:
            neg_point_idx1 = random.randint(0,y_liver_neg.shape[0]-1)
            neg_point_idx2 = random.randint(0,y_liver_neg.shape[0]-1)
            points_liver = (neg_prompts[neg_point_idx1].unsqueeze(0).unsqueeze(0).to(args.device), torch.Tensor([-1]).unsqueeze(0).to(args.device))

        #convert all grayscale pixels due to resizing back to 0, 1
        _, tumor_label = data_transform(img, tumor_label, is_train=False, apply_norm=True)
        tumor_label = (tumor_label>=0.5)+0
        # tumor_label = tumor_label[0]

        #get positive point prompts
        _,y_tumor,x_tumor = torch.where(tumor_label==1)
        pos_prompts = torch.cat([x_tumor.unsqueeze(1),y_tumor.unsqueeze(1)],dim=1)

        #get negative point prompts
        _,y_tumor_neg,x_tumor_neg = torch.where(tumor_label==0)
        neg_prompts = (torch.cat([x_tumor_neg.unsqueeze(1),y_tumor_neg.unsqueeze(1)],dim=1))

        if len(y_tumor)>0:
            pos_point_idx = random.randint(0,y_tumor.shape[0]-1)
            neg_point_idx = random.randint(0,y_tumor_neg.shape[0]-1)
            # points_tumor = (torch.cat([pos_prompts[pos_point_idx].unsqueeze(0), neg_prompts[neg_point_idx].unsqueeze(0)],dim=0).unsqueeze(0).to(args.device), torch.Tensor([1,-1]).unsqueeze(0).to(args.device))
            points_tumor = (pos_prompts[pos_point_idx].unsqueeze(0).unsqueeze(0).to(args.device), torch.Tensor([1]).unsqueeze(0).to(args.device))

        else:
            neg_point_idx1 = random.randint(0,y_tumor_neg.shape[0]-1)
            neg_point_idx2 = random.randint(0,y_tumor_neg.shape[0]-1)
            # points_tumor = (torch.cat([neg_prompts[neg_point_idx1].unsqueeze(0), neg_prompts[neg_point_idx2].unsqueeze(0)],dim=0).unsqueeze(0).to(args.device), torch.Tensor([-1,-1]).unsqueeze(0).to(args.device))
            points_tumor = (neg_prompts[neg_point_idx1].unsqueeze(0).unsqueeze(0).to(args.device), torch.Tensor([-1]).unsqueeze(0).to(args.device))



        #get image embeddings
        img = img1.unsqueeze(0).to(args.device)  #1XCXHXW
        img_embeds = model.get_image_embeddings(img)

        # generate masks for all labels of interest
        img_embeds_repeated = img_embeds.repeat(1,1,1,1)
        masks_liver = model.get_masks_with_manual_prompts(img_embeds_repeated, points=points_liver).cpu()
        masks_tumor = model.get_masks_with_manual_prompts(img_embeds_repeated, points=points_tumor).cpu()

        plt.imshow((masks_liver[0]>=0.5), cmap='gray')
        if len(y_liver)>0:
            plt.scatter(x_liver[pos_point_idx], y_liver[pos_point_idx], c='green')
            # plt.scatter(x_neg[neg_point_idx], y_neg[neg_point_idx], c='red')
        else:
            plt.scatter(x_liver_neg[neg_point_idx1], y_liver_neg[neg_point_idx1], c='red')
            # plt.scatter(x_neg[neg_point_idx2], y_neg[neg_point_idx2], c='red')
        plt.savefig(os.path.join(args.save_path,'rescaled_preds', image_name[:-4] +'_liver.png'))
        plt.close()
        # plt.show()

        plt.imshow((masks_tumor[0]>=0.5), cmap='gray')
        if len(y_tumor)>0:
            plt.scatter(x_tumor[pos_point_idx], y_tumor[pos_point_idx], c='green')
            # plt.scatter(x_neg[neg_point_idx], y_neg[neg_point_idx], c='red')
        else:
            plt.scatter(x_tumor_neg[neg_point_idx1], y_tumor_neg[neg_point_idx1], c='red')
            # plt.scatter(x_neg[neg_point_idx2], y_neg[neg_point_idx2], c='red')
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


        


