import torch
import yaml
import sys
import copy
import os
sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/SVDSAM/")

from data_utils import *
from model import *
from utils import *

label_names = ['liver', 'tumor']
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

    parser.add_argument('--labels_of_interest', default='tumor', help='labels of interest')

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
            'liver': 1,
            'tumor': 2,
        }

    #change the img size in model config according to data config
    model_config['sam']['img_size'] = data_config['data_transforms']['img_size']
    model_config['sam']['num_classes'] = len(data_config['data']['label_list'])


    #make folder to save visualizations
    os.makedirs(os.path.join(args.save_path,"preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_preds"),exist_ok=True)
    if args.gt_path:
        os.makedirs(os.path.join(args.save_path,"rescaled_gt"),exist_ok=True)

    #load model
    model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device, training_strategy='svdtuning')
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

    data_transform = Slice_Transforms(config=data_config)
    label_text = args.labels_of_interest
    #load data
    for i, file_name in enumerate(sorted(os.listdir(args.data_folder))):
        print(i)
        file_path = os.path.join(args.data_folder, file_name)
        im_nib = nib.load(file_path)

        # for 2d mode
        #image loading and conversion to rgb by replicating channels
        if data_config['data']['volume_channel']==2: #data originally is HXWXC
            im = (torch.Tensor(np.asanyarray(im_nib.dataobj)).permute(2,0,1).unsqueeze(1).repeat(1,3,1,1))
        else: #data originally is CXHXW
            im = (torch.Tensor(np.asanyarray(im_nib.dataobj)).unsqueeze(1).repeat(1,3,1,1))
        num_slices = im.shape[0]
        preds = []
        for i in range(num_slices):
            slice_im = im[i]
            slice_im = data_transform(slice_im)
            slice_im = torch.Tensor(slice_im).to(args.device)
            with torch.set_grad_enabled(False):
                outputs, reg_loss = model(slice_im, [label_text], [i])
                slice_pred = (outputs>=0.5) +0
                preds.append(slice_pred)
        
        # print(len(preds))
        # print(preds[0].shape)
        preds = (torch.cat(preds, dim=0).permute(1,2,0)).cpu().numpy().astype('uint8')
        # print(preds.shape)
        ni_img = nib.Nifti1Image(preds, im_nib.affine)
        nib.save(ni_img, os.path.join(args.save_path,'preds',file_name))


if __name__ == '__main__':
    main()
