import random
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, models
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn.functional import pad
from skimage.transform import resize
import nibabel as nib
import time
import json

from data_transforms.polyp_transform import Polyp_Transform


class Polyp_Dataset(Dataset):
    def __init__(self, config, is_train=False, shuffle_list = True, apply_norm=True, no_text_mode=False) -> None:
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.label_names = config['data']['label_names']
        self.num_classes = len(self.label_names)
        self.config = config
        self.apply_norm = apply_norm
        self.no_text_mode = no_text_mode
        self.train_df = os.path.join(self.root_path, 'train.csv')
        self.val_df = os.path.join(self.root_path, 'val.csv')

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

        #define data transform
        self.data_transform = Polyp_Transform(config=config)
        print("Length of dataset: ", len(self.img_path_list))

    def __len__(self):
        return len(self.img_path_list)

    def populate_lists(self):
        # imgs_path = os.path.join(self.root_path, 'CVC_clinicTRimage')
        # labels_path = os.path.join(self.root_path, 'CVC_clinicTRmask')
        # imgs_path = os.path.join(self.root_path, 'kvasirsegTRimage')
        # labels_path = os.path.join(self.root_path, 'kvasirsegTRmask')
        if self.is_train:
            imgs_path = os.path.join(self.root_path, 'TrainDataset/image')
            labels_path = os.path.join(self.root_path, 'TrainDataset/masks')
        else:
            imgs_path = os.path.join(self.root_path, 'TestDataset/CVC-ColonDB/images')
            labels_path = os.path.join(self.root_path, 'TestDataset/CVC-ColonDB/masks')
        # imgs_path = os.path.join(self.root_path, 'NewTRimage')
        # labels_path = os.path.join(self.root_path, 'NewTRmask')
        # if self.is_train:
        #     df = pd.read_csv(self.train_df)
        # else:
        #     df = pd.read_csv(self.val_df)
        # for i in range(len(df)):
        #     img = df['image_path'].iloc[i]
        #     lbl = df['mask_path'].iloc[i]
        for img in os.listdir(imgs_path):
            if self.no_text_mode:
                self.img_names.append(img)
                self.img_path_list.append(os.path.join(imgs_path,img))
                self.label_path_list.append(os.path.join(labels_path, img))
                self.label_list.append('')
            else:
                for label_name in self.label_names:
                    self.img_names.append(img)
                    self.img_path_list.append(os.path.join(imgs_path,img))
                    self.label_path_list.append(os.path.join(labels_path, img))
                    self.label_list.append(label_name)


    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['data']['volume_channel']==2:
            img = img.permute(2,0,1)
            
        try:
            label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
            if len(label.shape)==3:
                label = label[:,:,0]
        except:
            label = torch.zeros(img.shape[1], img.shape[2])

        
        label = label.unsqueeze(0)
        label = (label>0)+0
        label_of_interest = self.label_list[index]

        #convert all grayscale pixels due to resizing back to 0, 1
        img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
        label = (label>=0.5)+0
        label = label[0]


        return img, label, self.img_path_list[index], label_of_interest
