import random
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_transforms.btcv_transform import BTCV_Transform


class BTCV_Dataset(Dataset):
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
        self.label_dict = {
            "Spleen":1,
            "Right Kidney": 2,
            "Left Kidney": 3,
            "Gall Bladder": 4,
            "Liver": 5,
            "Stomach": 6,
            "Aorta": 7,
            "Pancreas": 8
        }

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

        #define data transform
        self.data_transform = BTCV_Transform(config=config)

    def __len__(self):
        return len(self.img_path_list)

    def populate_lists(self):
        imgs_labels_path = os.path.join(self.root_path, 'train_npz')

        for npz in os.listdir(imgs_labels_path):
            case_no = int(npz[4:npz.find("_")])
            if self.is_train:
                if case_no>=34:
                    continue
            else:
                if case_no<34:
                    continue
            # print(img)
            if self.no_text_mode:
                self.img_names.append(npz)
                self.img_path_list.append(os.path.join(imgs_labels_path,npz))
                self.label_path_list.append(os.path.join(imgs_labels_path, npz))
                self.label_list.append('')
            else:
                for label_name in self.label_names:
                    self.img_names.append(npz)
                    self.img_path_list.append(os.path.join(imgs_labels_path,npz))
                    self.label_path_list.append(os.path.join(imgs_labels_path, npz))
                    self.label_list.append(label_name)


    def __getitem__(self, index):
        data = np.load(self.img_path_list[index])
        img, all_class_labels = data['image'], data['label']
        # print("img max min: ", np.max(img), np.min(img))
        img = torch.Tensor(img).unsqueeze(0).repeat(3,1,1)
            
        try:
            label = torch.Tensor(all_class_labels)==self.label_dict[self.label_list[index]]+0
            if len(label.shape)==3:
                label = label[:,:,0]
        except:
            1/0
            label = torch.zeros(img.shape[1], img.shape[2])
        
        label = label.unsqueeze(0)
        label_of_interest = self.label_list[index]

        #convert all grayscale pixels due to resizing back to 0, 1
        img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
        label = (label>=0.5)+0
        label = label[0]


        return img, label, self.img_path_list[index], label_of_interest
