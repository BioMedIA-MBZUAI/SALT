import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from data_transforms.data_transform import Data_Transform

class GeneralDataset(Dataset):
    def __init__(self, config, file_list=None, is_train=False, shuffle_list=True, apply_norm=True, no_text_mode=False) -> None:
        super().__init__()
        self.root_dir = config['data']['root_path']
        self.is_train = is_train
        self.config = config
        self.apply_norm = apply_norm
        self.no_text_mode = no_text_mode
        self.label_names = config['data']['label_names']
        self.label_list = config['data']['label_list']

        self.image_dir = os.path.join(self.root_dir, 'images')
        self.mask_dir = os.path.join(self.root_dir, 'masks')

        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise ValueError(f"Image or mask directory not found in {self.root_dir}")

        # If a file list is provided, use it. Otherwise, load all images in the directory.
        if file_list is not None:
            self.images = file_list
        else:
            self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png') or f.endswith('.jpg')])

        if shuffle_list:
            np.random.shuffle(self.images)

        self.data_transform = Data_Transform(config=config)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        image_name = os.path.splitext(os.path.basename(img_name))[0]
        img_path = os.path.join(self.image_dir, image_name+'.png')
        mask_name = os.path.splitext(image_name)[0] + '_mask.png'
        # print(self.mask_dir)
        # print(mask_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # print("Data point")
        # print("Train: " , self.is_train)
        # print("Img:" , img_path)
        # print("Mask: ", mask_path)

        # Load and process image
        img = Image.open(img_path).convert("RGB")
        img = torch.as_tensor(np.array(img)).permute(2, 0, 1)  # Change to CHW format

        # Load and process mask
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = torch.as_tensor(np.array(mask))
        else:
            print(f"Mask not found for {mask_name}, using blank mask")
            mask = torch.zeros((img.shape[1], img.shape[2]), dtype=torch.uint8)

        # Resize mask to match image size if necessary
        if mask.shape != (img.shape[1], img.shape[2]):
            mask = torch.as_tensor(np.array(Image.fromarray(mask.numpy()).resize((img.shape[2], img.shape[1]))))

        # Convert mask to binary
        mask = (mask > 0).float()

        # Apply data transformations
        img, mask = self.data_transform(img, mask.unsqueeze(0), is_train=self.is_train, apply_norm=self.apply_norm)

        if self.no_text_mode:
            return img, mask, img_path, ""
        else:
            return img, mask[0], img_path, self.label_names[1]  # Assuming "Vein" is the label of interest

    def get_category_ids(self, image_id):
        img_name = self.images[image_id]
        mask_name = os.path.splitext(img_name)[0] + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        unique_values = np.unique(mask)
        category_ids = [self.label_list[v] for v in unique_values if v in self.label_list]
        return category_ids
