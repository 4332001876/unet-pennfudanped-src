import glob
import numpy as np
import torch
import cv2
from PIL import Image

from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train_images/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/train_masks/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/valid_images/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/valid_masks/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    # train_image_transform = A.Compose([
    #     A.Resize(img_size, img_size, always_apply=True),
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.2),
    #     A.RandomSunFlare(p=0.2),
    #     A.RandomFog(p=0.2),
    #     A.Rotate(limit=25),
    # ])


    train_image_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])
    return valid_image_transform

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        # Convert string names to class values for masks.
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image / 255.0
        mask = cv2.imread(self.mask_paths[index], -1)

        # Make all instances of person 255 pixel value and background 0.


        
        mask = Image.fromarray(mask)

        image = np.array(self.tfms(image))
        mask  = np.array(self.tfms(mask))

        im = mask > 0
        mask[im] = 1
        mask[np.logical_not(im)] = 0
        mask = mask[0,:,:]
    
      
        # Get colored label mask.
        # mask = get_label_mask(mask, self.class_values, self.label_colors_list)
       
        # image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 

        # print("######################################3", image.shape, mask.shape)

        return image, mask

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=False
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, drop_last=False
    )

    return train_data_loader, valid_data_loader