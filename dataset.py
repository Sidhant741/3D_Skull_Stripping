import copy
import nibabel as nib
import numpy as np
import os
import tarfile
import json
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split
from config import (
    DATASET_PATH, TASK_ID, TRAIN_VAL_TEST_SPLIT,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)

from transforms import train_transform

class SkullStrippingDataset(Dataset):
    """
    Dataset class for skull stripping.
    Assumes directory structure:
        dataset/
            ├── train/
            │   ├── images/
            │   └── masks/
            ├── valid/
            └── test/
    """
    def __init__(self, root_dir, mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.image_dir = os.path.join(root_dir, mode, 'images')
        self.mask_dir = os.path.join(root_dir, mode, 'masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Build corresponding mask filename
        if img_name.endswith('.nii.gz'):
            base_name = img_name[:-7]  # Remove '.nii.gz'
            suffixes = ['_manual.nii.gz', '_ss.nii.gz', '_mask.nii.gz', '_seg.nii.gz']
        elif img_name.endswith('.nii'):
            base_name = img_name[:-4]  # Remove '.nii'
            suffixes = ['_ss.nii', '_manual.nii', '_mask.nii', '_seg.nii']
        else:
            raise ValueError(f"Unexpected file extension in: {img_name}")

        mask_path = None
        candidates = [base_name + s for s in suffixes]
        for mask_name in candidates:
            candidate_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(candidate_path):
                mask_path = candidate_path
                break
        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for {img_name}. Tried: {', '.join(candidates)}"
            )

        # Load the image and mask
        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        image = np.moveaxis(image, -1, 0)  # Channels first
        mask = np.moveaxis(mask, -1, 0)

        # Add channel dimension -> (1, D, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        sample = {'name': img_name, 'image': image, 'label': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

def get_train_val_test_Dataloaders(train_transforms=None, val_transforms=None, test_transforms=None):
    train_dataset = SkullStrippingDataset(DATASET_PATH, mode='train', transform=train_transforms)
    val_dataset = SkullStrippingDataset(DATASET_PATH, mode='valid', transform=val_transforms)
    test_dataset = SkullStrippingDataset(DATASET_PATH, mode='test', transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_dataset = SkullStrippingDataset(DATASET_PATH, mode='train', transform=train_transform)
    # val_dataset = SkullStrippingDataset(DATASET_PATH, mode='valid', transform=val_transforms)
    # test_dataset = SkullStrippingDataset(DATASET_PATH, mode='test', transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    batch = next(iter(train_loader))

    # Check the keys in the batch
    print("Batch keys:", batch.keys())

    # Check individual shapes
    print("Image shape:", batch["image"].shape)
    print("Label shape:", batch["label"].shape)
    print("Name:", batch["name"])
