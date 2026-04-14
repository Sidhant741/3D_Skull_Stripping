import nibabel as nib
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from config import (
    DATASET_PATH, TASK_ID, TRAIN_VAL_TEST_SPLIT,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)
from transforms import train_transform


class SkullStrippingDataset(Dataset):
    """
    Dataset for skull stripping.

    Directory structure expected:
        dataset/
            train/
                images/   *.nii or *.nii.gz
                masks/    *_manual.nii.gz | *_ss.nii.gz | *_mask.nii.gz | *_seg.nii.gz
            valid/
                images/
                masks/
            test/
                images/
                masks/

    Axis convention
    ---------------
    NIfTI voxel array from nibabel is (x, y, z) = (R-L, A-P, S-I).
    We move the A-P axis (axis 1) to position 0 so the model's depth
    dimension D spans the anterior-posterior direction — this gives the
    corpus callosum and other thin sagittal structures the most slices
    and prevents them from being collapsed at the bottleneck.

        (x, y, z)  →  moveaxis(1, 0)  →  (y, x, z)  →  expand  →  (1, y, x, z)
         R-L A-P S-I                       A-P R-L S-I              C  D   H  W
    """

    def __init__(self, root_dir, mode='train', transform=None):
        self.mode      = mode
        self.transform = transform
        self.image_dir = os.path.join(root_dir, mode, 'images')
        self.mask_dir  = os.path.join(root_dir, mode, 'masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Resolve mask filename
        if img_name.endswith('.nii.gz'):
            base      = img_name[:-7]
            suffixes  = ['_manual.nii.gz', '_ss.nii.gz', '_mask.nii.gz', '_seg.nii.gz']
        elif img_name.endswith('.nii'):
            base      = img_name[:-4]
            suffixes  = ['_manual.nii', '_ss.nii', '_mask.nii', '_seg.nii']
        else:
            raise ValueError(f"Unexpected extension: {img_name}")

        mask_path = None
        for suffix in suffixes:
            candidate = os.path.join(self.mask_dir, base + suffix)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            tried = ', '.join(base + s for s in suffixes)
            raise FileNotFoundError(f"No mask found for {img_name}. Tried: {tried}")

        # Load — nibabel returns (x, y, z)
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask  = nib.load(mask_path).get_fdata().astype(np.float32)

        # Move A-P axis (1) to depth position (0): (x,y,z) → (y,x,z)
        image = np.moveaxis(image, 1, 0)
        mask  = np.moveaxis(mask,  1, 0)

        # Add channel dim → (1, D, H, W)
        image = np.expand_dims(image, axis=0)
        mask  = np.expand_dims(mask,  axis=0)

        sample = {'name': img_name, 'image': image, 'label': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_train_val_test_Dataloaders(train_transforms=None, val_transforms=None, test_transforms=None):
    train_ds = SkullStrippingDataset(DATASET_PATH, mode='train', transform=train_transforms)
    val_ds   = SkullStrippingDataset(DATASET_PATH, mode='valid', transform=val_transforms)
    test_ds  = SkullStrippingDataset(DATASET_PATH, mode='test',  transform=test_transforms)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE,   shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=TEST_BATCH_SIZE,  shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    ds    = SkullStrippingDataset(DATASET_PATH, mode='train', transform=train_transform)
    loader = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    batch  = next(iter(loader))
    print("Batch keys  :", list(batch.keys()))
    print("Image shape :", batch["image"].shape)   # expect (B, 1, D, H, W)
    print("Label shape :", batch["label"].shape)
    print("Name        :", batch["name"])