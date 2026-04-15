from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    RandAffined,
    RandGaussianNoised,
)

# Shared augmentation pipeline (without the final ToTensord)
_TRAIN_AUGS = [
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
    RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
]

# Transforms to be applied on training instances
train_transform = Compose(
    _TRAIN_AUGS + [ToTensord(keys=['image', 'label'])]
)

# CUDA version of train_transform — identical augmentations, tensor lands on GPU
train_transform_cuda = Compose(
    _TRAIN_AUGS + [ToTensord(keys=['image', 'label'], device='cuda')]
)

# Transforms to be applied on validation instances
val_transform = Compose([
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ToTensord(keys=['image', 'label']),
])

# CUDA version of val_transform
val_transform_cuda = Compose([
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ToTensord(keys=['image', 'label'], device='cuda'),
])