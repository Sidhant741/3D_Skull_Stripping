from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureChannelFirstd
)


#Transforms to be applied on training instances
train_transform = Compose(
    [   
        # EnsureChannelFirstd(keys=["image", "label"]),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=3),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=4),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        ToTensord(keys=['image', 'label'])
    ]
)

#Cuda version of "train_transform"
train_transform_cuda = Compose(
    [   
        # EnsureChannelFirstd(keys=["image", "label"]),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=3),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=4),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)

#Transforms to be applied on validation instances
val_transform = Compose(
    [   
        # EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        ToTensord(keys=['image', 'label'])
    ]
)

#Cuda version of "val_transform"
val_transform_cuda = Compose(
    [   
        # EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)
