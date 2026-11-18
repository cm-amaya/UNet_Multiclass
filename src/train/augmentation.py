"""Albumentations transforms for segmentation."""

import albumentations as A

transforms = A.Compose(
    [
        A.RandomResizedCrop(320, 320, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.CoarseDropout(max_holes=10, max_height=32, max_width=32, fill_value=0, p=0.5),
    ],
    is_check_shapes=False,
)
