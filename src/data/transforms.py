import os

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import warnings

warnings.filterwarnings("ignore")


class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


# Train Phase transformations
train_transforms = Transforms(
    transforms=A.Compose(
        [
            A.HorizontalFlip(p=0.25),
            A.ShiftScaleRotate(
                shift_limit=(-0.0625, 0.0625),
                scale_limit=(-0.1, 0.1),
                rotate_limit=(-45, 45),
                interpolation=1,
                border_mode=4,
                shift_limit_x=None,
                shift_limit_y=None,
                rotate_method="largest_box",
                p=0.5,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(16, 16),
                hole_width_range=(16, 16),
                fill=0.48,
                fill_mask=None,
                p=0.5,
            ),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(),
        ]
    )
)

# Test Phase transformations
test_transforms = Transforms(
    transforms=A.Compose(
        [
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(),
        ]
    )
)
