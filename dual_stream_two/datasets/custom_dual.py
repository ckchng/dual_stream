import os
from collections import namedtuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2

from .dataset_registry import register_dataset


@register_dataset
class CustomDual(Dataset):
    """
    Dual stream segmentation dataset.
    
    Expected structure if config.data_root2 is NOT provided:
      data_root/
        images/
          train/
          val/
        images2/   <-- Second stream
          train/
          val/
        labels/
          train/
          val/

    If config.data_root2 IS provided, it will be used as the root for the second stream images.
    """

    Label = namedtuple('Label', [
        'name',
        'id',
        'trainId',
        'category',
        'categoryId',
        'hasInstances',
        'ignoreInEval',
        'color',
    ])

    labels = [
        #   name         id  trainId  category      catId hasInst ignore color
        Label('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        Label('foreground', 1, 1, 'foreground', 1, True,  False, (255, 255, 255)),
    ]

    def __init__(self, config, mode: str = 'train'):
        assert mode in ['train', 'val', 'test'], f"Unsupported mode: {mode}"

        data_root = os.path.expanduser(config.data_root)
        
        # Check if a second data root is provided for the second stream
        if hasattr(config, 'data_root2') and config.data_root2 is not None:
            data_root2 = os.path.expanduser(config.data_root2)
            img2_dir = os.path.join(data_root2, 'images', mode)
        else:
            # Default to 'images2' in the same data_root
            img2_dir = os.path.join(data_root, 'actual_images', mode)

        img_dir = os.path.join(data_root, 'images', mode)
        msk_dir = os.path.join(data_root, 'labels', mode)

        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Image directory: {img_dir} does not exist.')
        if not os.path.isdir(img2_dir):
            raise RuntimeError(f'Image2 directory: {img2_dir} does not exist.')
        if not os.path.isdir(msk_dir):
            raise RuntimeError(f'Mask directory: {msk_dir} does not exist.')

        # Get mean and std from config or use defaults
        mean = getattr(config, 'mean', (0.30566086, 0.30566086, 0.30566086))
        std = getattr(config, 'std', (0.21072077, 0.21072077, 0.21072077))
        mean2 = getattr(config, 'mean2', (0.34827731, 0.34827731, 0.34827731))
        std2 = getattr(config, 'std2', (0.16927711, 0.16927711, 0.16927711))

        if mode == 'train':
            # Shared geometric and color transforms
            transforms_list = [
                AT.RandomScale(scale_limit=config.randscale),
            ]

            if config.crop_h is not None and config.crop_w is not None:
                transforms_list.extend([
                    AT.PadIfNeeded(
                        min_height=config.crop_h,
                        min_width=config.crop_w,
                        value=(114, 114, 114),
                        mask_value=0,
                    ),
                    AT.RandomCrop(height=config.crop_h, width=config.crop_w),
                ])

            transforms_list.extend([
                AT.ColorJitter(
                    brightness=config.brightness,
                    contrast=config.contrast,
                    saturation=config.saturation
                ),
                AT.HorizontalFlip(p=config.h_flip),
            ])

            self.shared_transform = AT.Compose(transforms_list, additional_targets={'image2': 'image'})
        else:
            self.shared_transform = AT.Compose([], additional_targets={'image2': 'image'})

        # Separate normalization and tensor conversion
        self.norm1 = AT.Compose([
            AT.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        
        self.norm2 = AT.Compose([
            AT.Normalize(mean=mean2, std=std2),
            ToTensorV2(),
        ])

        self.images = []
        self.images2 = []
        self.masks = []

        for file_name in sorted(os.listdir(img_dir)):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            img_path = os.path.join(img_dir, file_name)
            img2_path = os.path.join(img2_dir, file_name)
            msk_path = os.path.join(msk_dir, file_name)

            if not os.path.isfile(img2_path):
                raise RuntimeError(f'Image2 not found for {img_path}: {img2_path}')
            if not os.path.isfile(msk_path):
                raise RuntimeError(f'Mask not found for {img_path}: {msk_path}')

            self.images.append(img_path)
            self.images2.append(img2_path)
            self.masks.append(msk_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        image2 = np.asarray(Image.open(self.images2[index]).convert('RGB'))
        mask = np.asarray(Image.open(self.masks[index]).convert('L'))

        # Apply shared transforms (geometric + color)
        augmented = self.shared_transform(image=image, image2=image2, mask=mask)
        
        image = augmented['image']
        image2 = augmented['image2']
        mask = augmented['mask']

        # Apply separate normalization
        image = self.norm1(image=image)['image']
        image2 = self.norm2(image=image2)['image']

        mask = torch.from_numpy(mask).long()
        mask = (mask > 0).long()

        return image, image2, mask
