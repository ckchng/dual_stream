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
class CustomDualMask(Dataset):
    """
    Dual-stream segmentation dataset with two masks (one per stream).

    Default layout (can be overridden via config):
      data_root/
        images/<mode>/
        images2/<mode>/            # or provide config.data_root2
        labels/<mode>/             # mask for stream1
        labels2/<mode>/            # mask for stream2 (or provide config.mask_root2)

    Returns (image1, image2, mask1, mask2) where masks are binarized {0,1}.
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
        Label('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        Label('foreground', 1, 1, 'foreground', 1, True,  False, (255, 255, 255)),
    ]

    def __init__(self, config, mode: str = 'train'):
        assert mode in ['train', 'val', 'test'], f"Unsupported mode: {mode}"

        # Allow per-mode root overrides (e.g. train_data_root / val_data_root)
        mode_root_key = f'{mode}_data_root'
        data_root = os.path.expanduser(
            getattr(config, mode_root_key, None) or config.data_root
        )

        # Images: allow separate root for stream2
        mode_root2_key = f'{mode}_data_root2'
        if getattr(config, mode_root2_key, None) or getattr(config, 'data_root2', None):
            raw2 = getattr(config, mode_root2_key, None) or config.data_root2
            data_root2 = os.path.expanduser(raw2)
            img2_dir = os.path.join(data_root2, 'images', mode)
        else:
            img2_dir = os.path.join(data_root, 'raw', mode)

        img_dir = os.path.join(data_root, 'rt', mode)

        # Masks: allow separate root for stream1 masks
        mode_mask_key = f'{mode}_mask_root'
        if getattr(config, mode_mask_key, None) or getattr(config, 'mask_root', None):
            raw_msk = getattr(config, mode_mask_key, None) or config.mask_root
            msk_dir = os.path.join(os.path.expanduser(raw_msk), mode)
        else:
            msk_dir = os.path.join(data_root, 'rt_labels', mode)
        mode_mask2_key = f'{mode}_mask_root2'
        if getattr(config, mode_mask2_key, None) or getattr(config, 'mask_root2', None):
            raw_msk2 = getattr(config, mode_mask2_key, None) or config.mask_root2
            msk2_dir = os.path.join(os.path.expanduser(raw_msk2), mode)
        else:
            msk2_dir = os.path.join(data_root, 'raw_labels', mode)

        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Image directory: {img_dir} does not exist.')
        if not os.path.isdir(img2_dir):
            raise RuntimeError(f'Image2 directory: {img2_dir} does not exist.')
        if not os.path.isdir(msk_dir):
            raise RuntimeError(f'Mask directory: {msk_dir} does not exist.')
        if not os.path.isdir(msk2_dir):
            raise RuntimeError(f'Mask2 directory: {msk2_dir} does not exist.')

        # Normalization parameters
        mean = getattr(config, 'mean', (0.30566086, 0.30566086, 0.30566086))
        std = getattr(config, 'std', (0.21072077, 0.21072077, 0.21072077))
        mean2 = getattr(config, 'mean2', (0.34827731, 0.34827731, 0.34827731))
        std2 = getattr(config, 'std2', (0.16927711, 0.16927711, 0.16927711))

        if mode == 'train':
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

            self.shared_transform = AT.Compose(
                transforms_list,
                additional_targets={'image2': 'image', 'mask2': 'mask'}
            )
        else:
            self.shared_transform = AT.Compose(
                [],
                additional_targets={'image2': 'image', 'mask2': 'mask'}
            )

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
        self.masks2 = []

        for file_name in sorted(os.listdir(img_dir)):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            img_path = os.path.join(img_dir, file_name)
            img2_path = os.path.join(img2_dir, file_name)
            msk_path = os.path.join(msk_dir, file_name)
            msk2_path = os.path.join(msk2_dir, file_name)

            if not os.path.isfile(img2_path):
                raise RuntimeError(f'Image2 not found for {img_path}: {img2_path}')
            if not os.path.isfile(msk_path):
                raise RuntimeError(f'Mask not found for {img_path}: {msk_path}')
            if not os.path.isfile(msk2_path):
                raise RuntimeError(f'Mask2 not found for {img_path}: {msk2_path}')

            self.images.append(img_path)
            self.images2.append(img2_path)
            self.masks.append(msk_path)
            self.masks2.append(msk2_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        image2 = np.asarray(Image.open(self.images2[index]).convert('RGB'))
        mask = np.asarray(Image.open(self.masks[index]).convert('L'))
        mask2 = np.asarray(Image.open(self.masks2[index]).convert('L'))

        augmented = self.shared_transform(image=image, image2=image2, mask=mask, mask2=mask2)
        image = augmented['image']
        image2 = augmented['image2']
        mask = augmented['mask']
        mask2 = augmented['mask2']

        image = self.norm1(image=image)['image']
        image2 = self.norm2(image=image2)['image']

        mask = torch.from_numpy(mask).long()
        mask2 = torch.from_numpy(mask2).long()

        mask = (mask > 0).long()
        mask2 = (mask2 > 0).long()

        return image, image2, mask, mask2
