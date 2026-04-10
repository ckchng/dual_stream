import os
from collections import namedtuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2

from utils import transforms
from .dataset_registry import register_dataset


@register_dataset
class Custom(Dataset):
    """
    Binary segmentation dataset for your RT maps:

    Images:
      data_root/images/<mode>/rt_272/*.png
    Masks:
      data_root/labels/<mode>/rt_272/*.png

    Labels:
      0 = background
      1 = foreground (anything > 0 in the mask)
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
        """
        Set in your config:
            config.data_root = "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_gray_rt_272_mask"
        """
        assert mode in ['train', 'val', 'test'], f"Unsupported mode: {mode}"

        data_root = os.path.expanduser(config.data_root)

        # Your actual structure:
        # images/train/
        # labels/train/
        img_dir = os.path.join(data_root, 'images', mode)
        msk_dir = os.path.join(data_root, 'labels', mode)

        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Image directory: {img_dir} does not exist.')
        if not os.path.isdir(msk_dir):
            raise RuntimeError(f'Mask directory: {msk_dir} does not exist.')

        if mode == 'train':
            do_crop = bool(getattr(config, 'crop_h', None) and getattr(config, 'crop_w', None) and not getattr(config, 'no_crop', False))

            tfs = [
                transforms.Scale(scale=config.scale),
                AT.RandomScale(scale_limit=config.randscale),
            ]

            if do_crop:
                tfs.extend([
                    AT.PadIfNeeded(
                        min_height=config.crop_h,
                        min_width=config.crop_w,
                        value=(114, 114, 114),
                        mask_value=0,  # background for mask
                    ),
                    AT.RandomCrop(height=config.crop_h, width=config.crop_w),
                ])

            tfs.extend([
                AT.ColorJitter(
                    brightness=config.brightness,
                    contrast=config.contrast,
                    saturation=config.saturation
                ),
                AT.HorizontalFlip(p=config.h_flip),
                # Per-channel mean (R, G, B): [0.30566086 0.30566086 0.30566086]
                # Per-channel std  (R, G, B): [0.21072077 0.21072077 0.21072077]
                AT.Normalize(
                    mean = (0.34827731, 0.34827731, 0.34827731),
                    std = (0.16927711, 0.16927711, 0.16927711)
                ),
                # AT.Normalize(
                #     mean=(0.30566086, 0.30566086, 0.30566086),
                #     std=(0.21072077, 0.21072077, 0.21072077 )
                # ),
                ToTensorV2(),
            ])

            self.transform = AT.Compose(tfs)
        else:  # 'val' or 'test'
            self.transform = AT.Compose([
                transforms.Scale(scale=config.scale),
                AT.Normalize(
                    mean = (0.34827731, 0.34827731, 0.34827731),
                    std = (0.16927711, 0.16927711, 0.16927711)
                ),
                # AT.Normalize(
                #     mean=(0.30566086, 0.30566086, 0.30566086),
                #     std=(0.21072077, 0.21072077, 0.21072077 )
                # ),
                ToTensorV2(),
            ])

        self.images = []
        self.masks = []

        # Pair image/mask by filename
        for file_name in sorted(os.listdir(img_dir)):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            img_path = os.path.join(img_dir, file_name)
            msk_path = os.path.join(msk_dir, file_name)  # same name in labels/

            if not os.path.isfile(msk_path):
                raise RuntimeError(f'Mask not found for {img_path}: {msk_path}')

            self.images.append(img_path)
            self.masks.append(msk_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image as RGB (even if original is gray, this keeps pipeline consistent)
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))

        # Load mask as single-channel
        mask = np.asarray(Image.open(self.masks[index]).convert('L'))

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']      # torch.FloatTensor [3,H,W]
        mask = augmented['mask']        # torch.* [H,W], usually uint8

        # Binary encoding: 0 background, 1 foreground
        mask = (mask > 0).long()        # torch.LongTensor in {0, 1}

        return image, mask
