import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2
from utils import transforms


class TestDataset(Dataset):
    def __init__(self, config):
        data_folder = os.path.expanduser(config.test_data_folder)

        if not os.path.isdir(data_folder):
            raise RuntimeError(f'Test image directory: {data_folder} does not exist.')

        self.transform = AT.Compose([
            transforms.Scale(scale=config.scale, is_testing=True),
            AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.images = []
        self.img_names = []

        for file_name in os.listdir(data_folder):
            self.images.append(os.path.join(data_folder, file_name))
            self.img_names.append(file_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        img_name = self.img_names[index]

        # Perform augmentation and normalization
        augmented = self.transform(image=image)
        image_aug = augmented['image']

        return image, image_aug, img_name


class DualMaskTestDataset(Dataset):
    """Dual-input test dataset aligned with ``CustomDualMask`` layout.

    Expected folder structure (defaults):
      data_root/
        images/test/
        images2/test/ or actual_images/test/ (or override via data_root2/test_data_folder2)

    Returns (img1_np, img2_np, img1_tensor, img2_tensor, img_name).
    """

    def __init__(self, config, mode: str = 'test'):
        assert mode in ['test', 'val', 'train'], f"Unsupported mode: {mode}"

        # Resolve primary root for stream 1
        data_root = os.path.expanduser(getattr(config, 'data_root', None) or '')
        if not data_root and not getattr(config, 'test_data_folder', None):
            raise RuntimeError('Provide either `data_root` or `test_data_folder` for DualMaskTestDataset.')

        if getattr(config, 'test_data_folder', None):
            img_dir = os.path.expanduser(config.test_data_folder)
        else:
            img_dir = os.path.join(data_root, 'images', mode)

        # Resolve root for stream 2 with fallbacks similar to CustomDualMask
        if getattr(config, 'test_data_folder2', None):
            img2_dir = os.path.expanduser(config.test_data_folder2)
        elif getattr(config, 'data_root2', None):
            data_root2 = os.path.expanduser(config.data_root2)
            img2_dir = os.path.join(data_root2, 'images', mode)
        else:
            img2_dir = os.path.join(data_root, 'actual_images', mode)

        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Image directory: {img_dir} does not exist.')
        if not os.path.isdir(img2_dir):
            raise RuntimeError(f'Image2 directory: {img2_dir} does not exist.')

        mean = getattr(config, 'mean', (0.30566086, 0.30566086, 0.30566086))
        std = getattr(config, 'std', (0.21072077, 0.21072077, 0.21072077))
        mean2 = getattr(config, 'mean2', (0.34827731, 0.34827731, 0.34827731))
        std2 = getattr(config, 'std2', (0.16927711, 0.16927711, 0.16927711))

        # Keep transforms consistent with CustomDualMask test-time path
        self.shared_transform = AT.Compose([], additional_targets={'image2': 'image'})

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
        self.img_names = []

        for file_name in sorted(os.listdir(img_dir)):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            img_path = os.path.join(img_dir, file_name)
            img2_path = os.path.join(img2_dir, file_name)

            if not os.path.isfile(img2_path):
                raise RuntimeError(f'Image2 not found for {img_path}: {img2_path}')

            self.images.append(img_path)
            self.images2.append(img2_path)
            self.img_names.append(file_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        image2 = np.asarray(Image.open(self.images2[index]).convert('RGB'))
        img_name = self.img_names[index]

        augmented = self.shared_transform(image=image, image2=image2)
        image = augmented['image']
        image2 = augmented['image2']

        image_aug = self.norm1(image=image)['image']
        image2_aug = self.norm2(image=image2)['image']

        return image, image2, image_aug, image2_aug, img_name