import os
from PIL import Image
import numpy as np


def compute_mean_std(image_dir, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    """
    Compute per-channel mean and std for all images in a directory.

    Args:
        image_dir (str): Path to directory containing images.
        exts (tuple): Allowed file extensions.

    Returns:
        mean (np.ndarray): Shape (3,), RGB channel means in [0, 1].
        std  (np.ndarray): Shape (3,), RGB channel stds in [0, 1].
    """
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir} with extensions {exts}")

    n_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0  # scale to [0, 1]
        # reshape to (H*W, C)
        h, w, c = img.shape
        pixels = h * w
        img_flat = img.reshape(-1, 3)

        channel_sum += img_flat.sum(axis=0)
        channel_sum_sq += (img_flat ** 2).sum(axis=0)
        n_pixels += pixels

    mean = channel_sum / n_pixels
    var = (channel_sum_sq / n_pixels) - (mean ** 2)
    std = np.sqrt(var)

    return mean, std


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Compute per-channel mean/std for an image directory")
    # parser.add_argument("image_dir", type=str, help="Path to image directory")
    # args = parser.parse_args()
    image_dir = "/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_25_wo_borders/actual_images/train/"

    mean, std = compute_mean_std(image_dir)
    print("Per-channel mean (R, G, B):", mean)
    print("Per-channel std  (R, G, B):", std)
