#!/usr/bin/env python3
"""
Tile (608x608) -> YOLO (Ultralytics) -> save only tiles with NO detections ≥ conf.
Edge tiles are adjusted inward (no padding). Filenames encode top-left (x,y).

Debug mode: set your params in the CONFIG dict below.
Flip USE_ARGPARSE=True later to switch to CLI flags without changing anything else.
"""

import os
import glob
from typing import List

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.HT_utils import _hough_accumulate_intensity_dh, endpoints_to_rho_theta_dh, rho_theta_to_indices_dh, _make_params_dh, hough_bruteforce_intensity_numba_dh
from utils.HT_utils import _hough_accumulate_intensity, endpoints_to_rho_theta, rho_theta_to_indices, _make_params, hough_bruteforce_intensity_numba

# import rawpy, cv2
import cv2
import random

import argparse
from types import SimpleNamespace
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from predict_wrapper import load_model, process_image as process_image_segmentation
    from utils import transforms
    from utils.HT_utils import hough_bruteforce_intensity_numba_dh
    import albumentations as AT
    from albumentations.pytorch import ToTensorV2
except ImportError:
    # If running from root, try direct import
    from predict_wrapper import load_model, process_image as process_image_segmentation
    from utils import transforms
    from utils.HT_utils import hough_bruteforce_intensity_numba_dh
    import albumentations as AT
    from albumentations.pytorch import ToTensorV2

def check_image_bit_depth(img):
    """
    Check the bit depth of an image and return information about it.
    
    Args:
        img: numpy array representing the image
        
    Returns:
        dict: Information about the image bit depth
    """
    if img is None:
        return None
    
    dtype = img.dtype
    bit_depth_info = {
        'dtype': str(dtype),
        'bit_depth': None,
        'max_value': None,
        'actual_max': np.max(img),
        'actual_min': np.min(img),
        'is_8_bit': False,
        'is_16_bit': False,
        'needs_conversion': False
    }
    
    if dtype == np.uint8:
        bit_depth_info.update({
            'bit_depth': 8,
            'max_value': 255,
            'is_8_bit': True,
            'needs_conversion': False
        })
    elif dtype == np.uint16:
        bit_depth_info.update({
            'bit_depth': 16,
            'max_value': 65535,
            'is_16_bit': True,
            'needs_conversion': True  # YOLO expects 8-bit
        })
    elif dtype == np.float32 or dtype == np.float64:
        bit_depth_info.update({
            'bit_depth': f'float ({dtype})',
            'max_value': 1.0 if np.max(img) <= 1.0 else np.max(img),
            'needs_conversion': True
        })
    else:
        bit_depth_info.update({
            'bit_depth': f'unknown ({dtype})',
            'needs_conversion': True
        })
    
    return bit_depth_info

def convert_to_8bit(img, bit_info):
    """
    Convert image to 8-bit format suitable for YOLO.
    
    Args:
        img: input image
        bit_info: bit depth information from check_image_bit_depth
        
    Returns:
        numpy array: 8-bit image
    """
    if bit_info['is_8_bit']:
        return img
    
    if bit_info['is_16_bit']:
        # Convert 16-bit to 8-bit by dividing by 256
        return (img / 256).astype(np.uint8)
    
    if 'float' in str(img.dtype):
        if np.max(img) <= 1.0:
            # Normalized float, scale to 0-255
            return (img * 255).astype(np.uint8)
        else:
            # Non-normalized float, clip and convert
            return np.clip(img, 0, 255).astype(np.uint8)
    
    # For other types, try to convert safely
    return np.clip(img, 0, 255).astype(np.uint8)



# =========================
# 🔧 DEBUG (no argparse)
# =========================
USE_ARGPARSE = True  # Default to True for CLI usage

# Merged Config: Defaults for argparse, or used directly if USE_ARGPARSE=False
CONFIG = {
    # Script behavior
    "weights": "/home/ckchng/Dropbox/SDA_ODA_SD/trained_models/YOLO-v8n/best.pt", # Kept for compatibility if needed, but likely unused
    "img_dir": "/media/ckchng/internal2TB/RAW_DATA/FIREOPAL000/",
    "img_dirs": ['/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL000',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL016',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL017',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL019',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL020',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL021',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL023',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL024',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL025',
                 '/media/ckchng/internal2TB/FILTERED_IMAGES/FIREOPAL026'],
    "label_dir": "/home/ckchng/Dropbox/SDA_ODA_SD/labels/",
    "out_dir": "/home/ckchng/Documents/SDA_ODA/LMA_data/background_patches_with_new_model/",
    "tile": 288,
    "overlap": 0.0,
    "conf": 0.05,
    "batch": 128,
    "imgsz": 288,
    "device": 0,
    "half": True,
    "exts": ".npy",
    "filter_class_ids": None,
    "max_saved": 200000,
    "random_seed": 42,
    "k_per_image": 25, # NEW: number of random crops per image (0 = disabled, use all tiles)
    
    # Model parameters (formerly HARD_CONFIG)
    "model": "bisenetv2dualmaskguidedv2",
    "num_classes": 2,
    "use_aux": True,
    "ckpt": "/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_snr_1_25_longer_wider/best.pth",
    "scale": 1.0,
    "mean": (0.30566086, 0.30566086, 0.30566086),
    "std": (0.21072077, 0.21072077, 0.21072077),
    "mean2": (0.34827731, 0.34827731, 0.34827731),
    "std2": (0.16927711, 0.16927711, 0.16927711),
    "tile_stride": 288, # NOTE: This overlaps with 'overlap' parameter? keeping both for now
    "num_angles": 192,
    "num_rhos": 416,
    "encoder": None,
    "decoder": None
}

def _parse_args_or_config():
    if not USE_ARGPARSE:
        class _A: pass
        a = _A()
        for k, v in CONFIG.items():
            setattr(a, k, v)
        # Ensure we have tile_size aliased to tile if not present
        if not hasattr(a, 'tile_size'):
            a.tile_size = a.tile
        return a

    # ---- argparse path (enable by setting USE_ARGPARSE=True) ----
    import argparse
    
    def parse_floats(values):
        if values is None: return None
        if isinstance(values, (tuple, list)): return values # already parsed default
        # If string "0.1 0.2 0.3"
        return tuple(map(float, values.replace(',', ' ').split()))

    p = argparse.ArgumentParser(description="Save only empty tiles from large images.")
    
    # Core script args
    p.add_argument("--weights", type=str, default=CONFIG["weights"])
    p.add_argument("--img_dir", type=str, default=CONFIG["img_dir"])
    p.add_argument("--img_dirs", type=str, nargs="+", default=CONFIG["img_dirs"], help="List of image directories")
    p.add_argument("--label_dir", type=str, default=CONFIG["label_dir"], help="Directory containing _label.mat files")
    p.add_argument("--out_dir", type=str, default=CONFIG["out_dir"])
    p.add_argument("--tile", type=int, default=CONFIG["tile"])
    p.add_argument("--overlap", type=float, default=CONFIG["overlap"])
    p.add_argument("--conf", type=float, default=CONFIG["conf"])
    p.add_argument("--batch", type=int, default=CONFIG["batch"])
    p.add_argument("--imgsz", type=int, default=CONFIG["imgsz"])
    p.add_argument("--device", default=CONFIG["device"])
    p.add_argument("--no-half", action="store_true")
    p.add_argument("--exts", type=str, nargs="+", default=CONFIG["exts"] if isinstance(CONFIG["exts"], list) else [CONFIG["exts"]])
    p.add_argument("--filter-class-ids", type=int, nargs="+", default=CONFIG["filter_class_ids"])
    p.add_argument("--max_saved", type=int, default=CONFIG["max_saved"])
    p.add_argument("--random_seed", type=int, default=CONFIG["random_seed"])
    p.add_argument("--k_per_image", type=int, default=CONFIG["k_per_image"], help="Number of random crops per image (0 = use sliding window)")

    # Model args
    p.add_argument("--model", type=str, default=CONFIG["model"])
    p.add_argument("--num_classes", type=int, default=CONFIG["num_classes"])
    p.add_argument("--use_aux", action="store_true", default=CONFIG["use_aux"])
    p.add_argument("--ckpt", type=str, default=CONFIG["ckpt"])
    p.add_argument("--scale", type=float, default=CONFIG["scale"])
    p.add_argument("--mean", type=float, nargs=3, default=CONFIG["mean"])
    p.add_argument("--std", type=float, nargs=3, default=CONFIG["std"])
    p.add_argument("--mean2", type=float, nargs=3, default=CONFIG["mean2"])
    p.add_argument("--std2", type=float, nargs=3, default=CONFIG["std2"])
    p.add_argument("--num_angles", type=int, default=CONFIG["num_angles"])
    p.add_argument("--num_rhos", type=int, default=CONFIG["num_rhos"])
    p.add_argument("--encoder", type=str, default=CONFIG["encoder"])
    p.add_argument("--decoder", type=str, default=CONFIG["decoder"])
    p.add_argument("--tile_stride", type=int, default=CONFIG["tile_stride"])
    
    args = p.parse_args()
    
    args.half = not getattr(args, "no_half", False)
    args.tile_size = args.tile # Alias
    
    return args




# =========================
# Core helpers
# =========================


def extract_random_crops(img, k, tile_size, max_rho, itheta, irho):
    """
    Extract k random crops from img and prepare them for inference.
    Similar to tile_image but random locations.
    """
    h_img, w_img, _ = img.shape
    tiles = []
    
    # Generate random top-left coordinates
    # Ensure they fit in the image
    max_y = max(0, h_img - tile_size)
    max_x = max(0, w_img - tile_size)
    
    ys = np.random.randint(0, max_y + 1, size=k)
    xs = np.random.randint(0, max_x + 1, size=k)
    
    for y, x in zip(ys, xs):
        # Adjust dimensions if necessary (requested by user to match tile_image logic)
        if y + tile_size > h_img:
            y = h_img - tile_size
        if x + tile_size > w_img:
            x = w_img - tile_size
        
        # Ensure non-negative (handles images smaller than tile_size by taking from 0)
        if y < 0: y = 0
        if x < 0: x = 0

        # Crop
        tile = img[y : y + tile_size, x : x + tile_size]
        
        # perform RT here (Replicated from predict_wrapper.py)
        rt_map, theta_deg, rhos = hough_bruteforce_intensity_numba_dh(tile[:, :, 0], max_rho/2, theta_res_deg=itheta, rho_res=irho)
        
        rt_map = rt_map - rt_map.min()
        if rt_map.max() > 0:
            rt_map = rt_map / rt_map.max() * 255
        rt_map = rt_map.astype(np.uint8)
        
        # pad the rt_map to be divisible by 32
        p_h = (32 - rt_map.shape[0] % 32) % 32
        p_w = (32 - rt_map.shape[1] % 32) % 32
        rt_map = cv2.copyMakeBorder(rt_map, 0, p_h, 0, p_w, cv2.BORDER_CONSTANT, value=0)
        
        # pad rt_map to 3 channels
        rt_map = rt_map[:, :, np.newaxis]
        rt_map = np.repeat(rt_map, 3, axis=2)
        th, tw, _ = rt_map.shape

        scaled_tile = tile.copy()

        zero_mask = scaled_tile == 0
        scaled_tile = scaled_tile - scaled_tile.min()
        if scaled_tile.max() > 0:
            scaled_tile = scaled_tile / scaled_tile.max() * 255.0
        scaled_tile[zero_mask] = 0
        scaled_tile = scaled_tile.astype(np.uint8)

        tiles.append(
            {
                'ori_img': tile,
                "scaled_img": scaled_tile,
                "rt_map": rt_map,  
                "y": y,
                "x": x,
                "h": th,
                "w": tw,
                "rhos": rhos,
                "thetas": theta_deg 
            }
        )
    return tiles

def process_random_crops(image_rgb, model, args, device, k):
    """
    Takes an RGB image (numpy), extracts k random crops, runs inference, and returns output masks.
    """
    
    # Prepare params
    itheta = 180 / args.num_angles
    max_rho = np.hypot(args.tile_size, args.tile_size) + 1
    irho = max_rho / (args.num_rhos - 1)
    
    # Prepare transforms
    is_dual = 'dual' in args.model.lower()
    is_maskguided = 'maskguided' in args.model.lower()
    
    if is_dual:
        scale_transform = transforms.Scale(scale=args.scale, is_testing=True)
        norm_transform1 = AT.Compose([
            AT.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])
        norm_transform2 = AT.Compose([
            AT.Normalize(mean=args.mean2, std=args.std2),
            ToTensorV2(),
        ])
    else:
        # Fallback for single stream if used
        tr = AT.Compose([
             transforms.Scale(scale=args.scale, is_testing=True),
            transforms.PadPairToMax(padding_value=0, mask_value=0),
            AT.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])
    
    # 1. Random Crops
    tiles = extract_random_crops(image_rgb, k=k, tile_size=args.tile_size, max_rho=max_rho, itheta=itheta, irho=irho)
    
    results = []
    
    # 2. Inference in batches
    for idx in range(0, len(tiles), args.batch_size):
        batch_tiles = tiles[idx : idx + args.batch_size]

        images_aug = []
        images2_aug = []
        
        for t in batch_tiles:
            if is_dual:
                res1 = scale_transform(image=t["rt_map"])
                res2 = scale_transform(image=t["scaled_img"])
                img1_scaled = res1["image"]
                img2_scaled = res2["image"]
                
                final1 = norm_transform1(image=img1_scaled)["image"]
                final2 = norm_transform2(image=img2_scaled)["image"]
                
                images_aug.append(final1)
                images2_aug.append(final2)
            else:
                # Single stream logic if needed
                augmented = tr(image=t["rt_map"])
                images_aug.append(augmented["image"])

        # batch_tensor = torch.stack(images_aug).to(device=device, dtype=torch.float32)
        # Check if list is not empty before stacking
        if not images_aug:
             continue
             
        batch_tensor = torch.stack(images_aug).to(device=device, dtype=torch.float32)
        batch_tensor2 = None
        if len(images2_aug) > 0:
            batch_tensor2 = torch.stack(images2_aug).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            if is_dual:
                logits = model(batch_tensor, batch_tensor2)
            else:
                logits = model(batch_tensor)

            logits_main = logits
            logits_s2 = None

            if isinstance(logits, (tuple, list)):
                if is_maskguided and len(logits) >= 2:
                    logits_main, logits_s2 = logits[0], logits[1]
                else:
                    logits_main = logits[0]
            
            if args.num_classes == 2:
                preds_rt = logits_main.argmax(dim=1)
                preds_streak = logits_s2.max(dim=1)[1] if logits_s2 is not None else None
            elif args.num_classes == 1:
                preds_rt = (logits_main.sigmoid() > 0.5).squeeze(1).long()
                preds_streak = (logits_s2.sigmoid() > 0.5).squeeze(1).long() if logits_s2 is not None else None

        preds_rt = preds_rt.cpu().numpy()
        preds_streak = preds_streak.cpu().numpy() if preds_streak is not None else [None]*len(preds_rt)
        
        for i, tile in enumerate(batch_tiles):
            results.append({
                'tile_idx': idx + i,
                'tile_info': tile,
                'pred_rt': preds_rt[i],         # Mask 1 (RT domain)
                'pred_streak': preds_streak[i]  # Mask 2 (Spatial domain)
            })
            
    return results


def check_label_exists(img_path: str, label_dir: str) -> bool:
    """
    Check if a corresponding label file exists for the given image.
    
    Args:
        img_path: Path to the image file
        label_dir: Directory containing label files
        
    Returns:
        bool: True if label file exists, False otherwise
    """
    if not label_dir or not os.path.exists(label_dir):
        return False
    
    # Get image filename without extension
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    
    # Create expected label filename
    label_filename = f"{img_basename}_label.mat"
    label_path = os.path.join(label_dir, label_filename)
    
    # Check if label file exists (case-insensitive for robustness)
    if os.path.exists(label_path):
        return True
    
    # Also check subdirectories recursively
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.lower() == label_filename.lower():
                return True
    
    return False


def filter_unlabeled_images(img_paths: List[str], label_dir: str) -> tuple:
    """
    Filter out images that have corresponding label files.
    
    Args:
        img_paths: List of image file paths
        label_dir: Directory containing label files
        
    Returns:
        tuple: (unlabeled_images, skipped_images)
    """
    if not label_dir:
        print("[INFO] No label directory specified, processing all images.")
        return img_paths, []
    
    unlabeled_images = []
    skipped_images = []
    
    print(f"[INFO] Checking for existing labels in: {label_dir}")
    
    for img_path in img_paths:
        if check_label_exists(img_path, label_dir):
            skipped_images.append(img_path)
            print(f"[SKIP] Found label for: {os.path.basename(img_path)}")
        else:
            unlabeled_images.append(img_path)
    
    print(f"[INFO] Images with labels (skipped): {len(skipped_images)}")
    print(f"[INFO] Images without labels (will process): {len(unlabeled_images)}")
    
    return unlabeled_images, skipped_images

def make_grid_starts(length: int, tile: int, overlap: float) -> List[int]:
    """Candidate starts for a sliding window of width=tile.
    Ensures last start = max(length - tile, 0) so the final crop fits exactly."""
    step = int(tile * (1.0 - overlap))
    step = max(1, step)
    starts = list(range(0, max(length - tile, 0) + 1, step))
    last = max(length - tile, 0)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def crop_tile_exact(img, x1, y1, tile):
    """Crop exactly tile x tile. Adjust x1/y1 inward if needed (no padding)."""
    H, W = img.shape[:2]
    x1 = max(0, min(x1, W - tile))
    y1 = max(0, min(y1, H - tile))
    return img[y1:y1 + tile, x1:x1 + tile], x1, y1


def save_npy(path: str, img_arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, img_arr)


def process_image(
    img_path: str,
    out_dir: str,
    tile: int = 608,
    overlap: float = 0.0,
    conf_thres: float = 0.05,
    batch: int = 8,
    device = 0,
    half: bool = True,
    imgsz: int = 608,
    filter_class_ids=None,
    max_to_save: int | None = None,   # NEW: remaining quota for this call
) -> int:                              # NEW: return number of tiles saved from this image
    ext = os.path.splitext(img_path)[1].lower()
    
    img = np.load(img_path) if ext == ".npy" else None
    
    # plt.imshow(img)
    # plt.show()
    
    if img is None:
        print(f"[WARN] Failed to read: {img_path}")
        return 0

    H, W = img.shape[:2]
    if W < tile or H < tile:
        print(f"[SKIP] {os.path.basename(img_path)} smaller than tile {tile} (W={W}, H={H}).")
        return 0 

    xs = make_grid_starts(W, tile, overlap)
    ys = make_grid_starts(H, tile, overlap)

    tiles, meta = [], []  # list of crops and (x,y)
    for y in ys:
        for x in xs:
            crop, xa, ya = crop_tile_exact(img, x, y, tile)
            tiles.append(crop)
            meta.append((xa, ya))

    saved_from_image = 0
    # Guard: nothing to save if quota is 0
    if max_to_save is not None and max_to_save <= 0:
        return 0
    
    # Inference in batches
    for i in range(0, len(tiles), batch):
        # Early stop if quota reached
        if max_to_save is not None and saved_from_image >= max_to_save:
            return saved_from_image

        batch_tiles = tiles[i:i + batch]
        results = model(
            batch_tiles,
            imgsz=imgsz,
            conf=conf_thres,
            device=device,
            half=bool(half and torch.cuda.is_available()),
            verbose=False
            # predictor=DetectionPredictor,   # <-- key line
            )
        
    
        for j, res in enumerate(results):
            if max_to_save is not None and saved_from_image >= max_to_save:
                return saved_from_image
            x, y = meta[i + j]

            # Count detections >= conf; optionally restrict to certain classes
            num_boxes = 0
            if res and hasattr(res, "boxes") and res.boxes is not None and res.boxes.data is not None:
                if filter_class_ids is None:
                    num_boxes = res.boxes.data.shape[0]
                    # draw boxes on batch_tiles[j] for visualization
                    # for box in res.boxes:
                    #     x1, y1, x2, y2 = map(int, box.xyxy[0])
                    #     conf = box.conf[0].item()
                    #     cls_id = int(box.cls[0].item())
                    #     cv2.rectangle(batch_tiles[j], (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #     cv2.putText(batch_tiles[j], f"{cls_id} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # plt.imshow(batch_tiles[j])
                    # plt.show()
                    # plt.close('all')
                else:
                    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
                    confs = res.boxes.conf.detach().cpu().numpy()
                    mask = np.isin(cls, np.array(filter_class_ids, dtype=int)) & (confs >= conf_thres)
                    num_boxes = int(mask.sum())

            if num_boxes == 0:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                out_name = f"{stem}_x{x}_y{y}.npy"
                out_path = os.path.join(out_dir, out_name)
                save_npy(out_path, batch_tiles[j])
                saved_from_image += 1

    return saved_from_image

def main():
    args = _parse_args_or_config()

    # Normalize exts to a list (supports string ".NEF" or "NEF, TIF" or list)
    def _normalize_exts(exts_val):
        if isinstance(exts_val, str):
            parts = [p.strip() for p in exts_val.replace(",", " ").split() if p.strip()]
        else:
            parts = list(exts_val)
        parts = [e if e.startswith(".") else f".{e}" for e in parts]
        return parts

    exts = _normalize_exts(getattr(args, "exts", [".nef"]))
    
    # Use img_dirs if provided; otherwise fall back to single img_dir
    img_dirs = []
    if getattr(args, "img_dirs", None):
        img_dirs = list(args.img_dirs)
    elif getattr(args, "img_dir", None):
        img_dirs = [args.img_dir]
    else:
        print("[ERROR] No image directories specified. Set img_dir or img_dirs.")
        return

    # Gather images from all directories
    # img_paths = []
    # for d in img_dirs:
    #     for ext in exts:
    #         img_paths.extend(glob.glob(os.path.join(d, f"*{ext}")))
    # img_paths = sorted(img_paths)

    # filtered_paths = {}
    img_paths = []
    for d in img_dirs:
        for root, dirs, files in os.walk(d):
            for filename in files:
                if filename.endswith(".npy"):
                    img_name = os.path.splitext(filename)[0]
                    img_path = os.path.join(root, filename)
                    img_paths.append(img_path)

    if not img_paths:
        print(f"[ERROR] No images found in {img_dirs} with exts {exts}")
        return
    
    # Filter out images that have corresponding labels
    unlabeled_img_paths, skipped_img_paths = filter_unlabeled_images(
        img_paths, 
        getattr(args, 'label_dir', None)
    )
    
    if not unlabeled_img_paths:
        print("[INFO] All images have corresponding labels. Nothing to process.")
        return

    # Optional: randomize processing order for unlabeled images
    if hasattr(args, "random_seed") and args.random_seed is not None:
        random.seed(args.random_seed)
        print(f"[INFO] Using random seed: {args.random_seed} for shuffling.")
    random.shuffle(unlabeled_img_paths)

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Initialize the segmentation model
    # Explicitly check and print GPU status
    print(f"----------------------------------------------------------------")
    print(f"[INFO] Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA Available: Yes")
        print(f"[INFO] GPU Device 0: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[WARN] CUDA Available: NO")
    
    device_str = "cpu"
    if torch.cuda.is_available() and str(args.device) != 'cpu':
        device_str = f"cuda:{args.device}"
    
    device = torch.device(device_str)
    print(f"[INFO] Selected Processing Device: {device}")
    print(f"----------------------------------------------------------------")

    print(f"[INFO] Loading model {args.model} to {device}...")
    
    # We can pass `args` directly because we ensured it has all the necessary fields from the merged config
    # However, load_model expects `batch_size` which maps to `args.batch` in our arg set.
    # Let's alias it.
    args.batch_size = args.batch
    
    seg_model = load_model(args, device)

    print(f"[INFO] Searched {len(img_dirs)} directories:")
    for d in img_dirs:
        print(f"       - {d}")
    print(f"[INFO] Found {len(img_paths)} images across all dirs.")
    print(f"[INFO] Images to process (unlabeled): {len(unlabeled_img_paths)}")
    print(f"[INFO] tile={args.tile}, overlap={args.overlap}, conf={args.conf}, batch={args.batch}, imgsz={args.imgsz}")
    print(f"[INFO] filter_class_ids={getattr(args, 'filter_class_ids', None)}")
    cap = getattr(args, "max_saved", None)
    print(f"[INFO] Max saved tiles: {cap if cap is not None else 'no cap'}")
    print(f"[INFO] Saving empty tiles to: {args.out_dir}")

    saved_total = 0
    os.makedirs(args.out_dir, exist_ok=True)
    for k, p in enumerate(unlabeled_img_paths, 1):
        if args.max_saved is not None and saved_total >= args.max_saved:
            return saved_total
        if cap is not None and saved_total >= cap:
            print(f"[INFO] Reached cap of {cap} saved tiles. Stopping.")
            break

        # only proceed if the image name has a corresponding file in filtered_paths
        img_stem = os.path.splitext(os.path.basename(p))[0]
        # img_stem = img_stem.split('/')[-1]  # already basename
        
        remaining = None if cap is None else (cap - saved_total)
        print(f"[{k}/{len(unlabeled_img_paths)}] {os.path.basename(p)} (remaining quota: {remaining if remaining is not None else '∞'})")

        # Load image
        try:
            img = np.load(p)
        except Exception as e:
            print(f"[WARN] Error loading {p}: {e}")
            continue

        if img is None:
            continue
            
        # Ensure RGB 3-channel
        if len(img.shape) == 2:
            img_rgb = np.stack((img,)*3, axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img_rgb = np.repeat(img, 3, axis=2)
        else:
            img_rgb = img

        # Check dimensions
        H, W = img_rgb.shape[:2]
        if H < args.tile_size or W < args.tile_size:
             print(f"[SKIP] {os.path.basename(p)} smaller than tile {args.tile_size}.")
             continue

        # Process with segmentation model
        t0 = time.time()
        
        if args.k_per_image > 0:
            results = process_random_crops(img_rgb, seg_model, args, device, args.k_per_image)
        else:
            results = process_image_segmentation(img_rgb, seg_model, args, device)
            
        dt = time.time() - t0
        print(f"      Processed {len(results)} tiles in {dt:.2f}s")
        
        for res in results:
            if cap is not None and saved_total >= cap:
                break
                
            pred_rt = res['pred_rt']
            # pred_streak = res['pred_streak']
            
            # Criterion: Empty mask means no detection (background)
            # You can adjust this threshold (e.g. sum < 10 pixels) if needed.
            # Here assuming strict 0 for background.
            is_background = True
            
            if pred_rt is not None and pred_rt.sum() > 0:
                is_background = False

                
            if is_background:
                # Save the original tile
                tile_info = res['tile_info']
                # scaled_img is what we usually want to save (visuals) 
                # or ori_img for raw data? 
                # The Yolo script saved 'crop' which was raw crop. 
                # tile_info['ori_img'] is the raw crop from original image.
                tile_img = tile_info['ori_img']
                
                # We need x, y for filename
                x, y = tile_info['x'], tile_info['y']
                
                out_name = f"{img_stem}_x{x}_y{y}.npy"
                out_path = os.path.join(args.out_dir, out_name)
                save_npy(out_path, tile_img)
                saved_total += 1

                # save png for debugging purposes
                # out_png_name = f"{img_stem}_x{x}_y{y}.png"
                # out_png_path = os.path.join(args.out_dir, out_png_name)
                # cv2.imwrite(out_png_path, np.stack((tile_info['scaled_img'])))   

    print(f"[INFO] Total tiles saved: {saved_total}")

if __name__ == "__main__":
    main()
