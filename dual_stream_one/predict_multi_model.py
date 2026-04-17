#!/usr/bin/env python
"""
Multi-model prediction script.

Loads multiple pretrained segmentation models (one per supplied directory) and
runs inference on every tiled image. Results for each model are saved to
separate sub-directories under a shared output root.
"""
import argparse
import os
from time import time
from typing import List, Sequence, Tuple

import albumentations as AT
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from models import get_model
from utils.bs_detector_sep import detect_roundish, remove_masked_with_zero

from utils import transforms
from utils.utils import get_colormap
import cv2
import json
from numba import prange, njit
from matplotlib import pyplot as plt
from tools.eval.full_frame_line_eval import line_intersection_check, line_angle_degrees
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'data_generation'))
from ht_utils import _make_params, compute_rt_map

# -----------------------------------------------------------------------------
# CONFIG: set USE_ARGPARSE = True to use the hardcoded parameters below.
# Toggle USE_ARGPARSE = True to read from CLI flags again.
# -----------------------------------------------------------------------------
USE_ARGPARSE = True
HARD_CONFIG = dict(
    model="bisenetv2dualmaskguidedv2",
    encoder=None,
    decoder=None,
    num_classes=2,
    use_aux=True,
    # List of checkpoint directories; each must contain best.pth (or set ckpt_filename below)
    model_dirs=[
        "/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/both_new_mean",
    ],
    ckpt_filename="best.pth",
    # Shared output root; per-model results go into <output_root>/<model_dir_name>/
    output_root="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/multi_model_eval/",
    img_dir= "/media/ckchng/internal2TB/FILTERED_IMAGES/",
    anno_json= "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_and_rt_two_stage_labeled_merged_labels.json",
    num_angles=192,
    num_rhos=288,
    sep=True,
    device="auto",
    batch_size=128,
    scale=1.0,
    mean = (0.39509313, 0.39509313, 0.395093130),
    std = (0.17064099, 0.17064099, 0.17064099),
    mean2=(0.34827731, 0.34827731, 0.34827731),
    std2=(0.16927711, 0.16927711, 0.16927711),
    palette="cityscapes",
    blend=True,
    blend_alpha=0.3,
    tile_size=288,
    tile_stride=144,
    sep_params=[3.0, 6, 5.5, 0.6, 6.0, 0.1],
    rho_min_cap=-144,
    rho_max_cap=143
)


# -----------------------------------------------------------------------------
# Config holder to satisfy get_model() and get_colormap() expectations
# -----------------------------------------------------------------------------
class PredictConfig:
    def __init__(
        self,
        model: str,
        num_class: int,
        encoder: str = None,
        decoder: str = None,
        # encoder_weights: str = "imagenet",
        use_aux: bool = False,
        use_detail_head: bool = False,
        colormap: str = "cityscapes",
    ):
        self.model = model
        self.num_class = num_class
        self.encoder = encoder
        self.decoder = decoder
        # self.encoder_weights = encoder_weights
        self.use_aux = use_aux
        self.use_detail_head = use_detail_head
        self.colormap = colormap


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


# ---------- core (numba) ----------



def build_transform(scale: float, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    return AT.Compose(
        [
            transforms.Scale(scale=scale, is_testing=True),
            transforms.PadPairToMax(padding_value=0, mask_value=0),
            AT.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)


def prepare_palette(args, config: PredictConfig) -> torch.Tensor:
    if args.palette == "binary":
        colors = torch.tensor([[0, 0, 0], [255, 255, 255]], dtype=torch.uint8)
    elif args.palette == "cityscapes":
        colors = torch.tensor(get_colormap(config), dtype=torch.uint8)
    else:
        # Fallback grayscale (no coloring, just class index replicated on 3 channels)
        colors = None
    return colors


def tile_image(
    image: np.ndarray, tile_size: int, stride: int, max_rho: float, num_rhos: int, num_angles: int, itheta: float, irho: float, rho_min_cap: int, rho_max_cap: int, params: List[float], sep: bool = False, pad_value: Tuple[int, int, int] = (0, 0, 0)
) -> List[dict]:
    """Slice an image into overlapping tiles. Pads on the bottom/right if needed."""
    h, w, _ = image.shape
    tiles = []
    
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # if tile goes beyond image boundary, adjust
            if y + tile_size > h:
                y = h - tile_size
            if x + tile_size > w:
                x = w - tile_size
            if y < 0:
                y = 0
            if x < 0:
                x = 0
            tile = image[y : y + tile_size, x : x + tile_size]
            

            # perform RT here
            # sep here
            if sep:
                objs_full, round_mask_full, keep_full, rms_full, removed_obj_bboxes_full = detect_roundish(tile[:, :, 0],
                                                                                                           params[0], params[1], 
                                                                                                           params[2], params[3], 
                                                                                                           params[4], deblend_cont=params[5])
                processed_tile = remove_masked_with_zero(tile[:, :, 0], round_mask_full)    
            else:
                processed_tile = tile[:, :, 0]

            processed_tile = np.stack((processed_tile,)*3, axis=-1)

            # RT map uses the unmasked tile so corner pixels contribute fully
            rt_map, theta_deg, rhos = compute_rt_map(processed_tile[:, :, 0], max_rho/2, itheta, irho,
                                     rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap)


            rt_map = rt_map - rt_map.min()
            rt_map_max = rt_map.max()
            if rt_map_max > 0:
                rt_map = rt_map / rt_map_max * 255

            rt_map = rt_map.astype(np.uint8)
            # pad the rt_map to be divisible by 32
            pad_h = (32 - rt_map.shape[0] % 32) % 32
            pad_w = (32 - rt_map.shape[1] % 32) % 32
            rt_map = cv2.copyMakeBorder(rt_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            # pad rt_map to 3 channels

            rt_map = rt_map[:, :, np.newaxis]
            rt_map = np.repeat(rt_map, 3, axis=2)
            th, tw, _ = rt_map.shape

            scaled_tile = processed_tile.copy()

            zero_mask = scaled_tile == 0
            scaled_tile = scaled_tile - scaled_tile.min()
            scaled_tile_max = scaled_tile.max()
            if scaled_tile_max > 0:
                scaled_tile = scaled_tile / scaled_tile_max * 255.0
            scaled_tile[zero_mask] = 0
            scaled_tile = scaled_tile.astype(np.uint8)


            # if th != num_angles or tw != num_rhos:
            #     print('something is wrong')

            tiles.append(
                {
                    'ori_img': processed_tile,
                    "scaled_img": scaled_tile,
                    "rt_map": rt_map,  # save the tile instead
                    "y": y,
                    "x": x,
                    "h": th,
                    "w": tw,
                }
            )
    return tiles


def line_endpoints_center_rho_theta(rho, theta, H, W, eps=1e-10):
    """
    Given a line in Hough form with origin at the image center:
        (x - cx)*cos(theta) + (y - cy)*sin(theta) = rho

    where:
        cx = (W-1)/2, cy = (H-1)/2
    return the two endpoints (x0, y0), (x1, y1) where the line segment
    lies inside an HxW image (0 <= x < W, 0 <= y < H).

    Returns:
        (x0, y0), (x1, y1) as integer tuples.
        If no valid segment exists, returns (None, None).
    """
    cx = (W) / 2.0
    cy = (H) / 2.0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    points = []

    # Intersections with vertical borders x = 0 and x = W-1
    if abs(sin_t) > eps:
        # x = 0
        y = cy + (rho + cx * cos_t) / sin_t
        if 0.0 <= y <= H + 0.5 - 1:
            points.append((0.0, y))

        # x = W-1  (note W-1 - cx = cx)
        y = cy + (rho - cx * cos_t) / sin_t
        if 0.0 <= y <= H + 0.5 - 1:
            points.append((W - 1.0, y))

    # Intersections with horizontal borders y = 0 and y = H-1
    if abs(cos_t) > eps:
        # y = 0
        x = cx + (rho + cy * sin_t) / cos_t
        if 0.0 <= x <= W + 0.5 - 1:
            points.append((x, 0.0))

        # y = H-1 (note H-1 - cy = cy)
        x = cx + (rho - cy * sin_t) / cos_t
        if 0.0 <= x <= W + 0.5 - 1:
            points.append((x, H - 1.0))

    # Deduplicate near-identical points
    uniq = []
    for p in points:
        if not any(np.allclose(p, q) for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return None, None

    # If more than 2 (corner case: line through a corner), take the pair with max distance
    if len(uniq) > 2:
        max_d = -1.0
        best_pair = (uniq[0], uniq[1])
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                dx = uniq[i][0] - uniq[j][0]
                dy = uniq[i][1] - uniq[j][1]
                d2 = dx * dx + dy * dy
                if d2 > max_d:
                    max_d = d2
                    best_pair = (uniq[i], uniq[j])
        p0, p1 = best_pair
    else:
        p0, p1 = uniq[0], uniq[1]

    # Round to integer pixel coords
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))

    # Clamp to image in case rounding nudged them outside
    x0 = min(max(x0, 0), W - 1)
    y0 = min(max(y0, 0), H - 1)
    x1 = min(max(x1, 0), W - 1)
    y1 = min(max(y1, 0), H - 1)

    return (x0, y0), (x1, y1)

from typing import List, Tuple
import math

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def merge_connected_segments_2d(segments: List[Segment], eps: float = 1e-6) -> List[Segment]:
    """
    Merge 2D line segments that lie on the same line and either overlap or touch.
    Each segment is ((x1, y1), (x2, y2)).
    """

    def dist(p: Point, q: Point) -> float:
        return math.hypot(p[0] - q[0], p[1] - q[1])

    def points_equal(p: Point, q: Point) -> bool:
        return dist(p, q) <= eps

    def are_colinear(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
        """
        Check if segments a1-a2 and b1-b2 lie on the same infinite line.
        """
        v = (a2[0] - a1[0], a2[1] - a1[1])
        w1 = (b1[0] - a1[0], b1[1] - a1[1])
        w2 = (b2[0] - a1[0], b2[1] - a1[1])

        cross1 = v[0] * w1[1] - v[1] * w1[0]
        cross2 = v[0] * w2[1] - v[1] * w2[0]
        return abs(cross1) <= eps and abs(cross2) <= eps

    def project_scalar(a1: Point, a2: Point, p: Point) -> float:
        """
        Scalar coordinate of p when projected onto the line through a1->a2.
        Only relative ordering matters, not absolute value.
        """
        v = (a2[0] - a1[0], a2[1] - a1[1])
        denom = v[0] * v[0] + v[1] * v[1]
        if denom == 0:
            return 0.0
        # Normalize direction so scalar is in units of length
        return ((p[0] - a1[0]) * v[0] + (p[1] - a1[1]) * v[1]) / math.sqrt(denom)

    def intervals_overlap(i1, i2) -> bool:
        a1, a2 = i1
        b1, b2 = i2
        if a1 > a2:
            a1, a2 = a2, a1
        if b1 > b2:
            b1, b2 = b2, b1
        # Overlap or just touch
        return not (a2 < b1 - eps or b2 < a1 - eps)

    def try_merge(s1: Segment, s2: Segment):
        a1, a2 = s1
        b1, b2 = s2

        # 1) Must be colinear
        if not are_colinear(a1, a2, b1, b2):
            return None

        # 2) Check if they overlap or share an endpoint along that line
        shares_endpoint = any(points_equal(p, q) for p in s1 for q in s2)

        t_a1 = project_scalar(a1, a2, a1)
        t_a2 = project_scalar(a1, a2, a2)
        t_b1 = project_scalar(a1, a2, b1)
        t_b2 = project_scalar(a1, a2, b2)

        if not intervals_overlap((t_a1, t_a2), (t_b1, t_b2)) and not shares_endpoint:
            return None

        # 3) Build the merged segment: endpoints are the extreme projected points
        ts = [t_a1, t_a2, t_b1, t_b2]
        pts = [a1, a2, b1, b2]

        t_min = min(ts)
        t_max = max(ts)

        # choose points whose projection is closest to extremes
        def proj(p: Point) -> float:
            return project_scalar(a1, a2, p)

        p_min = min(pts, key=lambda p: abs(proj(p) - t_min))
        p_max = min(pts, key=lambda p: abs(proj(p) - t_max))

        return (p_min, p_max)

    segs = segments[:]
    changed = True

    while changed:
        changed = False
        n = len(segs)
        for i in range(n):
            if changed:
                break
            for j in range(i + 1, n):
                merged = try_merge(segs[i], segs[j])
                if merged is not None:
                    # remove i, j and insert merged
                    new_segs = []
                    for k, s in enumerate(segs):
                        if k not in (i, j):
                            new_segs.append(s)
                    new_segs.append(merged)
                    segs = new_segs
                    changed = True
                    break

    return segs

def pad_to_size(img, target_h, target_w):
        """Pad to (target_h, target_w) with zeros, centering the original."""
        h, w = img.shape[:2]
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if pad_h == 0 and pad_w == 0:
            return img
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def crop_line_region_with_min_size(img, line_xyxy, min_size=75, pad=10):
    """Crop around a GT line, ensuring each crop dimension is at least min_size."""
    img_h, img_w = img.shape[:2]
    lx1, ly1, lx2, ly2 = line_xyxy

    x1 = int(round(lx1))
    y1 = int(round(ly1))
    x2 = int(round(lx2))
    y2 = int(round(ly2))

    x_min = max(min(x1, x2) - pad, 0)
    y_min = max(min(y1, y2) - pad, 0)
    x_max = min(max(x1, x2) + pad, img_w - 1)
    y_max = min(max(y1, y2) + pad, img_h - 1)

    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1

    target_w = min(max(box_w, int(min_size)), img_w)
    target_h = min(max(box_h, int(min_size)), img_h)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    crop_x0 = int(round(center_x - target_w / 2.0))
    crop_y0 = int(round(center_y - target_h / 2.0))

    crop_x0 = max(0, min(crop_x0, img_w - target_w))
    crop_y0 = max(0, min(crop_y0, img_h - target_h))
    crop_x1 = crop_x0 + target_w
    crop_y1 = crop_y0 + target_h

    crop = img[crop_y0:crop_y1, crop_x0:crop_x1].copy()

    zero_mask = crop == 0
    crop = crop - crop.min()
    if crop.max() > 0:
        crop = crop / crop.max() * 255.0
    crop[zero_mask] = 0
    crop = crop.astype(np.uint8)
    return crop, crop_x0, crop_y0


# -----------------------------------------------------------------------------
# Main prediction flow
# -----------------------------------------------------------------------------
def run_multi_model(args):
    """Run inference for all models in args.model_dirs over all images."""

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    is_dual = 'dual' in args.model.lower()
    is_maskguided = 'maskguided' in args.model.lower()

    # ── Build shared transforms ───────────────────────────────────────────────
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
        transform = build_transform(scale=args.scale, mean=args.mean, std=args.std)

    # ── Load all models once ──────────────────────────────────────────────────
    config = PredictConfig(
        model=args.model,
        num_class=args.num_classes,
        encoder=args.encoder,
        decoder=args.decoder,
        use_aux=args.use_aux,
        colormap=args.palette if args.palette == "cityscapes" else "cityscapes",
    )
    colors = prepare_palette(args, config)

    ckpt_filename = getattr(args, 'ckpt_filename', 'best.pth')
    loaded_models = []  # list of (model_dir_name, model, output_dir)
    for model_dir in args.model_dirs:
        model_dir_name = os.path.basename(os.path.normpath(model_dir))
        ckpt_path = os.path.join(model_dir, ckpt_filename)
        if not os.path.isfile(ckpt_path):
            print(f"[WARNING] Checkpoint not found, skipping: {ckpt_path}")
            continue
        m = get_model(config).to(device)
        m.eval()
        load_checkpoint(m, ckpt_path, device)
        if getattr(args, 'sep', False):
            param_str = "_".join(str(p).rstrip('0').rstrip('.') if '.' in str(p) else str(p)
                                 for p in args.sep_params)
            sep_suffix = f"_sep_{param_str}"
        else:
            sep_suffix = "_nosep"
        output_dir = os.path.join(args.output_root, model_dir_name + sep_suffix)
        # Create per-model output directories
        os.makedirs(os.path.join(output_dir, 'full_frame_result', 'txt'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'full_frame_result', 'vis'), exist_ok=True)
        loaded_models.append((model_dir_name, m, output_dir))
        print(f"Loaded model: {model_dir_name}  ckpt: {ckpt_path}")

    if not loaded_models:
        raise RuntimeError("No valid checkpoints found in model_dirs.")

    
    #### load image
    img_paths = {}
    if args.img_dir:
        print("Building NEF file mapping...")
        for root, dirs, files in os.walk(args.img_dir):
            for filename in files:
                if filename.endswith(".npy"):
                    img_name = os.path.splitext(filename)[0]
                    img_path = os.path.join(root, filename)
                    img_paths[img_name] = img_path

    # data_dict = np.load(args.anno_npz, allow_pickle=True)
    # image_dirs = [str(p) for p in data_dict['imgPath']]
    # annotations = data_dict['XY']

    # load json file for annotations
    with open(args.anno_json, 'r') as f:
        data_json = json.load(f)
    
    image_dirs = data_json['images']
    annotations = data_json['annotations']

    itheta = 180 / args.num_angles
    max_rho = np.hypot(args.tile_size, args.tile_size) + 1
    
    rho_min_cap = args.rho_min_cap
    rho_max_cap = args.rho_max_cap
    if rho_min_cap is not None and rho_max_cap is not None:
        _effective_half_rho = (rho_max_cap - rho_min_cap) / 2.0
    elif rho_max_cap is not None:
        _effective_half_rho = rho_max_cap
    elif rho_min_cap is not None:
        _effective_half_rho = abs(rho_min_cap)
    else:
        _effective_half_rho = max_rho / 2.0
    irho = (2.0 * _effective_half_rho) / (args.num_rhos - 1)
    
    

    category_ids_for_split = [1, 9, 10]
    category_for_not_relevant = 3
    category_for_hard_to_cat = 5

    # Create per-model tile sub-directories
    model_dirs_info = []  # (model_dir_name, model, output_dir, tp_dir, fp_dir, fn_dir)
    for model_dir_name, m, output_dir in loaded_models:
        tp_dir = os.path.join(output_dir, 'tile_tp')
        fp_dir = os.path.join(output_dir, 'tile_fp')
        fn_dir = os.path.join(output_dir, 'tile_fn')
        os.makedirs(tp_dir, exist_ok=True)
        os.makedirs(fp_dir, exist_ok=True)
        os.makedirs(fn_dir, exist_ok=True)
        for category_id in category_ids_for_split:
            os.makedirs(os.path.join(tp_dir, str(category_id)), exist_ok=True)
            os.makedirs(os.path.join(fn_dir, str(category_id)), exist_ok=True)
        os.makedirs(os.path.join(tp_dir, str(category_for_hard_to_cat)), exist_ok=True)
        os.makedirs(os.path.join(fn_dir, str(category_for_hard_to_cat)), exist_ok=True)
        model_dirs_info.append((model_dir_name, m, output_dir, tp_dir, fp_dir, fn_dir))

    #### prepare RT params
    thetas_deg, thetas, cos_t, sin_t, rhos = _make_params(max_rho/2, theta_res_deg=itheta, rho_res=irho, rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap)

    for int_id in range(len(image_dirs)):

        # # skip 'not relevant' if labeled
        # label = "unknown"
        # img_basename = os.path.basename(image_dirs[int_id]['file_name'])
        # # Try to find label in annotations if available
        # if anno:
        #     cat_id = anno[0].get('category_id', -1)
        #     if cat_id == 3: # Assuming 3 is "not relevant" based on category_for_not_relevant
        #         continue

        # if os.path.exists(out_path):
        img_path = str(image_dirs[int_id]['file_name'])
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = np.load(img_path)
        anno = [anno for anno in annotations if anno['image_id'] == image_dirs[int_id]['id']]
        gt_lines = [a['xyxy'] for a in anno]
        gt_category_ids = [int(a.get('category_id', -1)) for a in anno]
        
        # stack it to create 3 channels
        image_rgb = np.stack((img,)*3, axis=-1)
        img_scaled = img.copy()
        zero_mask = img_scaled == 0
        img_scaled = img_scaled - img_scaled.min()
        img_scaled = img_scaled / img_scaled.max() * 255.0
        img_scaled[zero_mask] = 0
        img_scaled = img_scaled.astype(np.uint8)
        img_scaled = np.stack((img_scaled,)*3, axis=-1)
        
        tiles = tile_image(image_rgb, tile_size=args.tile_size, stride=args.tile_stride, max_rho=max_rho,
                           num_rhos=args.num_rhos, num_angles=args.num_angles, itheta=itheta, irho=irho,
                           rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap, params=args.sep_params, sep=args.sep)
        print(f"{img_name}: {len(tiles)} tiles")

        # ── Pre-compute all batch tensors once (shared across models) ─────────
        all_batch_tensors  = []  # list of batch_tensor (one per batch)
        all_batch_tensors2 = []  # list of batch_tensor2 (dual stream only)
        for idx in range(0, len(tiles), args.batch_size):
            batch_tiles = tiles[idx : idx + args.batch_size]
            images_aug = []
            images2_aug = []
            for t in batch_tiles:
                if is_dual:
                    res1 = scale_transform(image=t["rt_map"])
                    res2 = scale_transform(image=t["scaled_img"])
                    final1 = norm_transform1(image=res1["image"])["image"]
                    final2 = norm_transform2(image=res2["image"])["image"]
                    images_aug.append(final1)
                    images2_aug.append(final2)
                else:
                    images_aug.append(transform(image=t["rt_map"])["image"])
            all_batch_tensors.append(torch.stack(images_aug).to(device=device, dtype=torch.float32))
            if is_dual:
                all_batch_tensors2.append(torch.stack(images2_aug).to(device=device, dtype=torch.float32))
            else:
                all_batch_tensors2.append(None)

        # ── Run each model independently ──────────────────────────────────────
        for model_dir_name, model, output_dir, tp_dir, fp_dir, fn_dir in model_dirs_info:
            print(f"  [{model_dir_name}] running inference...")
            all_pred_lines = []

            for batch_idx, (idx, batch_tensor, batch_tensor2) in enumerate(
                zip(range(0, len(tiles), args.batch_size), all_batch_tensors, all_batch_tensors2)
            ):
                batch_tiles = tiles[idx : idx + args.batch_size]

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
                        preds_rt_tensor = logits_main.argmax(dim=1)
                    elif args.num_classes == 1:
                        preds_rt_tensor = (logits_main.sigmoid() > 0.5).squeeze(1).long()
                    else:
                        preds_rt_tensor = logits_main.argmax(dim=1)

                    if logits_s2 is None:
                        preds_streak_tensor = torch.zeros_like(preds_rt_tensor)
                    else:
                        try:
                            if args.num_classes == 2:
                                preds_streak_tensor = logits_s2.argmax(dim=1)
                            elif args.num_classes == 1:
                                preds_streak_tensor = (logits_s2.sigmoid() > 0.5).squeeze(1).long()
                            else:
                                preds_streak_tensor = logits_s2.argmax(dim=1)
                        except Exception:
                            preds_streak_tensor = torch.zeros_like(preds_rt_tensor)

                preds_rt = preds_rt_tensor.cpu().numpy()
                preds_streak = preds_streak_tensor.cpu().numpy()

                peak = []
                bboxes = []

                processed_batch_tiles = []
                for pred, pstreak, tile in zip(preds_rt, preds_streak, batch_tiles):
                    pred_lines = []
                    pred_streak_mask = []
                    if pred.max() == 0:
                        continue

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        pred.astype(np.uint8), connectivity=8
                    )

                    legit_pred_flag = False
                    for label_id in range(1, num_labels):
                        x, y, w, h, area = stats[label_id]
                        if area == 0:
                            continue
                        x1, y1 = x, y
                        x2, y2 = x + w - 1, y + h - 1
                        if w < 10 or h < 10:
                            continue
                        bboxes.append((x1, y1, x2, y2))

                        crop_rt_map = tile['rt_map'][y1:y2+1, x1:x2+1, 0]
                        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(crop_rt_map)
                        peak_x = np.clip(maxLoc[0] + x1, 0, len(rhos) - 1)
                        peak_y = np.clip(maxLoc[1] + y1, 0, len(thetas) - 1)

                        pred_rho = rhos[int(round(peak_x))]
                        pred_theta = thetas[int(round(peak_y))]

                        if pred_rho < -120 or pred_rho > 120:
                            continue

                        pred_p0, pred_p1 = line_endpoints_center_rho_theta(pred_rho, pred_theta, args.tile_size, args.tile_size)
                        if pred_p0 is None or pred_p1 is None:
                            continue

                        global_pred_p0 = (pred_p0[0] + tile['x'], pred_p0[1] + tile['y'])
                        global_pred_p1 = (pred_p1[0] + tile['x'], pred_p1[1] + tile['y'])

                        pred_lines.append([global_pred_p0[0], global_pred_p0[1], global_pred_p1[0], global_pred_p1[1]])
                        all_pred_lines.append([global_pred_p0[0], global_pred_p0[1], global_pred_p1[0], global_pred_p1[1]])

                    gt_box_matched_id = line_intersection_check(pred_lines, gt_lines)
                    gt_line_matched_id = np.ones(len(gt_lines)) * -1
                    matched_pred_idx = set()
                    pred_idx_to_category = {}
                    for gt_idx, matched_pred_indices in enumerate(gt_box_matched_id):
                        if not matched_pred_indices:
                            continue
                        gt_line = gt_lines[gt_idx]
                        for pred_idx in matched_pred_indices:
                            pred_line = pred_lines[pred_idx]
                            try:
                                ang_diff = line_angle_degrees(gt_line[0:2], gt_line[2:4], pred_line[0:2], pred_line[2:4])
                            except:
                                continue
                            if ang_diff < 10.0:
                                gt_line_matched_id[gt_idx] = pred_idx
                                matched_pred_idx.add(pred_idx)
                                if pred_idx not in pred_idx_to_category:
                                    pred_idx_to_category[pred_idx] = gt_category_ids[gt_idx]

                    for pred_line_id, pred_line in enumerate(pred_lines):
                        mask = np.zeros_like(tile['ori_img'][:, :, 0], dtype=np.uint8)
                        cv2.line(mask, (int(pred_line[0] - tile['x']), int(pred_line[1] - tile['y'])),
                                       (int(pred_line[2] - tile['x']), int(pred_line[3] - tile['y'])), 1, 30)

                        masked_tile = tile['ori_img'] * mask[:, :, np.newaxis]

                        sub_rt_map, _, _ = compute_rt_map(masked_tile[:, :, 0], max_rho/2, theta_res_deg=itheta, rho_res=irho,
                                                           rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap)

                        zero_mask = masked_tile == 0
                        masked_tile = masked_tile - masked_tile.min()
                        masked_tile = masked_tile / masked_tile.max() * 255
                        masked_tile[zero_mask] = 0

                        sub_rt_map = sub_rt_map - sub_rt_map.min()
                        sub_rt_map = sub_rt_map / sub_rt_map.max() * 255
                        sub_rt_map = sub_rt_map.astype(np.uint8)
                        pad_h = (32 - sub_rt_map.shape[0] % 32) % 32
                        pad_w = (32 - sub_rt_map.shape[1] % 32) % 32
                        sub_rt_map = cv2.copyMakeBorder(sub_rt_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                        sub_rt_map = sub_rt_map[:, :, np.newaxis]
                        sub_rt_map = np.repeat(sub_rt_map, 3, axis=2)
                        th, tw, _ = sub_rt_map.shape

                        tile_disp   = np.clip(tile['scaled_img'], 0, 255).astype(np.uint8)
                        masked_disp = np.clip(masked_tile, 0, 255).astype(np.uint8)
                        rt_disp     = np.clip(tile['rt_map'], 0, 255).astype(np.uint8)
                        sub_rt_disp = np.clip(sub_rt_map, 0, 255).astype(np.uint8)
                        streak_disp = cv2.cvtColor(pstreak.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

                        panel_w  = max(tile_disp.shape[1], masked_disp.shape[1], streak_disp.shape[1], rt_disp.shape[1], sub_rt_disp.shape[1])
                        top_h    = max(tile_disp.shape[0], masked_disp.shape[0], streak_disp.shape[0])
                        bottom_h = max(rt_disp.shape[0], sub_rt_disp.shape[0])

                        tile_pad    = pad_to_size(tile_disp,    top_h,    panel_w)
                        masked_pad  = pad_to_size(masked_disp,  top_h,    panel_w)
                        streak_pad  = pad_to_size(streak_disp,  top_h,    panel_w)
                        rt_pad      = pad_to_size(rt_disp,      bottom_h, panel_w)
                        sub_rt_pad  = pad_to_size(sub_rt_disp,  bottom_h, panel_w)
                        blank_pad   = np.zeros_like(rt_pad)

                        top_row    = cv2.hconcat([tile_pad, masked_pad, streak_pad])
                        bottom_row = cv2.hconcat([rt_pad, sub_rt_pad, blank_pad])
                        final_w    = max(top_row.shape[1], bottom_row.shape[1])
                        top_row    = pad_to_size(top_row,    top_row.shape[0],    final_w)
                        bottom_row = pad_to_size(bottom_row, bottom_row.shape[0], final_w)
                        composite  = cv2.vconcat([top_row, bottom_row])

                        save_name     = f"{img_name}_y{tile['y']}_x{tile['x']}_pred{pred_line_id}.png"
                        save_txt_path = save_name.replace('.png', '.txt')

                        if pred_line_id in matched_pred_idx:
                            tp_category_id = pred_idx_to_category.get(pred_line_id, -1)
                            if tp_category_id in category_ids_for_split or tp_category_id == category_for_hard_to_cat:
                                tp_subdir = str(tp_category_id)
                                save_path     = os.path.join(tp_dir, tp_subdir, save_name)
                                save_txt_path = os.path.join(tp_dir, tp_subdir, save_txt_path)
                            elif tp_category_id == category_for_not_relevant:
                                continue
                            elif tp_category_id in [6, 11]:
                                save_path     = os.path.join(fp_dir, save_name)
                                save_txt_path = os.path.join(fp_dir, save_txt_path)
                            else:
                                save_path     = os.path.join(fp_dir, save_name)
                                save_txt_path = os.path.join(fp_dir, save_txt_path)
                        else:
                            save_path     = os.path.join(fp_dir, save_name)
                            save_txt_path = os.path.join(fp_dir, save_txt_path)

                        cv2.imwrite(save_path, composite)
                        with open(save_txt_path, 'a') as f:
                            f.write(f"{pred_line[0]},{pred_line[1]},{pred_line[2]},{pred_line[3]}\n")

            # ── Full-frame FN detection and save ─────────────────────────
            if len(gt_lines) > 0:
                if len(all_pred_lines) > 0:
                    gt_box_matched_id_full = line_intersection_check(all_pred_lines, gt_lines)
                else:
                    gt_box_matched_id_full = [[] for _ in range(len(gt_lines))]

                gt_line_matched_full = np.zeros(len(gt_lines), dtype=bool)
                for gt_idx, matched_pred_indices in enumerate(gt_box_matched_id_full):
                    if not matched_pred_indices:
                        continue
                    gt_line = gt_lines[gt_idx]
                    for pred_idx in matched_pred_indices:
                        pred_line = all_pred_lines[pred_idx]
                        try:
                            ang_diff = line_angle_degrees(gt_line[0:2], gt_line[2:4], pred_line[0:2], pred_line[2:4])
                        except Exception:
                            continue
                        if ang_diff < 10.0:
                            gt_line_matched_full[gt_idx] = True
                            break

                fn_gt_indices = np.where(~gt_line_matched_full)[0]
                for gt_idx in fn_gt_indices:
                    gt_line = gt_lines[int(gt_idx)]
                    fn_category_id = gt_category_ids[int(gt_idx)]
                    if fn_category_id not in category_ids_for_split and fn_category_id != category_for_hard_to_cat:
                        continue
                    fn_crop, crop_x0, crop_y0 = crop_line_region_with_min_size(
                        image_rgb, gt_line, min_size=75, pad=10,
                    )
                    fn_subdir    = str(fn_category_id)
                    fn_save_name = f"{img_name}_gt{int(gt_idx)}_cat{fn_category_id}_fn.png"
                    fn_save_path = os.path.join(fn_dir, fn_subdir, fn_save_name)
                    cv2.imwrite(fn_save_path, fn_crop)

            # ── Full-frame visualisation & line save ──────────────────────
            all_pred_lines_tuples = [((line[0], line[1]), (line[2], line[3])) for line in all_pred_lines]
            merged_lines = merge_connected_segments_2d(all_pred_lines_tuples, 1e-6)

            img_scaled_vis = img_scaled.copy()
            for curr_line, curr_category_id in zip(gt_lines, gt_category_ids):
                if curr_category_id not in category_ids_for_split:
                    continue
                cv2.line(img_scaled_vis,
                         (int(round(curr_line[0])), int(round(curr_line[1]))),
                         (int(round(curr_line[2])), int(round(curr_line[3]))),
                         (0, 0, 255), 5)

            for line in merged_lines:
                p0, p1 = line
                cv2.line(img_scaled_vis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 0, 0), 2)

            output_image_path = os.path.join(output_dir, 'full_frame_result', 'vis',
                                             f"{os.path.splitext(os.path.basename(img_path))[0]}_lines.png")
            # cv2.imwrite(output_image_path, img_scaled_vis)

            output_lines_path = os.path.join(output_dir, 'full_frame_result', 'txt',
                                             f"{os.path.splitext(os.path.basename(img_path))[0]}_lines.txt")
            with open(output_lines_path, "w") as f:
                for line in merged_lines:
                    f.write(f"{line[0][0]},{line[0][1]},{line[1][0]},{line[1][1]}\n")

            print(f"  [{model_dir_name}] done: {img_name}")




def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model prediction script for realtime semantic segmentation.")
    parser.add_argument("--model", type=str, default="bisenetv2dualmaskguidedv2", help="Model name registered in models/model_registry.")
    parser.add_argument("--encoder", type=str, default=None, help="Encoder (only needed if model == smp).")
    parser.add_argument("--decoder", type=str, default=None, help="Decoder (only needed if model == smp).")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes.")
    parser.add_argument("--use-aux", action="store_true", help="Set if the checkpoint was trained with auxiliary heads.")
    parser.add_argument("--model-dirs", type=str, nargs='+', required=True,
                        help="One or more directories, each containing a checkpoint file.")
    parser.add_argument("--ckpt-filename", type=str, default="best.pth",
                        help="Checkpoint filename to look for inside each model-dir (default: best.pth).")
    parser.add_argument("--img-dir", type=str, default=None, help="Directory containing .npy images (for name remap).")
    parser.add_argument("--anno-json", type=str, required=True, help="Path to JSON annotations file.")
    parser.add_argument("--output-root", type=str, default="predictions",
                        help="Shared output root; per-model results are saved under <output-root>/<model-dir-name>/.")
    parser.add_argument("--device", type=str, default="auto", help="Device string (e.g., cuda, cuda:0, cpu, or auto).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor applied before normalization.")
    parser.add_argument("--sep", type=lambda x: x.lower() != 'false', default=True, help="Whether to use separate processing.")
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=(0.30566086, 0.30566086, 0.30566086),
        help="Normalization mean (R G B). Default matches custom dataset in this repo.",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=(0.21072077, 0.21072077, 0.21072077),
        help="Normalization std (R G B). Default matches custom dataset in this repo.",
    )
    parser.add_argument(
        "--mean2",
        type=float,
        nargs=3,
        default=(0.34827731, 0.34827731, 0.34827731),
        help="Normalization mean for second stream (R G B).",
    )
    parser.add_argument(
        "--std2",
        type=float,
        nargs=3,
        default=(0.16927711, 0.16927711, 0.16927711),
        help="Normalization std for second stream (R G B).",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="binary",
        choices=["binary", "cityscapes", "none"],
        help="Color palette for saving masks (binary, cityscapes, or none for grayscale index).",
    )
    parser.add_argument("--blend", action="store_true", help="Also save image/mask blend overlays.")
    parser.add_argument("--blend-alpha", type=float, default=0.3, help="Alpha for blending.")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile height/width for slicing the input image.")
    parser.add_argument("--tile-stride", type=int, default=256, help="Stride between tiles (overlap = tile_size - stride).")
    parser.add_argument("--num_angles", type=int, default=180, help="Number of angle bins for RT transform.")
    parser.add_argument("--num_rhos", type=int, default=720, help="Number of rho bins for RT transform.")
    parser.add_argument("--sep-params", type=float, nargs=6, default=[3.0, 6, 5.5, 0.6, 6.0, 0.1], help="Sep params: thresh_sigma minarea elong_max r_eff_min r_eff_max hough_thresh.")
    parser.add_argument("--rho-min-cap", type=float, default=None, help="Minimum rho cap for RT transform.")
    parser.add_argument("--rho-max-cap", type=float, default=None, help="Maximum rho cap for RT transform.")
    return parser.parse_args()


if __name__ == "__main__":
    if USE_ARGPARSE:
        args = parse_args()
    else:
        args = argparse.Namespace(**HARD_CONFIG)

    run_multi_model(args)
