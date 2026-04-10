#!/usr/bin/env python
"""
Standalone prediction script.

Loads a pretrained segmentation model and runs a forward pass on images you
provide (file or directory). This mirrors the repo's test-time preprocessing:
optional scaling, normalization, and ToTensorV2 -> model -> argmax. Outputs
class-index masks and optional color-blended overlays.
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
from sub_ht_network.bs_detector_sep import detect_roundish, compute_rt_map, scale_to_255, remove_masked_with_zero

from utils import transforms
from utils.utils import get_colormap
import cv2
import json
from numba import prange, njit
from matplotlib import pyplot as plt
from tools.eval.full_frame_line_eval import line_intersection_check, line_angle_degrees
from utils.HT_utils import _hough_accumulate_intensity_dh, endpoints_to_rho_theta_dh, rho_theta_to_indices_dh, _make_params_dh, hough_bruteforce_intensity_numba_dh
from utils.HT_utils import _hough_accumulate_intensity, endpoints_to_rho_theta, rho_theta_to_indices, _make_params, hough_bruteforce_intensity_numba

# -----------------------------------------------------------------------------
# CONFIG: set USE_ARGPARSE = False to use the hardcoded parameters below.
# Toggle USE_ARGPARSE = True to read from CLI flags again.
# -----------------------------------------------------------------------------
USE_ARGPARSE = False
HARD_CONFIG = dict(
    model="bisenetv2dualmaskguidedv2",
    # model="bisenetv2",
    # model="litehrnet",
    encoder=None,
    decoder=None,
    # encoder_weights="imagenet",
    num_classes=2,
    use_aux=True,
    # use_aux=False,
    ckpt="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/best.pth",
    img_dir= "/media/ckchng/internal2TB/FILTERED_IMAGES/",
    anno_json= "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_and_rt_two_stage_labeled_merged_labels.json",
    num_angles=192,
    num_rhos=416,
    sep=True,

    # input="/path/to/your/input_image.png",
    # output_dir="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_15_new_bg_longer_dimmer/rt_map_only/single_stage/vis/",
    output_dir="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage/vis/",
    device="auto",
    batch_size=64,
    scale=1.0,
    mean=(0.30566086, 0.30566086, 0.30566086),
    std=(0.21072077, 0.21072077, 0.21072077),
    mean2=(0.34827731, 0.34827731, 0.34827731),
    std2=(0.16927711, 0.16927711, 0.16927711),
    palette="cityscapes",
    blend=True,
    blend_alpha=0.3,
    tile_size=288,
    tile_stride=144,  # overlap = tile_size - stride
    sep_params=[3.0, 6, 5.5, 0.6, 6.0, 0.1],
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



def parse_floats(values: Sequence[str]) -> Tuple[float, float, float]:
    vals = [float(v) for v in values]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("mean/std must have exactly 3 values (R G B).")
    return tuple(vals)  # type: ignore[return-value]


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
    image: np.ndarray, tile_size: int, stride: int, max_rho: float, itheta: float, irho: float, params: List[float], sep: bool = False, pad_value: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[List[dict], float, float]:
    """Slice an image into overlapping tiles. Pads on the bottom/right if needed."""
    h, w, _ = image.shape
    tiles = []
    
    sep_time_total = 0.0
    ht_time_total = 0.0
    
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
            # TODO: time this module separately from the RT, I want to see how much time the sep module takes
            t0_sep = time()
            if sep:
                objs_full, round_mask_full, keep_full, rms_full, removed_obj_bboxes_full = detect_roundish(tile[:, :, 0],
                                                                                                           params[0], params[1], 
                                                                                                           params[2], params[3], 
                                                                                                           params[4], deblend_cont=params[5])
                processed_tile = remove_masked_with_zero(tile[:, :, 0], round_mask_full)    
            else:
                processed_tile = tile[:, :, 0]
            sep_time_total += time() - t0_sep

            processed_tile = np.stack((processed_tile,)*3, axis=-1)

            # TODO: time this module separately from the sep, I want to see how much time the hough transform module takes, and also compare the hough map with and without sep
            t0_ht = time()
            # Ensure the input to Numba is a contiguous array, usually float32 is better for performance if the function allows it.
            # You can test with float32 depending on HT_utils implementation. Given `processed_tile` data is probably uint8
            # coming from image, let's keep the C-contiguous structure.
            ht_input = np.ascontiguousarray(processed_tile[:, :, 0].astype(np.float32))
            
            # rt_map, theta_deg, rhos = hough_bruteforce_intensity_numba(tile[:, :, 0], theta_res_deg=itheta, rho_res=irho)
            rt_map, theta_deg, rhos = hough_bruteforce_intensity_numba_dh(ht_input, max_rho/2, theta_res_deg=itheta, rho_res=irho)
            ht_time_total += time() - t0_ht
            
            # Fast C++ normalization using OpenCV to avoid massive memory float allocations
            rt_map_uint8 = np.zeros_like(rt_map, dtype=np.uint8)
            cv2.normalize(rt_map, rt_map_uint8, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # pad the rt_map to be divisible by 32
            # pad_h = (32 - rt_map.shape[0] % 32) % 32
            # pad_w = (32 - rt_map.shape[1] % 32) % 32
            # rt_map = cv2.copyMakeBorder(rt_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            
            # Fast repeat channels with OpenCV
            rt_map = cv2.cvtColor(rt_map_uint8, cv2.COLOR_GRAY2RGB)
            th, tw, _ = rt_map.shape

            # Fast single-channel normalisation to avoid float32 array replications across 3 channels
            scaled_tile_1c = np.zeros(processed_tile.shape[:2], dtype=np.uint8)
            arr = processed_tile[:, :, 0]
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            cv2.normalize(arr, scaled_tile_1c, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            scaled_tile_1c[arr == 0] = 0
            scaled_tile = cv2.cvtColor(scaled_tile_1c, cv2.COLOR_GRAY2RGB)
            # scaled_tile = np.zeros_like(scaled_tile)


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
    return tiles, sep_time_total, ht_time_total

def _bresenham_line(y0, x0, y1, x1):
    """
    Integer Bresenham line algorithm.
    Returns two 1D arrays: rows (y) and cols (x) of pixels on the line.
    """
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)

    dy = abs(y1 - y0)
    dx = abs(x1 - x0)

    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1

    err = dx - dy

    ys = []
    xs = []

    while True:
        ys.append(y0)
        xs.append(x0)

        if y0 == y1 and x0 == x1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return np.array(ys), np.array(xs)


def lines_to_binary_mask(lines, image_shape):
    """
    Create a binary mask with foreground pixels where lines are drawn.

    Parameters
    ----------
    lines : iterable
        Each element is ((y1, x1), (y2, x2)) specifying endpoints of a line
        in image coordinates (row = y, col = x). Floats will be rounded.
    image_shape : tuple
        (H, W) of the output mask.

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape (H, W), dtype uint8, where 1 = line, 0 = background.
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for (x1, y1), (x2, y2) in lines:
        # round in case inputs are floats
        y1_r, x1_r = int(round(y1)), int(round(x1))
        y2_r, x2_r = int(round(y2)), int(round(x2))

        rr, cc = _bresenham_line(y1_r, x1_r, y2_r, x2_r)

        # clip to image bounds
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr = rr[valid]
        cc = cc[valid]

        mask[rr, cc] = 1

    return mask

def save_prediction(
    pred_indices: np.ndarray,
    colors: torch.Tensor,
    original_rgb: np.ndarray,
    out_path: str,
    blend: bool,
    blend_alpha: float,
):
    if colors is not None:
        color_mask = colors[pred_indices].byte().cpu().numpy()
        mask_img = Image.fromarray(color_mask)
    else:
        mask_img = Image.fromarray(pred_indices.astype(np.uint8), mode="L")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # mask_img.save(out_path)

    if blend:
        base = Image.fromarray(original_rgb)
        if mask_img.mode != "RGB":
            mask_img = mask_img.convert("RGB")
        blend_path = out_path.replace(".png", "_blend.png")
        blended = Image.blend(base, mask_img, blend_alpha)
        blended.save(blend_path)



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

from typing import Tuple, Optional, Union
import math

Number = Union[int, float]

def tile_for_centering_line(
    p1: Tuple[Number, Number],
    p2: Tuple[Number, Number],
    tile_w: int,
    tile_h: int,
    image_w: Optional[int] = None,
    image_h: Optional[int] = None,
    snap_to_grid: bool = False,
    snap_mode: str = "round",   # "round" (nearest), "floor", or "ceil"
    return_tile_index: bool = False,
) -> Tuple[int, int]:
    """
    Compute the tile location that places the line segment (p1->p2) in the center of the tile.

    Parameters
    ----------
    p1, p2 : (x, y)
        Endpoints of the line segment in image coordinates (pixels).
    tile_w, tile_h : int
        Tile width/height in pixels.
    image_w, image_h : int or None
        If provided, tile origin is clamped so the tile stays inside the image.
    snap_to_grid : bool
        If True, tile origin is snapped to a regular grid: origin_x is multiple of tile_w,
        origin_y is multiple of tile_h.
    snap_mode : {"round","floor","ceil"}
        How to snap to grid.
    return_tile_index : bool
        If True, return (col, row) tile indices instead of (x0, y0) pixel origin.

    Returns
    -------
    (x0, y0) as ints (top-left pixel of tile) OR (col, row) indices if return_tile_index=True.
    """

    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("tile_w and tile_h must be positive integers.")

    x1, y1 = p1
    x2, y2 = p2

    # 1) Use segment midpoint as the target point to center
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    # 2) Tile origin that makes (mx, my) the tile center
    x0 = mx - tile_w / 2.0
    y0 = my - tile_h / 2.0

    # 3) Optionally snap to a (tile_w, tile_h) grid
    if snap_to_grid:
        def snap(v: float, step: int) -> int:
            t = v / step
            if snap_mode == "round":
                k = int(round(t))
            elif snap_mode == "floor":
                k = int(math.floor(t))
            elif snap_mode == "ceil":
                k = int(math.ceil(t))
            else:
                raise ValueError("snap_mode must be one of: 'round', 'floor', 'ceil'")
            return k * step

        x0 = snap(x0, tile_w)
        y0 = snap(y0, tile_h)
    else:
        # If not snapping, you still need an integer pixel origin.
        # floor is common; round is also reasonable. Here we use floor for stability.
        x0 = int(math.floor(x0))
        y0 = int(math.floor(y0))

    # 4) Optionally clamp to image bounds so tile is fully inside image
    if image_w is not None and image_h is not None:
        if image_w < tile_w or image_h < tile_h:
            raise ValueError("Image is smaller than the tile size.")
        x0 = max(0, min(int(x0), image_w - tile_w))
        y0 = max(0, min(int(y0), image_h - tile_h))
    else:
        x0 = int(x0)
        y0 = int(y0)

    # 5) Optionally return tile indices (col,row) instead of pixel origin
    if return_tile_index:
        # Only meaningful if using a grid; otherwise this is a “virtual” index.
        col = int(math.floor(x0 / tile_w))
        row = int(math.floor(y0 / tile_h))
        return (col, row)

    return (x0, y0)


# -----------------------------------------------------------------------------
# Main prediction flow
# -----------------------------------------------------------------------------
def run(args):
    # check if output_dir exists
    # if not, makedirs
    os.makedirs(os.path.join(args.output_dir, 'full_frame_result', 'txt'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'full_frame_result', 'vis'), exist_ok=True)
    
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    config = PredictConfig(
        model=args.model,
        num_class=args.num_classes,
        encoder=args.encoder,
        decoder=args.decoder,
        # encoder_weights=args.encoder_weights,
        use_aux=args.use_aux,
        colormap=args.palette if args.palette == "cityscapes" else "cityscapes",  # used only when palette == cityscapes
    )
    ## setup model, load checkpoint
    model = get_model(config).to(device)
    model.eval()
    load_checkpoint(model, args.ckpt, device)

    is_dual = 'dual' in args.model.lower()
    is_maskguided = 'maskguided' in args.model.lower()
    if is_dual:
        # Split transforms for dual stream; keep shapes as-is (no paired padding)
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
    
    colors = prepare_palette(args, config)

    
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
    irho = max_rho / (args.num_rhos - 1)

    tp_dir = os.path.join(args.output_dir, 'tile_tp')
    fp_dir = os.path.join(args.output_dir, 'tile_fp')
    fn_dir = os.path.join(args.output_dir, 'tile_fn')
    category_ids_for_split = [1, 9, 10]
    category_for_not_relevant = 3
    category_for_hard_to_cat = 5

    os.makedirs(tp_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)
    for category_id in category_ids_for_split:
        os.makedirs(os.path.join(tp_dir, str(category_id)), exist_ok=True)
        os.makedirs(os.path.join(fn_dir, str(category_id)), exist_ok=True)

    os.makedirs(os.path.join(tp_dir, str(category_for_hard_to_cat)), exist_ok=True)
    os.makedirs(os.path.join(fn_dir, str(category_for_hard_to_cat)), exist_ok=True)

    #### prepare RT params
    thetas_deg, thetas, cos_t, sin_t, rhos = _make_params_dh(max_rho/2, theta_res_deg=itheta, rho_res=irho)

    # Dictionary to store times for each module per image
    timing_dict = {
        'tile_image': [],
        'sep': [],
        'hough': [],
        'tile_processing_loop': [],
        'model_inference': [],
        'post_process': [],
        'merge_lines': [],
        'total_image': []
    }

    # for int_id in range(len(image_dirs)):
    for int_id in range(100):
        # t0_image = time()

        # check if the output file exists
        out_path = os.path.join(args.output_dir, 'full_frame_result/vis', f"{os.path.splitext(os.path.basename(image_dirs[int_id]['file_name']))[0]}_lines.png")

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
        # zero_mask = img == 0
        # img = img / img.min()
        # img = img / img.max() * 255.0
        # img[zero_mask] = 0
        # img = img.astype(np.uint8)
        # stack it to create 3 channels
        image_rgb = np.stack((img,)*3, axis=-1)
        img_scaled = img.copy()
        zero_mask = img_scaled == 0
        img_scaled = img_scaled - img_scaled.min()
        img_scaled = img_scaled / img_scaled.max() * 255.0
        img_scaled[zero_mask] = 0
        img_scaled = img_scaled.astype(np.uint8)
        img_scaled = np.stack((img_scaled,)*3, axis=-1)
        img_scaled_for_fn = img_scaled.copy()
        # TODO: time this module
        t0_tile_image = time()
        tiles, sep_time_total, ht_time_total = tile_image(image_rgb, tile_size=args.tile_size, stride=args.tile_stride, max_rho=max_rho, itheta=itheta, irho=irho, params=args.sep_params, sep=args.sep)
        timing_dict['tile_image'].append(time() - t0_tile_image)
        timing_dict['sep'].append(sep_time_total)
        timing_dict['hough'].append(ht_time_total)
        print(len(tiles))

        os.makedirs(args.output_dir, exist_ok=True)

        # Process tiles in batches
        # time start here
        # TODO: time the entire tile processing loop, and also time the model inference separately to see how much time it takes compared to the rest of the processing
        t0_tile_processing = time()
        model_inference_time = 0.0
        post_process_time = 0.0
        
        all_pred_lines = []
        preds_id_with_values = []
        for idx in range(0, len(tiles), args.batch_size):
            batch_tiles = tiles[idx : idx + args.batch_size]

            images_aug = []
            images2_aug = []
            for t in batch_tiles:
                if is_dual:
                    # 1. Scale
                    res1 = scale_transform(image=t["rt_map"])
                    res2 = scale_transform(image=t["scaled_img"])
                    img1_scaled = res1["image"]
                    img2_scaled = res2["image"]
                    
                    # 2. Normalize & ToTensor (no extra padding)
                    final1 = norm_transform1(image=img1_scaled)["image"]
                    final2 = norm_transform2(image=img2_scaled)["image"]
                    
                    images_aug.append(final1)
                    images2_aug.append(final2)
                else:
                    augmented = transform(image=t["rt_map"])
                    images_aug.append(augmented["image"])

            batch_tensor = torch.stack(images_aug).to(device=device, dtype=torch.float32)
            
            if is_dual:
                batch_tensor2 = torch.stack(images2_aug).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                t0_model = time()
                if is_dual:
                    logits = model(batch_tensor, batch_tensor2)
                else:
                    logits = model(batch_tensor)

                logits_main = logits
                logits_s2 = None

                # For maskguided dual models, expect two outputs; otherwise take single
                if isinstance(logits, (tuple, list)):
                    if is_maskguided and len(logits) >= 2:
                        logits_main, logits_s2 = logits[0], logits[1]
                    else:
                        logits_main = logits[0]
                # Compute main predictions (rt) first
                if args.num_classes == 2:
                    preds_rt_tensor = logits_main.argmax(dim=1)
                elif args.num_classes == 1:
                    preds_rt_tensor = (logits_main.sigmoid() > 0.5).squeeze(1).long()
                else:
                    preds_rt_tensor = logits_main.argmax(dim=1)

                # Compute streak predictions (second stream) if present; otherwise create a zero-mask
                if logits_s2 is None:
                    preds_streak_tensor = torch.zeros_like(preds_rt_tensor)
                else:
                    # Handle common shapes for logits_s2
                    try:
                        if args.num_classes == 2:
                            # multi-class map -> take argmax over channel dim
                            preds_streak_tensor = logits_s2.argmax(dim=1)
                        elif args.num_classes == 1:
                            preds_streak_tensor = (logits_s2.sigmoid() > 0.5).squeeze(1).long()
                        else:
                            preds_streak_tensor = logits_s2.argmax(dim=1)
                    except Exception:
                        # Fallback: create zero mask matching preds_rt shape
                        preds_streak_tensor = torch.zeros_like(preds_rt_tensor)
                model_inference_time += time() - t0_model

            # Move to CPU and numpy for downstream processing
            preds_rt = preds_rt_tensor.cpu().numpy()
            preds_streak = preds_streak_tensor.cpu().numpy()
            
            peak = []
            bboxes = []

            processed_batch_tiles = []
            # TODO: time this module
            t0_post = time()
            for pred, pstreak, tile in zip(preds_rt, preds_streak, batch_tiles):
                pred_lines = []
                pred_streak_mask = []
                if pred.max() == 0:
                    continue

                # Connected components: labels + statistics
                # stats: [label, x, y, width, height, area]
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    pred.astype(np.uint8), connectivity=8
                )

                # bboxes = []
                # Label 0 is background, skip it
                legit_pred_flag = False
                for label_id in range(1, num_labels):
                    x, y, w, h, area = stats[label_id]
                    if area == 0:
                        continue
                    x1, y1 = x, y
                    x2, y2 = x + w - 1, y + h - 1  # inclusive coordinates
                    if w < 10 or h < 10:
                        continue
                    bboxes.append((x1, y1, x2, y2))
                    
                    # find peak intensity location in original image
                    crop_rt_map = tile['rt_map'][y1:y2+1, x1:x2+1, 0]

                    # Find peak intensity location in cropped image
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(crop_rt_map)
                    peak_x = np.clip(maxLoc[0] + x1, 0, len(rhos) - 1)
                    peak_y = np.clip(maxLoc[1] + y1, 0, len(thetas) - 1)

                    pred_rho = rhos[int(round(peak_x))]
                    pred_theta = thetas[int(round(peak_y))]
                    
                    # remove lines with rho too close to the border
                    if pred_rho < (args.tile_size / 5) - (args.tile_size / 2) or pred_rho > (4 * (args.tile_size / 5)) - (args.tile_size / 2):
                        continue

                    # convert the peak back to line parameters
                    pred_p0, pred_p1 = line_endpoints_center_rho_theta(pred_rho, pred_theta, args.tile_size, args.tile_size)

                    if pred_p0 is None or pred_p1 is None:
                        continue
                    
                    # convert the endpoints to global coordinates
                    global_pred_p0 = (pred_p0[0] + tile['x'], pred_p0[1] + tile['y'])
                    global_pred_p1 = (pred_p1[0] + tile['x'], pred_p1[1] + tile['y'])                        

                    # append the line parameters
                    pred_lines.append([global_pred_p0[0], global_pred_p0[1], global_pred_p1[0], global_pred_p1[1]])
                    all_pred_lines.append([global_pred_p0[0], global_pred_p0[1], global_pred_p1[0], global_pred_p1[1]])
            post_process_time += time() - t0_post
            
        timing_dict['tile_processing_loop'].append(time() - t0_tile_processing)
        timing_dict['model_inference'].append(model_inference_time)
        timing_dict['post_process'].append(post_process_time)
                    
        # for each of the pred_lines, convert to ((x1, y1), (x2, y2))
        all_pred_lines = [((line[0], line[1]), (line[2], line[3])) for line in all_pred_lines]

        # use merge line
        from tools.merge_lines import merge_connected_segments_2d
        max_angle=5    # degrees
        max_dist=10.0    # pixels
        eps=1e-2
        # TODO: time this module too
        t0_merge = time()
        merged_lines = merge_connected_segments_2d(all_pred_lines, max_angle_deg=max_angle, max_dist_px=max_dist, eps=eps)
        timing_dict['merge_lines'].append(time() - t0_merge)

        # write a function to merge lines that are connected to each other
        # merged_lines = merge_connected_segments_2d(all_pred_lines, 1e-6)
        # merged_mask = lines_to_binary_mask(merged_lines, [img.shape[0], img.shape[1]]).astype(np.uint64)
        # output_image_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_mask.png")
        # save_prediction(
        #         pred_indices=merged_mask,
        #         colors=colors,
        #         original_rgb=img_scaled,
        #         out_path=output_image_path,
        #         blend=args.blend,
        #         blend_alpha=args.blend_alpha,
        #     )
            # check this first, if it works, then we proceed to evaluate the lines
            # visualise the merged lines on the original image
        
        # for curr_line, curr_category_id in zip(gt_lines, gt_category_ids):
        #     if curr_category_id not in category_ids_for_split:
        #         continue

        #     cv2.line(img_scaled, (int(round(curr_line[0])), int(round(curr_line[1]))), 
        #              (int(round(curr_line[2])), int(round(curr_line[3]))), (0, 0, 255), 5)

        # not_relevant_gt_lines = [
        #     curr_line
        #     for curr_line, curr_category_id in zip(gt_lines, gt_category_ids)
        #     if curr_category_id == category_for_not_relevant
        # ]


        # for line in merged_lines:
        #     p0, p1 = line
        #     cv2.line(img_scaled, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 0, 0), 2)
            # then blend it with img_scaled
            # img_scaled = cv2.addWeighted(img_scaled, 0.7, mask, 0.3, 0)


        # save the image with lines
        # output_image_path = os.path.join(args.output_dir, 'full_frame_result', 'vis', f"{os.path.splitext(os.path.basename(img_path))[0]}_lines.png")
        # cv2.imwrite(output_image_path, img_scaled)

        # save the merged lines to a text file
        # output_lines_path = os.path.join(args.output_dir, 'full_frame_result', 'txt', f"{os.path.splitext(os.path.basename(img_path))[0]}_lines.txt")
        # with open(output_lines_path, "w") as f:
        #     for line in merged_lines:
        #         f.write(f"{line[0][0]},{line[0][1]},{line[1][0]},{line[1][1]}\n")

        timing_dict['total_image'].append(time() - t0_image)
        print(f"done processing {img_name}")

    print("Timing Statistics (vectors of time per image):")
    for k, v in timing_dict.items():
        print(f"  {k}: mean={np.mean(v):.4f}s, std={np.std(v):.4f}s")
        
    timing_file = os.path.join(args.output_dir, "timing_dict.json")
    with open(timing_file, "w") as f:
        json.dump(timing_dict, f, indent=4)
    print(f"Saved timing details to {timing_file}")
    
    return timing_dict




def parse_args():
    parser = argparse.ArgumentParser(description="Standalone prediction script for realtime semantic segmentation.")
    parser.add_argument("--model", type=str, default="litehrnet", help="Model name registered in models/model_registry.")
    parser.add_argument("--encoder", type=str, default=None, help="Encoder (only needed if model == smp).")
    parser.add_argument("--decoder", type=str, default=None, help="Decoder (only needed if model == smp).")
    parser.add_argument("--encoder-weights", type=str, default="imagenet", help="Encoder weights for smp models.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes.")
    parser.add_argument("--use-aux", action="store_true", help="Set if the checkpoint was trained with auxiliary heads.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth) containing state_dict.")
    parser.add_argument("--img-dir", type=str, default=None, help="Directory containing .npy images (for name remap).")
    parser.add_argument("--anno-npz", type=str, required=True, help="Path to .npz annotations with imgPath and XY.")
    parser.add_argument("--output-dir", type=str, default="predictions", help="Where to save masks/overlays.")
    parser.add_argument("--device", type=str, default="auto", help="Device string (e.g., cuda, cuda:0, cpu, or auto).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor applied before normalization.")
    parser.add_argument("--sep", type=bool, default=True, help="Whether to use separate processing.")
    parser.add_argument(
        "--mean",
        type=parse_floats,
        nargs=3,
        default=(0.30566086, 0.30566086, 0.30566086),
        help="Normalization mean (R G B). Default matches custom dataset in this repo.",
    )
    parser.add_argument(
        "--std",
        type=parse_floats,
        nargs=3,
        default=(0.21072077, 0.21072077, 0.21072077),
        help="Normalization std (R G B). Default matches custom dataset in this repo.",
    )
    parser.add_argument(
        "--mean2",
        type=parse_floats,
        nargs=3,
        default=(0.34827731, 0.34827731, 0.34827731),
        help="Normalization mean for second stream (R G B).",
    )
    parser.add_argument(
        "--std2",
        type=parse_floats,
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
    return parser.parse_args()


if __name__ == "__main__":
    if USE_ARGPARSE:
        args = parse_args()
    else:
        args = argparse.Namespace(**HARD_CONFIG)

    # Grid search for sep_params: [thresh_sigma, minarea, elong_max, r_eff_min, r_eff_max, hough_thresh]
    param_grid = [
        [3.0, 6, 5.0, 0.6, 6.0, 0.1]
    ]
    
    original_output_dir = args.output_dir
    for params in param_grid:
        param_str = "_".join([str(p) for p in params])
        args.output_dir = os.path.join(original_output_dir, f"params_{param_str}")
        args.sep_params = params
        print(f"\n>>> Running with params: {params} output_dir: {args.output_dir}")
        run(args)
