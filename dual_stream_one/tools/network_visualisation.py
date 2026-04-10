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
from utils import transforms
from utils.utils import get_colormap
import cv2
import json
from numba import prange, njit
from matplotlib import pyplot as plt
# from line_eval import line_intersection_check, line_angle_degrees
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
    num_classes=1,
    use_aux=True,
    # use_aux=False,
    ckpt="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_bce/best.pth",
    img_dir= "/media/ckchng/internal2TB/FILTERED_IMAGES/",
    anno_json= "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.json",
    num_angles=192,
    num_rhos=416,
    # img_name='000_2020-12-08_103708_E_DSC_0487',
    # crop_xmin=0,
    # crop_ymin=3168,
    # img_name='017_2020-12-08_101128_E_DSC_0332',
    # crop_xmin=6768,
    # crop_ymin=3888,
    # crop_ymin=3788,
    # crop_ymin=4500,
    # img_name='000_2020-12-08_095228_E_DSC_0219',
    # crop_xmin=7091,
    # crop_ymin=2160,
    # img_name='000_2020-12-08_095238_E_DSC_0220',
    # crop_xmin=6336,
    # crop_ymin=0,
    # img_name='025_2020-12-08_134639_E_DSC_1623',
    # crop_xmin=2304,
    # crop_ymin=0,
    # img_name='000_2020-12-08_104028_E_DSC_0507',
    
    # crop_xmin=6912, 
    # crop_ymin=4032,
    # img_name='000_2020-12-08_105108_E_DSC_0571',
    crop_xmin=6624,
    crop_ymin=3888,
    img_name='017_2020-12-08_101128_E_DSC_0332',
    # '017_2020-12-08_101128_E_DSC_0332_y3888_x6624_pred0.png'


    # input="/path/to/your/input_image.png",
    output_dir="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_bce/",
    device="auto",
    batch_size=128,
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
    image: np.ndarray, tile_size: int, stride: int, max_rho: float, itheta: float, irho: float, pad_value: Tuple[int, int, int] = (0, 0, 0)
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
            # rt_map, theta_deg, rhos = hough_bruteforce_intensity_numba(tile[:, :, 0], theta_res_deg=itheta, rho_res=irho)
            rt_map, theta_deg, rhos = hough_bruteforce_intensity_numba_dh(tile[:, :, 0], max_rho/2, theta_res_deg=itheta, rho_res=irho)
            
            rt_map = rt_map - rt_map.min()
            rt_map = rt_map / rt_map.max() * 255
            rt_map = rt_map.astype(np.uint8)
            # pad the rt_map to be divisible by 32
            pad_h = (32 - rt_map.shape[0] % 32) % 32
            pad_w = (32 - rt_map.shape[1] % 32) % 32
            rt_map = cv2.copyMakeBorder(rt_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            # pad rt_map to 3 channels
            
            rt_map = rt_map[:, :, np.newaxis]
            rt_map = np.repeat(rt_map, 3, axis=2)
            th, tw, _ = rt_map.shape

            scaled_tile = tile.copy()

            zero_mask = scaled_tile == 0
            scaled_tile = scaled_tile - scaled_tile.min()
            scaled_tile = scaled_tile / scaled_tile.max() * 255.0
            scaled_tile[zero_mask] = 0
            scaled_tile = scaled_tile.astype(np.uint8)


            # if th != num_angles or tw != num_rhos:
            #     print('something is wrong')

            tiles.append(
                {
                    'ori_img': tile,
                    "scaled_img": scaled_tile,
                    "rt_map": rt_map,  # save the tile instead
                    "y": y,
                    "x": x,
                    "h": th,
                    "w": tw,
                }
            )
    return tiles

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


import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_4d(feat: ArrayLike, batch_index: int = 0) -> torch.Tensor:
    """
    Convert feature map to torch.Tensor of shape [1, C, H, W].
    Accepts [C,H,W] or [B,C,H,W].
    """
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)
    if not torch.is_tensor(feat):
        raise TypeError("feat must be a torch.Tensor or numpy.ndarray")

    if feat.ndim == 3:
        feat = feat.unsqueeze(0)  # [1,C,H,W]
    elif feat.ndim == 4:
        feat = feat[batch_index : batch_index + 1]
    else:
        raise ValueError(f"feat must have 3 or 4 dims, got {feat.ndim}")

    # Ensure float for visualization math
    if not torch.is_floating_point(feat):
        feat = feat.float()

    return feat


def _normalize_map(
    x: torch.Tensor,
    per_channel: bool = True,
    percentile: Tuple[float, float] = (1.0, 99.0),
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize tensor to [0,1] with optional percentile clipping.
    If per_channel=True, expects x shape [C,H,W] and normalizes each channel separately.
    Else normalizes the whole tensor globally.
    """
    x = x.detach().cpu()

    if per_channel:
        if x.ndim != 3:
            raise ValueError("per_channel normalization expects x with shape [C,H,W]")
        C = x.shape[0]
        out = torch.empty_like(x)
        for c in range(C):
            xc = x[c]
            lo = torch.quantile(xc.flatten(), percentile[0] / 100.0)
            hi = torch.quantile(xc.flatten(), percentile[1] / 100.0)
            xc = torch.clamp(xc, lo, hi)
            out[c] = (xc - xc.min()) / (xc.max() - xc.min() + eps)
        return out
    else:
        lo = torch.quantile(x.flatten(), percentile[0] / 100.0)
        hi = torch.quantile(x.flatten(), percentile[1] / 100.0)
        x = torch.clamp(x, lo, hi)
        return (x - x.min()) / (x.max() - x.min() + eps)


def plot_topk_montage(
    feat: ArrayLike,
    k: int = 16,
    score: str = "mean_abs",
    batch_index: int = 0,
    cols: int = 8,
    per_channel_norm: bool = True,
    percentile: Tuple[float, float] = (1.0, 99.0),
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
):
    """
    Plot a montage of top-K channels from a CNN feature map.

    Parameters
    ----------
    feat : [C,H,W] or [B,C,H,W]
    k : number of channels to show
    score : channel ranking criterion:
        - "mean_abs": mean(|F|) over H,W
        - "max_abs":  max(|F|) over H,W
        - "mean":     mean(F)   over H,W
        - "var":      var(F)    over H,W
    per_channel_norm : if True, each channel is normalized independently for visibility
    percentile : clipping percentiles before normalization (helps avoid outliers dominating)
    """
    x = _to_4d(feat, batch_index=batch_index)  # [1,C,H,W]
    x0 = x[0]  # [C,H,W]
    C = x0.shape[0]

    if score == "mean_abs":
        s = x0.abs().mean(dim=(1, 2))
    elif score == "max_abs":
        s = x0.abs().amax(dim=(1, 2))
    elif score == "mean":
        s = x0.mean(dim=(1, 2))
    elif score == "var":
        s = x0.var(dim=(1, 2), unbiased=False)
    else:
        raise ValueError(f"Unknown score='{score}'")

    k = min(k, C)
    top_idx = torch.topk(s, k=k, largest=True).indices

    # Normalize for display
    disp = x0[top_idx]  # [k,H,W]
    disp = _normalize_map(disp, per_channel=per_channel_norm, percentile=percentile)

    rows = math.ceil(k / cols)
    if figsize is None:
        figsize = (cols * 2.2, rows * 2.2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < k:
            ch = int(top_idx[i].item())
            ax.imshow(disp[i].numpy(), cmap="gray", interpolation="nearest")
            ax.set_title(f"ch {ch}\n{score}={s[ch].item():.3g}", fontsize=9)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)

    fig.tight_layout()
    plt.show()


def plot_energy_overlay(
    feat: ArrayLike,
    image: Optional[ArrayLike] = None,
    method: str = "rms",
    batch_index: int = 0,
    upsample_to_image: bool = True,
    alpha: float = 0.45,
    percentile: Tuple[float, float] = (1.0, 99.0),
    figsize: Tuple[float, float] = (6.5, 6.5),
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    show: bool = True,
):
    """
    Compute a spatial "energy" heatmap from feature maps and (optionally) overlay on an input image.

    Energy map collapses channels: [C,H,W] -> [H,W]
    - method="rms": sqrt(mean_c(F^2))
    - method="mean_abs": mean_c(|F|)

    image (optional) can be:
    - [H,W] grayscale or [H,W,3] RGB
    - values in [0,1] or [0,255]

    If image is provided and upsample_to_image=True, heatmap is resized to match image size.
    """
    x = _to_4d(feat, batch_index=batch_index)  # [1,C,H,W]
    x0 = x[0]  # [C,H,W]

    if method == "mean":
        E = x0.mean(dim=0) 
    elif method == "rms":
        E = (x0.pow(2).mean(dim=0)).sqrt()  # [H,W]
    elif method == "mean_abs":
        E = x0.abs().mean(dim=0)  # [H,W]
    else:
        raise ValueError(f"Unknown method='{method}'")

    # Normalize E to [0,1] for display (global)
    E_disp = _normalize_map(E, per_channel=False, percentile=percentile)

    # Prepare image
    if image is not None:
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu().numpy()
        else:
            img = np.asarray(image)

        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            # likely CHW -> HWC
            img = np.transpose(img, (1, 2, 0))

        # Convert to float [0,1]
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)

        H_img, W_img = img.shape[:2]

        # Upsample heatmap to match image size
        if upsample_to_image:
            E_up = F.interpolate(
                E_disp.unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
                size=(H_img, W_img),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        else:
            E_up = E_disp

        E_show = E_up.numpy()

        if ax is None:
            fig, ax_local = plt.subplots(1, 1, figsize=figsize)
            ax_use = ax_local
        else:
            fig = None
            ax_use = ax

        ax_use.imshow(img, interpolation="nearest")
        ax_use.imshow(E_show, cmap="jet", alpha=alpha, interpolation="nearest")
        ax_use.axis("off")
        if title is None:
            title = f"Energy overlay ({method})"
        ax_use.set_title(title)

        if show and fig is not None:
            plt.show()

    else:
        # Just show the heatmap itself
        if ax is None:
            fig, ax_local = plt.subplots(1, 1, figsize=figsize)
            ax_use = ax_local
        else:
            fig = None
            ax_use = ax

        ax_use.imshow(E_disp.numpy(), cmap="jet", interpolation="nearest")
        ax_use.axis("off")
        if title is None:
            title = f"Energy map ({method})"
        ax_use.set_title(title)

        if show and fig is not None:
            plt.show()

    return E_disp  # torch.Tensor [H,W], normalized to [0,1]


def plot_featuremap_energy_overlays_side_by_side(
    feature_map: ArrayLike,
    image: ArrayLike,
    out_path: str,
    method: str = "mean",
    alpha: float = 0.45,
    percentile: Tuple[float, float] = (1.0, 99.0),
    max_cols: int = 4,
    dpi: int = 200,
    show: bool = True,
):
    """Plot energy overlays for each `feat` in `feature_map` side-by-side and save.

    This matches the existing behavior of:
        for feat in feature_map:
            plot_energy_overlay(feat, image=scaled_tile, method=...)
    but renders them into a single grid figure.
    """
    # `feature_map` is typically torch.Tensor [B,C,H,W]. Iteration yields [C,H,W].
    if isinstance(feature_map, np.ndarray):
        feats = list(feature_map)
    elif isinstance(feature_map, torch.Tensor):
        feats = list(feature_map)
    else:
        feats = list(feature_map)  # allow iterables

    if len(feats) == 0:
        raise ValueError("feature_map is empty")

    k = len(feats)
    cols = min(max_cols, k)
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i >= k:
            continue

        plot_energy_overlay(
            feats[i],
            image=image,
            method=method,
            alpha=alpha,
            percentile=percentile,
            title=f"feat[{i}] ({method})",
            ax=ax,
            show=False,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _overlay_image_for_layer_name(layer_name: str, rt_map: ArrayLike, scaled_tile: ArrayLike) -> ArrayLike:
    """Pick overlay background for a given layer name.

    - names containing 'dht' -> rt_map
    - names containing '1' -> rt_map
    - names containing '2' -> scaled_tile
    - otherwise -> scaled_tile
    """
    lname = str(layer_name).lower()
    if "dht" in lname:
        return rt_map
    if "1" in lname:
        return rt_map
    if "2" in lname:
        return scaled_tile
    return scaled_tile


def _group_for_name(name: str) -> int | None:
    """Return group id (1 or 2) based on name.

    Rules:
    - contains 'dht' -> 2
    - contains '1'   -> 1
    - contains '2'   -> 2
    - else           -> None
    """
    lname = str(name).lower()
    if "dht" in lname:
        return 2
    if "1" in lname:
        return 1
    if "2" in lname:
        return 2
    return None


def save_grouped_features_and_predictions_grids(
    activations: dict,
    preds_rt_raw: np.ndarray,
    preds_rt: np.ndarray,
    preds_streak_raw: np.ndarray,
    preds_streak: np.ndarray,
    rt_map: ArrayLike,
    scaled_tile: ArrayLike,
    out_path_group1: str,
    out_path_group2: str,
    method: str = "mean",
    alpha: float = 0.45,
    percentile: Tuple[float, float] = (1.0, 99.0),
    max_cols: int = 4,
    dpi: int = 200,
    show: bool = True,
):
    def _prep_bg(img: ArrayLike) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    def _norm01(x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(x)
        xt = xt.squeeze()
        xt = xt.float()
        xt = _normalize_map(xt, per_channel=False, percentile=percentile)
        return xt.numpy()

    def _save_grid(panels: List[Tuple[str, str, object]], out_path: str):
        # panel tuple: (kind, title, payload)
        if len(panels) == 0:
            return None

        k = len(panels)
        cols = min(max_cols, k)
        rows = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
        axes = np.array(axes).reshape(rows, cols)

        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.axis("off")
            if i >= k:
                continue
            kind, title, payload = panels[i]

            if kind == "energy":
                feat, overlay_img = payload
                plot_energy_overlay(
                    feat,
                    image=overlay_img,
                    method=method,
                    alpha=alpha,
                    percentile=percentile,
                    title=title,
                    ax=ax,
                    show=False,
                )
                continue

            if kind == "pred_raw":
                pred_arr, bg_img = payload
                ax.imshow(bg_img, interpolation="nearest")
                ax.imshow(_norm01(pred_arr), cmap="jet", alpha=alpha, interpolation="nearest")
                ax.set_title(title)
                ax.axis("off")
                continue

            if kind == "pred_bin":
                pred_arr, bg_img = payload
                ax.imshow(bg_img, interpolation="nearest")
                ax.imshow(
                    np.asarray(pred_arr).squeeze().astype(np.float32),
                    cmap="Reds",
                    alpha=alpha,
                    interpolation="nearest",
                    vmin=0,
                    vmax=1,
                )
                ax.set_title(title)
                ax.axis("off")
                continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    # Build feature panels, grouped by layer name.
    group1: List[Tuple[str, str, object]] = []
    group2: List[Tuple[str, str, object]] = []

    for name, data in activations.items():
        gid = _group_for_name(str(name))
        if gid is None:
            continue
        lname = str(name).lower()
        # Special case: keep dht in group 2, but overlay it on rt_map.
        overlay_img = rt_map if ("dht" in lname or gid == 1) else scaled_tile

        if isinstance(data, torch.Tensor):
            for bi, feat in enumerate(list(data)):
                title = f"{name}[{bi}] ({method})" if data.ndim == 4 else f"{name} ({method})"
                (group1 if gid == 1 else group2).append(("energy", title, (feat, overlay_img)))
            continue

        if isinstance(data, (list, tuple)):
            for si, sub in enumerate(data):
                if not isinstance(sub, torch.Tensor):
                    continue
                if sub.ndim == 4:
                    for bi, feat in enumerate(list(sub)):
                        title = f"{name}[{si}][{bi}] ({method})"
                        (group1 if gid == 1 else group2).append(("energy", title, (feat, overlay_img)))
                else:
                    title = f"{name}[{si}] ({method})"
                    (group1 if gid == 1 else group2).append(("energy", title, (sub, overlay_img)))

    # Add prediction panels: RT -> group 1, Streak -> group 2.
    rt_bg = _prep_bg(rt_map)
    st_bg = _prep_bg(scaled_tile)

    group1.extend(
        [
            ("pred_raw", "preds_rt_raw", (preds_rt_raw, rt_bg)),
            ("pred_bin", "preds_rt", (preds_rt, rt_bg)),
        ]
    )
    group2.extend(
        [
            ("pred_raw", "preds_streak_raw", (preds_streak_raw, st_bg)),
            ("pred_bin", "preds_streak", (preds_streak, st_bg)),
        ]
    )

    saved1 = _save_grid(group1, out_path_group1)
    saved2 = _save_grid(group2, out_path_group2)
    return saved1, saved2


def save_all_layers_energy_overlays_grid(
    activations: dict,
    rt_map: ArrayLike,
    scaled_tile: ArrayLike,
    out_path: str,
    method: str = "mean",
    alpha: float = 0.45,
    percentile: Tuple[float, float] = (1.0, 99.0),
    max_cols: int = 4,
    dpi: int = 200,
    show: bool = True,
):
    """Save energy overlays for ALL extracted layers in a single grid.

    Handles activations that are:
    - torch.Tensor [B,C,H,W]
    - list/tuple of tensors (common for multi-scale outputs)
    """

    panels: List[Tuple[str, ArrayLike, ArrayLike]] = []  # (title, feat, overlay_image)

    for name, data in activations.items():
        overlay_img = _overlay_image_for_layer_name(str(name), rt_map=rt_map, scaled_tile=scaled_tile)

        if isinstance(data, torch.Tensor):
            # Usually [B,C,H,W]. Iteration yields [C,H,W].
            for bi, feat in enumerate(list(data)):
                title = f"{name}[{bi}] ({method})" if data.ndim == 4 else f"{name} ({method})"
                panels.append((title, feat, overlay_img))
            continue

        if isinstance(data, (list, tuple)):
            for si, sub in enumerate(data):
                if not isinstance(sub, torch.Tensor):
                    continue
                if sub.ndim == 4:
                    for bi, feat in enumerate(list(sub)):
                        panels.append((f"{name}[{si}][{bi}] ({method})", feat, overlay_img))
                else:
                    panels.append((f"{name}[{si}] ({method})", sub, overlay_img))

    if len(panels) == 0:
        raise ValueError("No plottable activations found in activations dict")

    k = len(panels)
    cols = min(max_cols, k)
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i >= k:
            continue
        title, feat, overlay_img = panels[i]
        plot_energy_overlay(
            feat,
            image=overlay_img,
            method=method,
            alpha=alpha,
            percentile=percentile,
            title=title,
            ax=ax,
            show=False,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def save_predictions_visualisation(
    preds_rt_raw: np.ndarray,
    preds_rt: np.ndarray,
    preds_streak_raw: np.ndarray,
    preds_streak: np.ndarray,
    rt_map: ArrayLike,
    scaled_tile: ArrayLike,
    out_path: str,
    alpha_raw: float = 0.45,
    alpha_bin: float = 0.45,
    percentile: Tuple[float, float] = (1.0, 99.0),
    dpi: int = 200,
    show: bool = True,
):
    def _prep_bg(img: ArrayLike) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    def _norm01(x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(x)
        if xt.ndim == 3:
            # accept [B,H,W] -> take first
            xt = xt[0]
        xt = xt.float()
        xt = _normalize_map(xt, per_channel=False, percentile=percentile)
        return xt.numpy()

    # Squeeze any [B,H,W] into [H,W] for display.
    rt_raw = np.squeeze(preds_rt_raw)
    rt_bin = np.squeeze(preds_rt)
    st_raw = np.squeeze(preds_streak_raw)
    st_bin = np.squeeze(preds_streak)

    rt_bg = _prep_bg(rt_map)
    st_bg = _prep_bg(scaled_tile)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = np.array(axes).reshape(2, 2)

    # RT raw
    ax = axes[0, 0]
    ax.imshow(rt_bg, interpolation="nearest")
    ax.imshow(_norm01(rt_raw), cmap="jet", alpha=alpha_raw, interpolation="nearest")
    ax.set_title("preds_rt_raw")
    ax.axis("off")

    # RT bin
    ax = axes[0, 1]
    ax.imshow(rt_bg, interpolation="nearest")
    ax.imshow(rt_bin.astype(np.float32), cmap="Reds", alpha=alpha_bin, interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("preds_rt")
    ax.axis("off")

    # Streak raw
    ax = axes[1, 0]
    ax.imshow(st_bg, interpolation="nearest")
    ax.imshow(_norm01(st_raw), cmap="jet", alpha=alpha_raw, interpolation="nearest")
    ax.set_title("preds_streak_raw")
    ax.axis("off")

    # Streak bin
    ax = axes[1, 1]
    ax.imshow(st_bg, interpolation="nearest")
    ax.imshow(st_bin.astype(np.float32), cmap="Reds", alpha=alpha_bin, interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("preds_streak")
    ax.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# -------------------------
# Example usage (assuming you already have feature maps)
# -------------------------
# feat = ...  # torch.Tensor [B,C,H,W] or [C,H,W]
# img  = ...  # torch.Tensor/np.ndarray [H,W,3] or [H,W]

# plot_topk_montage(feat, k=16, score="mean_abs", suptitle="Block3 features")
# plot_energy_overlay(feat, image=img, method="rms", title="Block3 energy overlay")

# -----------------------------------------------------------------------------
# Main prediction flow
# -----------------------------------------------------------------------------
def run(args):
    # check if output_dir exists
    # if not, makedirs
    os.makedirs(args.output_dir + 'full_frame_result/txt', exist_ok=True)
    os.makedirs(args.output_dir + 'full_frame_result/vis', exist_ok=True)
    
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
    # find img_name in img_paths
    img_path = img_paths[args.img_name]

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

    os.makedirs(tp_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)

    #### prepare RT params
    thetas_deg, thetas, cos_t, sin_t, rhos = _make_params_dh(max_rho/2, theta_res_deg=itheta, rho_res=irho)
    
    # make it run on a given image
    img = np.load(img_path)
    # allow user to specify the crop window too
    img_crop = img[args.crop_ymin:args.crop_ymin+args.tile_size, args.crop_xmin:args.crop_xmin+args.tile_size]

    # run rt
    rt_map, theta_deg, rhos = hough_bruteforce_intensity_numba_dh(img_crop, max_rho/2, theta_res_deg=itheta, rho_res=irho)
    
    rt_map = rt_map - rt_map.min()
    rt_map = rt_map / rt_map.max() * 255
    rt_map = rt_map.astype(np.uint8)
    # pad the rt_map to be divisible by 32
    pad_h = (32 - rt_map.shape[0] % 32) % 32
    pad_w = (32 - rt_map.shape[1] % 32) % 32
    rt_map = cv2.copyMakeBorder(rt_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    # pad rt_map to 3 channels
    
    rt_map = rt_map[:, :, np.newaxis]
    rt_map = np.repeat(rt_map, 3, axis=2)
    th, tw, _ = rt_map.shape

    # scale img_crop
    scaled_tile = img_crop.copy()
    
    zero_mask = scaled_tile == 0
    scaled_tile = scaled_tile - scaled_tile.min()
    scaled_tile = scaled_tile / scaled_tile.max() * 255.0
    scaled_tile[zero_mask] = 0
    scaled_tile = scaled_tile.astype(np.uint8)
    # stack scaled_tile to make it 3 channels
    scaled_tile = scaled_tile[:, :, np.newaxis]
    scaled_tile = np.repeat(scaled_tile, 3, axis=2)
    # scaled_tile = np.zeros_like(scaled_tile)
    # preprocess them
    res1 = scale_transform(rt_map)
    res2 = scale_transform(image=scaled_tile)
    img1_scaled = res1["image"]
    img2_scaled = res2["image"]
    # img2_scaled = np.zeros_like(img2_scaled)  # ablation: zero out the image input to see only RT effect
    
    # 2. Normalize & ToTensor (no extra padding)
    final1 = norm_transform1(image=img1_scaled)["image"]
    final2 = norm_transform2(image=img2_scaled)["image"]
    
    images_aug = []
    images2_aug = []
    # pass it through the model
    images_aug.append(final1)
    images2_aug.append(final2)

    batch_tensor = torch.stack(images_aug).to(device=device, dtype=torch.float32)
    batch_tensor2 = torch.stack(images2_aug).to(device=device, dtype=torch.float32)


    # --- Hook registration to extract intermediate layers ---
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            # store output (detach is usually good practice if not training)
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
            else:
                activations[name] = output # handle tuples if necessary
        return hook

    # Register hooks on likely BiSeNet components.
    target_layers = ['detail', 'segment', 'bga', 'detail_branch', 'semantic_branch', 'bga_layer',
                     'detail_branch1', 'semantic_branch1', 'bga_layer1', 'seg_head1',
                     'detail_branch2', 'semantic_branch2', 'bga_layer2',
                    'seg_head2', 'dht_layer']
    
    for name, module in model.named_modules():
        print(name)
        if name in target_layers:
            print(f"Registered hook for layer: {name}")
            module.register_forward_hook(get_activation(name))

    # extract the wanted layers
    with torch.no_grad():
        if is_dual:
            logits = model(batch_tensor, batch_tensor2)
            logits_main, logits_s2 = logits[0], logits[1]

            # Keep "raw" as raw logits for visualization.
            preds_rt_raw = logits_main.squeeze(1)
            preds_rt = (logits_main.sigmoid() > 0.5).squeeze(1).long()
            preds_streak_raw = logits_s2.squeeze(1) if logits_s2 is not None else None
            preds_streak = (logits_s2.sigmoid() > 0.5).squeeze(1).long() if logits_s2 is not None else None
    
    preds_rt_raw = preds_rt_raw.detach().cpu().numpy()
    preds_rt = preds_rt.detach().cpu().numpy()
    if preds_streak is None or preds_streak_raw is None:
        raise ValueError("Expected streak predictions (logits_s2) but got None")
    preds_streak = preds_streak.detach().cpu().numpy()
    preds_streak_raw = preds_streak_raw.detach().cpu().numpy()

    def _extract_first_tensor(act):
        if act is None:
            return None
        if isinstance(act, torch.Tensor):
            if act.ndim >= 1 and act.shape[0] > 0:
                return act[0].detach().float().cpu()
            return None
        if isinstance(act, (list, tuple)):
            if len(act) == 0:
                return None
            first = act[0]
            if isinstance(first, torch.Tensor):
                return first.detach().float().cpu()
        return None

    def _norm01_np(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        if amax <= amin:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - amin) / (amax - amin)

    def _prep_bg(img: ArrayLike) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    def _save_overlay_png(
        arr: np.ndarray,
        bg_img: ArrayLike,
        out_path: str,
        cmap: str = "jet",
        alpha: float = 0.45,
        is_binary: bool = False,
    ):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        bg = _prep_bg(bg_img)
        overlay = np.squeeze(arr)
        if is_binary:
            overlay = overlay.astype(np.float32)
        else:
            overlay = _norm01_np(overlay)

        if overlay.ndim > 2:
            overlay = np.squeeze(overlay)
        if overlay.ndim != 2:
            raise ValueError(f"Expected 2D overlay map after squeeze, got shape {overlay.shape}")

        h_bg, w_bg = bg.shape[:2]
        if overlay.shape[0] != h_bg or overlay.shape[1] != w_bg:
            interpolation = cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR
            overlay = cv2.resize(overlay, (w_bg, h_bg), interpolation=interpolation)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(bg, interpolation="nearest")
        if is_binary:
            ax.imshow(overlay, cmap="Reds", alpha=alpha, interpolation="nearest", vmin=0, vmax=1)
        else:
            ax.imshow(overlay, cmap=cmap, alpha=alpha, interpolation="nearest")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    save_dir = os.path.join(args.output_dir, "logits_and_means")
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{args.img_name}_x{args.crop_xmin}_y{args.crop_ymin}"

    _save_overlay_png(
        preds_rt_raw,
        rt_map,
        os.path.join(save_dir, f"{prefix}_preds_rt_raw.png"),
        cmap="jet",
        alpha=0.45,
        is_binary=False,
    )
    _save_overlay_png(
        preds_rt,
        rt_map,
        os.path.join(save_dir, f"{prefix}_preds_rt.png"),
        alpha=0.45,
        is_binary=True,
    )
    _save_overlay_png(
        preds_streak_raw,
        img2_scaled,
        os.path.join(save_dir, f"{prefix}_preds_streak_raw.png"),
        cmap="jet",
        alpha=0.45,
        is_binary=False,
    )
    _save_overlay_png(
        preds_streak,
        img2_scaled,
        os.path.join(save_dir, f"{prefix}_preds_streak.png"),
        alpha=0.45,
        is_binary=True,
    )

    seg_head1_0 = _extract_first_tensor(activations.get("seg_head1"))
    if seg_head1_0 is not None and seg_head1_0.ndim >= 3:
        _save_overlay_png(
            seg_head1_0.mean(dim=0).numpy(),
            rt_map,
            os.path.join(save_dir, f"{prefix}_seg_head1_0_mean.png"),
            cmap="jet",
            alpha=0.45,
        )

    dht_layer_0 = _extract_first_tensor(activations.get("dht_layer"))
    if dht_layer_0 is not None:
        dht_mean = dht_layer_0.mean(dim=0).numpy() if dht_layer_0.ndim >= 3 else dht_layer_0.numpy()
        _save_overlay_png(
            dht_mean,
            rt_map,
            os.path.join(save_dir, f"{prefix}_dht_layer_0_mean.png"),
            cmap="jet",
            alpha=0.45,
        )

    seg_head2_0 = _extract_first_tensor(activations.get("seg_head2"))
    if seg_head2_0 is not None and seg_head2_0.ndim >= 3:
        _save_overlay_png(
            seg_head2_0.mean(dim=0).numpy(),
            img2_scaled,
            os.path.join(save_dir, f"{prefix}_seg_head2_0_mean.png"),
            cmap="jet",
            alpha=0.45,
        )

    print(f"Saved PNG logits/means in: {save_dir}")
    
    # plt.imshow(np.squeeze(preds_rt))
    # plt.show()
    # print('ck')
    # plt.close('all')
    # # show preds_streak_raw and preds_streak side by side
    # plt.subplot(1,2,1)
    # plt.imshow(np.squeeze(preds_streak))
    # plt.title('Binarized Streak Prediction')
    # plt.subplot(1,2,2)  
    # plt.imshow(np.squeeze(preds_streak_raw))
    # plt.title('Raw Streak Prediction')
    # plt.show()
    # print('ck')
    # plt.close('all')
    
    # --- Print extracted info to verify ---
    print("\n--- Extracted Layers ---")
    for name, data in activations.items():
        if isinstance(data, torch.Tensor):
            print(f"Layer '{name}' output shape: {data.shape}")
        elif isinstance(data, (list, tuple)):
             print(f"Layer '{name}' output is a sequence/tuple of length {len(data)}")
    
    int_layer = 'detail_branch2'
    int_layer = 'semantic_branch2'
    # int_layer = 'bga_layer2'
    feature_map = activations[int_layer]
      # shape [B,C,H,W]
    # plot_topk_montage(feature_map, k=16, score="mean", suptitle="detail_branch2")
    
    # TODO: plot all the energy overlays side by side and save them
    energy_dir = os.path.join(args.output_dir, "energy_overlays")
    os.makedirs(energy_dir, exist_ok=True)

    # preds_out_path = os.path.join(
    #     energy_dir,
    #     f"{args.img_name}_x{args.crop_xmin}_y{args.crop_ymin}_preds.png",
    # )
    # preds_saved = save_predictions_visualisation(
    #     preds_rt_raw=preds_rt_raw,
    #     preds_rt=preds_rt,
    #     preds_streak_raw=preds_streak_raw,
    #     preds_streak=preds_streak,
    #     rt_map=rt_map,
    #     scaled_tile=scaled_tile,
    #     out_path=preds_out_path,
    #     show=True,
    # )
    # print(f"Saved preds visualisation: {preds_saved}")

    group1_out_path = os.path.join(
        energy_dir,
        f"{args.img_name}_x{args.crop_xmin}_y{args.crop_ymin}_group1.png",
    )
    group2_out_path = os.path.join(
        energy_dir,
        f"{args.img_name}_x{args.crop_xmin}_y{args.crop_ymin}_group2.png",
    )
    g1_saved, g2_saved = save_grouped_features_and_predictions_grids(
        activations=activations,
        preds_rt_raw=preds_rt_raw,
        preds_rt=preds_rt,
        preds_streak_raw=preds_streak_raw,
        preds_streak=preds_streak,
        rt_map=rt_map,
        scaled_tile=scaled_tile,
        out_path_group1=group1_out_path,
        out_path_group2=group2_out_path,
        method="mean",
        show=True,
    )
    print(f"Saved group1 grid: {g1_saved}")
    print(f"Saved group2 grid: {g2_saved}")

    

    

    

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

    run(args)
