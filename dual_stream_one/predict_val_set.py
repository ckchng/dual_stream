#!/usr/bin/env python
"""
Validation-set prediction script.

Runs inference over all tiles in a pre-computed dataset split (default: val),
overlays both pred_rt and pred_streak on their respective input images, and
annotates each panel with the IoU against the GT mask.

Expected dataset layout:
    data_root/
        raw/        <split>/  *.png    (spatial tiles, uint8 grayscale 288×288)
        raw_labels/ <split>/  *.png    (spatial GT masks, uint8 binary 0/1)
        rt/         <split>/  *.png    (RT maps, uint8 grayscale 192×W)
        rt_labels/  <split>/  *.png    (RT GT masks, uint8 binary 0/1)

Filename convention: {img_stem}_tile_{x}_{y}.png

Usage (CLI):
    python predict_val_set.py --ckpt /path/to/best.pth --data-root /path/to/dataset

Usage (hardcoded):
    Set USE_ARGPARSE = False and edit HARD_CONFIG below.
"""
import argparse
import os
from typing import List, Tuple

import albumentations as AT
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from models import get_model
from utils import transforms

# -----------------------------------------------------------------------------
# Toggle: set False to use HARD_CONFIG dict instead of CLI flags
# -----------------------------------------------------------------------------
USE_ARGPARSE = False

HARD_CONFIG = dict(
    # model="bisenetv2dualmaskguidedv2",
    model="bisenetv2",
    encoder=None,
    decoder=None,
    num_classes=1,
    use_aux=True,
    ckpt="/home/ckchng/Documents/dual_stream/dual_stream_one/save/snr_1_32_len_200_for_m1/single_class/rt_run5/best.pth",
    data_root="/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_32_len_200_for_m1",
    split="val",
    output_dir="./rt_val_set_vis",
    device="auto",
    batch_size=64,
    scale=1.0,
    mean=(0.39509313, 0.39509313, 0.39509313),
    std=(0.17064099, 0.17064099, 0.17064099),
    mean2=(0.34827731, 0.34827731, 0.34827731),
    std2=(0.16927711, 0.16927711, 0.16927711),
    overlay_alpha=0.45,
    # Only save tiles where at least one GT mask has a positive pixel
    positive_only=False,
)

# Overlay colour per class (BGR)
CLASS_COLORS_BGR = {
    1: (0, 255, 0),    # foreground – green
}


# -----------------------------------------------------------------------------
class PredictConfig:
    def __init__(self, model, num_class, encoder=None, decoder=None,
                 use_aux=False, use_detail_head=False, colormap="cityscapes"):
        self.model = model
        self.num_class = num_class
        self.encoder = encoder
        self.decoder = decoder
        self.use_aux = use_aux
        self.use_detail_head = use_detail_head
        self.colormap = colormap


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)


def compute_iou(pred: np.ndarray, gt: np.ndarray, fg_class: int = 1) -> float:
    """Binary IoU for fg_class. Returns NaN when there are no positives in either mask."""
    tp = int(((pred == fg_class) & (gt == fg_class)).sum())
    fp = int(((pred == fg_class) & (gt != fg_class)).sum())
    fn = int(((pred != fg_class) & (gt == fg_class)).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


def build_overlay(tile_bgr: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a colour layer for each foreground class onto tile_bgr."""
    colour_layer = tile_bgr.copy().astype(np.float32)
    for cls_id, color in CLASS_COLORS_BGR.items():
        colour_layer[pred_mask == cls_id] = color
    overlay = alpha * colour_layer + (1.0 - alpha) * tile_bgr.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def annotate_iou(img: np.ndarray, iou: float, label: str = "") -> np.ndarray:
    """Draw an IoU annotation string on the top-left of an image copy."""
    out = img.copy()
    iou_str = f"{label}IoU={'N/A' if np.isnan(iou) else f'{iou:.3f}'}"
    cv2.putText(out, iou_str, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def normalize_tile_to_uint8(raw: np.ndarray) -> np.ndarray:
    """Min-max scale a grayscale tile to uint8, preserving zero pixels as zero."""
    raw = raw.astype(np.float32)
    zero_mask = raw == 0
    raw -= raw.min()
    mx = raw.max()
    if mx > 0:
        raw = raw / mx * 255.0
    raw[zero_mask] = 0
    return raw.astype(np.uint8)


def pad_to_32(img: np.ndarray) -> np.ndarray:
    """Zero-pad H and W to be multiples of 32."""
    h, w = img.shape[:2]
    ph = (32 - h % 32) % 32
    pw = (32 - w % 32) % 32
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    return img


def make_3ch(img: np.ndarray) -> np.ndarray:
    """Ensure image is (H, W, 3)."""
    if img.ndim == 2:
        return np.stack((img,) * 3, axis=-1)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def discover_files(data_root: str, split: str, is_dual_input: bool) -> List[str]:
    """Return sorted list of basenames. Uses rt/<split>/ for single-input models, raw/<split>/ for dual."""
    subdir = "raw" if is_dual_input else "rt"
    scan_dir = os.path.join(data_root, subdir, split)
    return sorted(f for f in os.listdir(scan_dir) if f.endswith(".png"))


def load_tile_data(data_root: str, split: str, fname: str, is_dual_input: bool):
    """
    Load one tile's inputs and GT masks.

    Returns dict with:
        raw_img   – uint8 grayscale (H, W)       [None for single-input models]
        raw_lbl   – uint8 binary (H, W)           [None for single-input models]
        rt_img    – uint8 grayscale (H_rt, W_rt)
        rt_lbl    – uint8 binary (H_rt, W_rt)
    """
    def _imread_gray(path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    base = data_root
    d = dict(
        raw_img=None,
        raw_lbl=None,
        rt_img= _imread_gray(os.path.join(base, "rt",        split, fname)),
        rt_lbl= _imread_gray(os.path.join(base, "rt_labels", split, fname)),
    )
    if is_dual_input:
        d["raw_img"] = _imread_gray(os.path.join(base, "raw",        split, fname))
        d["raw_lbl"] = _imread_gray(os.path.join(base, "raw_labels", split, fname))
    return d


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------

def build_batch_tensors(
    items: List[dict],
    scale_transform,
    norm_rt, norm_spatial,
    device: torch.device,
):
    """
    Convert a list of tile dicts into two stacked tensors (t1, t2).

    Each dict must have keys: rt_3ch, spatial_3ch
    """
    t1s, t2s = [], []
    for it in items:
        img1 = scale_transform(image=it["rt_3ch"])["image"]
        img2 = scale_transform(image=it["spatial_3ch"])["image"]
        t1s.append(norm_rt(image=img1)["image"])
        t2s.append(norm_spatial(image=img2)["image"])
    t1 = torch.stack(t1s).to(device, dtype=torch.float32)
    t2 = torch.stack(t2s).to(device, dtype=torch.float32)
    return t1, t2


def run_batch(model, t1, t2, num_classes: int, is_maskguided: bool, is_dual_input: bool = True):
    """Run model on (t1[, t2]), return (pred_rt_batch, pred_streak_batch) as numpy arrays (B, H, W)."""
    with torch.no_grad():
        logits = model(t1, t2) if is_dual_input else model(t1)

        if isinstance(logits, (tuple, list)):
            logits_main = logits[0]
            logits_s2 = logits[1] if (is_maskguided and len(logits) >= 2) else None
        else:
            logits_main = logits
            logits_s2 = None

        if num_classes == 1:
            pred_rt = (logits_main.sigmoid() > 0.5).squeeze(1).long().cpu().numpy()
        else:
            pred_rt = logits_main.argmax(dim=1).cpu().numpy()

        if logits_s2 is None:
            pred_streak = np.zeros_like(pred_rt)
        else:
            if num_classes == 1:
                pred_streak = (logits_s2.sigmoid() > 0.5).squeeze(1).long().cpu().numpy()
            else:
                pred_streak = logits_s2.argmax(dim=1).cpu().numpy()

    return pred_rt, pred_streak


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------

def make_composite(
    spatial_bgr: np.ndarray,   # (H, W, 3)  spatial tile (uint8)
    pred_streak: np.ndarray,   # (H, W)     predicted spatial mask
    raw_lbl: np.ndarray,       # (H, W)     GT spatial mask
    rt_bgr: np.ndarray,        # (H_rt, W_rt, 3) RT map (uint8)
    pred_rt: np.ndarray,       # (H_rt, W_rt)  predicted RT mask
    rt_lbl: np.ndarray,        # (H_rt, W_rt)  GT RT mask
    alpha: float,
) -> np.ndarray:
    """
    Build a 2-panel composite:
        [Spatial tile + streak pred | RT map + RT pred]
    Each panel is annotated with its IoU.
    """
    iou_streak = compute_iou(pred_streak, raw_lbl)
    iou_rt     = compute_iou(pred_rt,     rt_lbl)

    # --- panel 1: spatial ---
    panel_spatial = build_overlay(spatial_bgr, pred_streak, alpha)
    panel_spatial = annotate_iou(panel_spatial, iou_streak, label="Streak ")
    # Draw GT contour in yellow
    gt_contours, _ = cv2.findContours(raw_lbl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_spatial, gt_contours, -1, (0, 255, 255), 1)

    # --- panel 2: RT ---
    panel_rt = build_overlay(rt_bgr, pred_rt, alpha)
    panel_rt = annotate_iou(panel_rt, iou_rt, label="RT ")
    gt_rt_contours, _ = cv2.findContours(rt_lbl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_rt, gt_rt_contours, -1, (0, 255, 255), 1)

    # Pad both panels to the same height before horizontal concat
    h1, h2 = panel_spatial.shape[0], panel_rt.shape[0]
    target_h = max(h1, h2)
    if h1 < target_h:
        panel_spatial = cv2.copyMakeBorder(panel_spatial, 0, target_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if h2 < target_h:
        panel_rt = cv2.copyMakeBorder(panel_rt, 0, target_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=0)

    composite = cv2.hconcat([panel_spatial, panel_rt])

    # Column header labels
    for col_j, lbl in enumerate(["Spatial (streak pred)", "RT map (rt pred)"]):
        x_off = col_j * panel_spatial.shape[1]
        cv2.putText(composite, lbl, (x_off + 4, target_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return composite, iou_streak, iou_rt


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # --- Model ---
    config = PredictConfig(
        model=args.model,
        num_class=args.num_classes,
        encoder=args.encoder,
        decoder=args.decoder,
        use_aux=args.use_aux,
    )
    model = get_model(config).to(device)
    model.eval()
    load_checkpoint(model, args.ckpt, device)
    print(f"Loaded checkpoint: {args.ckpt}")

    is_maskguided = "maskguided" in args.model.lower()
    is_dual_input = "dual" in args.model.lower()

    scale_transform = transforms.Scale(scale=args.scale, is_testing=True)
    norm_rt      = AT.Compose([AT.Normalize(mean=args.mean,  std=args.std),  ToTensorV2()])
    norm_spatial = AT.Compose([AT.Normalize(mean=args.mean2, std=args.std2), ToTensorV2()])

    # --- Discover files ---
    all_files = discover_files(args.data_root, args.split, is_dual_input)
    print(f"Found {len(all_files)} tiles in {args.data_root}/{args.split}")

    # --- Optional filter: positive-only ---
    if args.positive_only:
        print("Filtering to tiles with at least one positive GT pixel...")
        positive_files = []
        for fname in all_files:
            d = load_tile_data(args.data_root, args.split, fname, is_dual_input)
            has_pos = d["rt_lbl"].max() > 0
            if is_dual_input:
                has_pos = has_pos or d["raw_lbl"].max() > 0
            if has_pos:
                positive_files.append(fname)
        all_files = positive_files
        print(f"  → {len(all_files)} tiles after filtering")

    # --- Batch inference loop ---
    total = len(all_files)
    batch_size = args.batch_size
    iou_streak_list, iou_rt_list = [], []

    for batch_start in range(0, total, batch_size):
        batch_fnames = all_files[batch_start: batch_start + batch_size]
        batch_data = [load_tile_data(args.data_root, args.split, f, is_dual_input) for f in batch_fnames]

        # Prepare tensors for model
        items = []
        for d in batch_data:
            rt_3ch = make_3ch(pad_to_32(d["rt_img"]))
            if is_dual_input:
                spatial_3ch = make_3ch(normalize_tile_to_uint8(d["raw_img"]))
            else:
                spatial_3ch = rt_3ch  # unused by model; placeholder to keep build_batch_tensors happy
            items.append({"rt_3ch": rt_3ch, "spatial_3ch": spatial_3ch})

        t1, t2 = build_batch_tensors(items, scale_transform, norm_rt, norm_spatial, device)
        pred_rt_batch, pred_streak_batch = run_batch(model, t1, t2, args.num_classes, is_maskguided, is_dual_input)

        # Post-process and save each tile
        for i, (fname, d) in enumerate(zip(batch_fnames, batch_data)):
            pred_rt = pred_rt_batch[i]
            rh, rw = d["rt_lbl"].shape[:2]
            pred_rt = pred_rt[:rh, :rw]
            rt_bgr  = make_3ch(pad_to_32(d["rt_img"]))[:rh, :rw]

            if is_dual_input:
                pred_streak = pred_streak_batch[i]
                sh, sw = d["raw_lbl"].shape[:2]
                pred_streak = pred_streak[:sh, :sw]
                spatial_bgr = cv2.cvtColor(make_3ch(normalize_tile_to_uint8(d["raw_img"])), cv2.COLOR_RGB2BGR)
                composite, iou_s, iou_r = make_composite(
                    spatial_bgr, pred_streak, d["raw_lbl"],
                    rt_bgr,      pred_rt,     d["rt_lbl"],
                    alpha=args.overlay_alpha,
                )
                iou_streak_list.append(iou_s)
            else:
                # Single-input: only RT panel
                iou_r = compute_iou(pred_rt, d["rt_lbl"])
                panel_rt = build_overlay(rt_bgr, pred_rt, args.overlay_alpha)
                panel_rt = annotate_iou(panel_rt, iou_r, label="RT ")
                gt_rt_contours, _ = cv2.findContours(
                    d["rt_lbl"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel_rt, gt_rt_contours, -1, (0, 255, 255), 1)
                composite = panel_rt
                iou_s = float("nan")

            iou_rt_list.append(iou_r)

            has_pred = pred_rt.max() > 0 or (is_dual_input and pred_streak_batch[i][:rh, :rw].max() > 0)
            if has_pred:
                out_fname = os.path.splitext(fname)[0] + "_pred.png"
                cv2.imwrite(os.path.join(args.output_dir, out_fname), composite)

        done = min(batch_start + batch_size, total)
        rt_mean = np.nanmean([v for v in iou_rt_list[-len(batch_fnames):] if not np.isnan(v)] or [float('nan')])
        if is_dual_input:
            s_mean = np.nanmean([v for v in iou_streak_list[-len(batch_fnames):] if not np.isnan(v)] or [float('nan')])
            print(f"  [{done:5d}/{total}]  last batch IoU — streak: {s_mean:.3f}  rt: {rt_mean:.3f}")
        else:
            print(f"  [{done:5d}/{total}]  last batch IoU — rt: {rt_mean:.3f}")

    # --- Summary ---
    valid_r = [v for v in iou_rt_list if not np.isnan(v)]
    print("\n=== Summary ===")
    print(f"Tiles processed : {total}")
    if is_dual_input:
        valid_s = [v for v in iou_streak_list if not np.isnan(v)]
        print(f"Streak IoU      : mean={np.mean(valid_s):.4f}  median={np.median(valid_s):.4f}  (n={len(valid_s)})")
    print(f"RT IoU          : mean={np.mean(valid_r):.4f}  median={np.median(valid_r):.4f}  (n={len(valid_r)})")
    print(f"Output saved to : {args.output_dir}")


# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Validation-set prediction with overlay and IoU annotation.")
    p.add_argument("--model",       type=str, default="bisenetv2dualmaskguidedv2")
    p.add_argument("--encoder",     type=str, default=None)
    p.add_argument("--decoder",     type=str, default=None)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--use-aux",     action="store_true")
    p.add_argument("--ckpt",        type=str, required=True)
    p.add_argument("--data-root",   type=str, required=True,
                   help="Root of the dataset (parent of raw/, rt/, raw_labels/, rt_labels/).")
    p.add_argument("--split",       type=str, default="val",
                   help="Dataset split subdirectory (default: val).")
    p.add_argument("--output-dir",  type=str, default="./val_set_vis")
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--scale",       type=float, default=1.0)
    p.add_argument("--mean",  type=float, nargs=3, default=(0.39509313, 0.39509313, 0.39509313))
    p.add_argument("--std",   type=float, nargs=3, default=(0.17064099, 0.17064099, 0.17064099))
    p.add_argument("--mean2", type=float, nargs=3, default=(0.34827731, 0.34827731, 0.34827731))
    p.add_argument("--std2",  type=float, nargs=3, default=(0.16927711, 0.16927711, 0.16927711))
    p.add_argument("--overlay-alpha", type=float, default=0.45)
    p.add_argument("--positive-only", action="store_true",
                   help="Only save tiles where at least one GT mask has a positive pixel.")
    return p.parse_args()


if __name__ == "__main__":
    if USE_ARGPARSE:
        args = parse_args()
    else:
        args = argparse.Namespace(**HARD_CONFIG)
    run(args)
