#!/usr/bin/env python
"""
Single-tile prediction script.

Loads a pretrained segmentation model and runs inference on ONE tile extracted
from the image at the given (x, y) pixel origin. The predicted mask is overlaid
on the raw tile and saved / displayed.

Usage (CLI):
    python predict_single_tile.py --img /path/to/image.npy --x 576 --y 288

Usage (hardcoded):
    Set USE_ARGPARSE = False and edit HARD_CONFIG below.
"""
import argparse
import os

import albumentations as AT
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

from models import get_model
from utils.bs_detector_sep import detect_roundish, remove_masked_with_zero
from utils import transforms
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'data_generation'))
from ht_utils import _make_params, compute_rt_map

# -----------------------------------------------------------------------------
# Toggle: set False to use HARD_CONFIG dict instead of CLI flags
# -----------------------------------------------------------------------------
USE_ARGPARSE = False

HARD_CONFIG = dict(
    model="bisenetv2dualmaskguidedv2",
    encoder=None,
    decoder=None,
    num_classes=1,
    use_aux=True,
    ckpt="/home/ckchng/Documents/dual_stream_two/save/bg_50_no_crop/snr_1_32_len_200/single_class/both_run1/best.pth",
    # ---- image name (filename only) and root search directory ----
    # img_name="000_2020-12-08_101438_E_DSC_0352.npy",
    # img_name="000_2020-12-08_102218_E_DSC_0398.npy",
    img_name='000_2020-12-08_104028_E_DSC_0507.npy',
    img_dir="/media/ckchng/internal2TB/FILTERED_IMAGES/",
    # ---- tile origin (top-left pixel of the tile) ----
    # x=1584,
    # y=1008,
    # x=288,
    # y=3024,
    x=5328,
    y=2448,
    output_dir="./single_tile_vis",
    device="auto",
    scale=1.0,
    mean=(0.39509313, 0.39509313, 0.39509313),
    std=(0.17064099, 0.17064099, 0.17064099),
    mean2=(0.34827731, 0.34827731, 0.34827731),
    std2=(0.16927711, 0.16927711, 0.16927711),
    tile_size=288,
    sep=True,
    sep_params=[3.0, 6, 6.0, 0.6, 6.0, 0.1],
    num_angles=192,
    num_rhos=288,
    rho_min_cap=-144,
    rho_max_cap=143,
    # overlay options
    overlay_alpha=0.45,   # blend weight for the mask colour
    show=True,            # pop up a matplotlib window
)

# Overlay colour per class index  (BGR for cv2, RGB for display)
CLASS_COLORS_BGR = {
    0: None,           # background – transparent
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


def normalize_to_uint8(arr, zero_mask=None):
    """Min-max scale to [0, 255] uint8; keep zeros from zero_mask as zero."""
    out = arr.astype(float)
    out = out - out.min()
    mx = out.max()
    if mx > 0:
        out = out / mx * 255.0
    if zero_mask is not None:
        out[zero_mask] = 0
    return out.astype(np.uint8)


def process_single_tile(image_rgb, x, y, tile_size, sep, sep_params,
                        max_rho, itheta, irho, rho_min_cap, rho_max_cap):
    """
    Extract the tile at (x, y) from image_rgb and compute all streams.
    Returns a dict with keys: ori_img, scaled_img, rt_map, x, y.
    """
    h, w, _ = image_rgb.shape

    # Clamp so tile stays inside image
    x = min(max(x, 0), max(w - tile_size, 0))
    y = min(max(y, 0), max(h - tile_size, 0))

    tile = image_rgb[y: y + tile_size, x: x + tile_size]

    # --- SEP (source extraction / roundish removal) ---
    if sep:
        _, round_mask, _, _, _ = detect_roundish(
            tile[:, :, 0],
            sep_params[0], sep_params[1],
            sep_params[2], sep_params[3],
            sep_params[4], deblend_cont=sep_params[5],
        )
        processed = remove_masked_with_zero(tile[:, :, 0], round_mask)
    else:
        processed = tile[:, :, 0]

    processed = np.stack((processed,) * 3, axis=-1)

    # --- scaled image stream ---
    zero_mask = processed == 0
    scaled = normalize_to_uint8(processed, zero_mask)

    # --- RT map stream: use the unmasked tile so corner pixels contribute ---
    rt_map, _, _ = compute_rt_map(
        processed[:, :, 0], max_rho / 2, itheta, irho,
        rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap,
    )
    rt_map = rt_map - rt_map.min()
    mx = rt_map.max()
    if mx > 0:
        rt_map = rt_map / mx * 255
    rt_map = rt_map.astype(np.uint8)

    # Pad RT map to multiple of 32
    pad_h = (32 - rt_map.shape[0] % 32) % 32
    pad_w = (32 - rt_map.shape[1] % 32) % 32
    rt_map = cv2.copyMakeBorder(rt_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    rt_map = np.repeat(rt_map[:, :, np.newaxis], 3, axis=2)

    return {"ori_img": processed, "scaled_img": scaled, "rt_map": rt_map, "x": x, "y": y}


def run_inference(tile_dict, model, device, is_dual, is_maskguided,
                  scale_transform, norm_transform1, norm_transform2,
                  transform, num_classes):
    """Run the model on a single tile dict. Returns (pred_rt, pred_streak) as numpy arrays."""
    if is_dual:
        img1 = scale_transform(image=tile_dict["rt_map"])["image"]
        img2 = scale_transform(image=tile_dict["scaled_img"])["image"]
        t1 = norm_transform1(image=img1)["image"].unsqueeze(0).to(device, dtype=torch.float32)
        t2 = norm_transform2(image=img2)["image"].unsqueeze(0).to(device, dtype=torch.float32)
    else:
        aug = transform(image=tile_dict["rt_map"])
        t1 = aug["image"].unsqueeze(0).to(device, dtype=torch.float32)
        t2 = None

    with torch.no_grad():
        logits = model(t1, t2) if is_dual else model(t1)

        logits_main = logits
        logits_s2 = None
        if isinstance(logits, (tuple, list)):
            if is_maskguided and len(logits) >= 2:
                logits_main, logits_s2 = logits[0], logits[1]
            else:
                logits_main = logits[0]

        if num_classes == 1:
            pred_rt = (logits_main.sigmoid() > 0.5).squeeze().long().cpu().numpy()
        else:
            pred_rt = logits_main.argmax(dim=1).squeeze(0).cpu().numpy()

        if logits_s2 is None:
            pred_streak = np.zeros_like(pred_rt)
        else:
            if num_classes == 1:
                pred_streak = (logits_s2.sigmoid() > 0.5).squeeze().long().cpu().numpy()
            else:
                pred_streak = logits_s2.argmax(dim=1).squeeze(0).cpu().numpy()

    return pred_rt, pred_streak


def apply_rho_coverage_mask(tile_bgr, tile_size, rho_min_cap, rho_max_cap, dim_factor=0.0):
    """
    Dim pixels in tile_bgr whose distance from the tile centre exceeds the
    effective rho half-range.  Those pixels can produce rho values outside
    [rho_min_cap, rho_max_cap] for some angles and are therefore not fully
    covered by the RT transform.

    dim_factor: brightness multiplier for out-of-range pixels (0 = black, 1 = unchanged).
    """
    if rho_min_cap is not None and rho_max_cap is not None:
        effective_half = (rho_max_cap - rho_min_cap) / 2.0
    elif rho_max_cap is not None:
        effective_half = float(rho_max_cap)
    elif rho_min_cap is not None:
        effective_half = abs(float(rho_min_cap))
    else:
        return tile_bgr.copy()  # no caps — nothing to mask

    cx = tile_size / 2.0
    cy = tile_size / 2.0

    ys, xs = np.ogrid[:tile_size, :tile_size]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)   # (H, W)

    outside = dist > effective_half  # bool mask

    out = tile_bgr.astype(np.float32).copy()
    out[outside] *= dim_factor
    return np.clip(out, 0, 255).astype(np.uint8)


def featuremap_to_heatmap_bgr(feat_tensor, target_h, target_w, alpha=0.45, bg_bgr=None):
    """
    Convert a [C, H, W] feature tensor to a jet-colourmap BGR image blended
    onto bg_bgr.  If bg_bgr is None, returns the plain heatmap.
    """
    if feat_tensor.ndim == 3:
        energy = feat_tensor.float().mean(dim=0).numpy()   # [H, W]
    else:
        energy = feat_tensor.float().numpy()

    e_min, e_max = float(energy.min()), float(energy.max())
    if e_max > e_min:
        energy = (energy - e_min) / (e_max - e_min) * 255.0
    else:
        energy = np.zeros_like(energy)
    energy = energy.astype(np.uint8)

    heatmap = cv2.applyColorMap(energy, cv2.COLORMAP_JET)

    if heatmap.shape[0] != target_h or heatmap.shape[1] != target_w:
        heatmap = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if bg_bgr is not None:
        bg = bg_bgr[:target_h, :target_w]
        heatmap = cv2.addWeighted(bg, 1.0 - alpha, heatmap, alpha, 0)

    return heatmap


def build_overlay(tile_bgr, pred_mask, alpha=0.45):
    """
    Overlay pred_mask on tile_bgr.
    Each foreground class is filled with its CLASS_COLORS_BGR colour,
    blended with the original tile at weight alpha.
    Returns a uint8 BGR image.
    """
    overlay = tile_bgr.copy().astype(np.float32)
    colour_layer = tile_bgr.copy().astype(np.float32)

    for class_id, color in CLASS_COLORS_BGR.items():
        if color is None:
            continue
        mask = (pred_mask == class_id)
        colour_layer[mask] = color  # fill that class with solid colour

    # blend: alpha * colour + (1-alpha) * original
    overlay = alpha * colour_layer + (1.0 - alpha) * tile_bgr.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

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

    is_dual = "dual" in args.model.lower()
    is_maskguided = "maskguided" in args.model.lower()

    scale_transform = transforms.Scale(scale=args.scale, is_testing=True)
    if is_dual:
        norm_transform1 = AT.Compose([AT.Normalize(mean=args.mean, std=args.std), ToTensorV2()])
        norm_transform2 = AT.Compose([AT.Normalize(mean=args.mean2, std=args.std2), ToTensorV2()])
        transform = None
    else:
        norm_transform1 = norm_transform2 = None
        transform = AT.Compose([
            transforms.Scale(scale=args.scale, is_testing=True),
            AT.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])

    # --- Locate image by name under img_dir ---
    target_name = args.img_name
    img_path = None
    print(f"Searching for '{target_name}' under '{args.img_dir}' ...")
    for root, _, files in os.walk(args.img_dir):
        if target_name in files:
            img_path = os.path.join(root, target_name)
            break
    if img_path is None:
        raise FileNotFoundError(
            f"'{target_name}' not found anywhere under '{args.img_dir}'"
        )
    img_name = os.path.splitext(target_name)[0]
    print(f"Found: {img_path}")
    img = np.load(img_path)
    image_rgb = np.stack((img,) * 3, axis=-1)
    print(f"Image shape: {image_rgb.shape}  |  tile origin: x={args.x}, y={args.y}")

    # --- RT / rho params ---
    itheta = 180.0 / args.num_angles
    max_rho = np.hypot(args.tile_size, args.tile_size) + 1
    rho_min_cap = args.rho_min_cap
    rho_max_cap = args.rho_max_cap
    if rho_min_cap is not None and rho_max_cap is not None:
        _half = (rho_max_cap - rho_min_cap) / 2.0
    elif rho_max_cap is not None:
        _half = rho_max_cap
    elif rho_min_cap is not None:
        _half = abs(rho_min_cap)
    else:
        _half = max_rho / 2.0
    irho = (2.0 * _half) / (args.num_rhos - 1)

    # --- Process tile ---
    tile_dict = process_single_tile(
        image_rgb, args.x, args.y, args.tile_size,
        args.sep, args.sep_params,
        max_rho, itheta, irho, rho_min_cap, rho_max_cap,
    )
    print(f"Tile extracted at clamped origin: x={tile_dict['x']}, y={tile_dict['y']}")

    # --- Register forward hook on dht_layer before inference ---
    activations = {}
    _hooks = []
    for name, module in model.named_modules():
        if name == "dht_layer":
            def _make_hook(n):
                def hook(_mod, _inp, out):
                    activations[n] = out.detach() if isinstance(out, torch.Tensor) else out
                return hook
            _hooks.append(module.register_forward_hook(_make_hook(name)))
            print(f"Registered hook on: {name}")
            break

    # --- Inference ---
    pred_rt, _ = run_inference(
        tile_dict, model, device,
        is_dual, is_maskguided,
        scale_transform, norm_transform1, norm_transform2,
        transform, args.num_classes,
    )

    for h in _hooks:
        h.remove()

    # --- Build visualisation panels ---
    tile_bgr = cv2.cvtColor(tile_dict["scaled_img"], cv2.COLOR_RGB2BGR)
    rt_bgr = tile_dict["rt_map"]  # (num_angles, num_rhos, 3) — same space as pred_rt
    rt_h, rt_w = rt_bgr.shape[:2]

    # Overlay pred_rt on the RT map (both are in RT/Hough space)
    overlay_rt = build_overlay(rt_bgr, pred_rt, alpha=args.overlay_alpha)

    # DHT layer energy heatmap blended onto the RT map
    dht_act = activations.get("dht_layer")
    if dht_act is not None:
        feat0 = dht_act[0].detach().float().cpu()   # [C, H, W]
        dht_panel = featuremap_to_heatmap_bgr(feat0, rt_h, rt_w,
                                              alpha=args.overlay_alpha, bg_bgr=rt_bgr)
        print(f"DHT layer feature shape: {dht_act.shape}")
    else:
        dht_panel = rt_bgr.copy()
        print("WARNING: dht_layer hook did not fire — check layer name with model.named_modules()")

    # --- Composite: scaled tile | RT map | RT map + pred overlay | DHT energy on RT map ---
    panel_h = max(tile_bgr.shape[0], rt_h)

    def pad_h(img, target_h):
        if img.shape[0] < target_h:
            img = cv2.copyMakeBorder(img, 0, target_h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
        return img

    panels = [
        (pad_h(tile_bgr,    panel_h), "Scaled tile"),
        (pad_h(rt_bgr,      panel_h), "RT map"),
        (pad_h(overlay_rt,  panel_h), "RT map + pred"),
        (pad_h(dht_panel,   panel_h), "DHT energy"),
    ]

    composite = cv2.hconcat([p for p, _ in panels])

    panel_w = panels[0][0].shape[1]
    for col_j, (_, label) in enumerate(panels):
        cv2.putText(
            composite, label,
            (col_j * panel_w + 4, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA,
        )

    # --- Save ---
    x_str = tile_dict["x"]
    y_str = tile_dict["y"]
    save_name = f"{img_name}_y{y_str}_x{x_str}_single_tile.png"
    save_path = os.path.join(args.output_dir, save_name)
    cv2.imwrite(save_path, composite)
    print(f"Saved composite to: {save_path}")

    # Also save just the DHT heatmap
    dht_save_path = os.path.join(args.output_dir, f"{img_name}_y{y_str}_x{x_str}_dht.png")
    cv2.imwrite(dht_save_path, dht_panel)
    print(f"Saved DHT panel to: {dht_save_path}")

    # --- Display ---
    if args.show:
        composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(18, 5))
        ax.imshow(composite_rgb)
        ax.axis("off")
        ax.set_title(
            f"{img_name}  |  tile x={x_str} y={y_str}  |  "
            f"Scaled tile  |  RT map  |  RT map + pred  |  DHT energy"
        )
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-tile prediction with mask overlay.")
    p.add_argument("--model", type=str, default="bisenetv2dualmaskguidedv2")
    p.add_argument("--encoder", type=str, default=None)
    p.add_argument("--decoder", type=str, default=None)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--use-aux", action="store_true")
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint.")
    p.add_argument("--img-name", type=str, required=True, help="Filename of the .npy image (e.g. foo.npy).")
    p.add_argument("--img-dir", type=str, required=True, help="Root directory to search recursively for --img-name.")
    p.add_argument("--x", type=int, required=True, help="Tile left-edge x pixel coordinate.")
    p.add_argument("--y", type=int, required=True, help="Tile top-edge y pixel coordinate.")
    p.add_argument("--output-dir", type=str, default="./single_tile_vis")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--mean", type=float, nargs=3, default=(0.39509313, 0.39509313, 0.39509313))
    p.add_argument("--std", type=float, nargs=3, default=(0.17064099, 0.17064099, 0.17064099))
    p.add_argument("--mean2", type=float, nargs=3, default=(0.34827731, 0.34827731, 0.34827731))
    p.add_argument("--std2", type=float, nargs=3, default=(0.16927711, 0.16927711, 0.16927711))
    p.add_argument("--tile-size", type=int, default=288)
    p.add_argument("--sep", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--sep-params", type=float, nargs=6,
                   default=[3.0, 6, 6.0, 0.6, 6.0, 0.1])
    p.add_argument("--num-angles", type=int, default=192)
    p.add_argument("--num-rhos", type=int, default=288)
    p.add_argument("--rho-min-cap", type=float, default=-144.0)
    p.add_argument("--rho-max-cap", type=float, default=143.0)
    p.add_argument("--overlay-alpha", type=float, default=0.45,
                   help="Blend weight for the mask colour (0=invisible, 1=opaque).")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="Skip the matplotlib display window.")
    p.set_defaults(show=True)
    return p.parse_args()


if __name__ == "__main__":
    if USE_ARGPARSE:
        args = parse_args()
    else:
        args = argparse.Namespace(**HARD_CONFIG)

    run(args)
