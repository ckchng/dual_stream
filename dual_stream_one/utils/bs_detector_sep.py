import numpy as np
import sep
import argparse
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import cv2
import sys
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as AT
from PIL import Image

from models import get_model


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import transforms
from utils.utils import get_colormap
from utils.HT_utils import _hough_accumulate_intensity_dh, endpoints_to_rho_theta_dh, rho_theta_to_indices_dh, _make_params_dh, hough_bruteforce_intensity_numba_dh


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


DEFAULT_PANEL_W = 416
DEFAULT_TOP_H = 288

# -----------------------------------------------------------------------------
# CONFIG: set USE_ARGPARSE = False to use the hardcoded parameters below.
# Toggle USE_ARGPARSE = True to read from CLI flags again.
# -----------------------------------------------------------------------------
USE_ARGPARSE = False
HARD_CONFIG = dict(
    img_dir="/media/ckchng/internal2TB/FILTERED_IMAGES/",
    input="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/tile_fp/",
    output_dir="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/sep/from_fp",
    ckpt="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/best.pth",
    panel_width=DEFAULT_PANEL_W,
    top_height=DEFAULT_TOP_H,
    unpad=True,
    inner_width=None,
    device="auto",
    model="bisenetv2dualmaskguidedv2",
    num_classes=2,
    encoder=None,
    decoder=None,
    use_aux=True,  
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
    num_thetas=192,
    num_rhos=416   
)



def _build_args() -> argparse.Namespace:
    if not USE_ARGPARSE:
        return argparse.Namespace(**HARD_CONFIG)

    parser = argparse.ArgumentParser(
        description=(
            "Read composite images from predict_dh_convention_dual_bright_spot_check.py "
            "and extract the masked_disp panel (optionally removing padding)."
        )
    )
    parser.add_argument("input", help="Composite image path or directory.")
    parser.add_argument("--output-dir", default=None, help="Directory to save extracted panels.")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_PANEL_W)
    parser.add_argument("--panel-width", type=int, default=DEFAULT_PANEL_W,
                        help="Width of each panel in the composite image.")
    parser.add_argument("--top-height", type=int, default=DEFAULT_TOP_H,
                        help="Height of the top row panels in the composite image.")
    parser.add_argument("--unpad", action="store_true",
                        help="If set, unpad the masked_disp panel to inner_width.") 
    parser.add_argument("--inner-width", type=int, default=None,
                        help="Inner width to unpad the masked_disp panel to, if --unpad is set.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run the model on ('auto', 'cpu', or 'cuda').") 
    parser.add_argument("--model", type=str, required=True, help="Model architecture name.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--encoder", type=str, default=None, help="Encoder name.")
    parser.add_argument("--decoder", type=str, default=None, help="Decoder name.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--use-aux", action="store_true", help="Use auxiliary classifier head.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for input images.")
    parser.add_argument("--mean", type=float, nargs=3, default=(0.485, 0.456, 0.406), help="Mean for normalization.")
    parser.add_argument("--std", type=float, nargs=3, default=(0.229, 0.224, 0.225), help="Std for normalization.")
    parser.add_argument("--mean2", type=float, nargs=3, default=(0.485, 0.456, 0.406), help="Mean for normalization of second stream.")
    parser.add_argument("--std2", type=float, nargs=3, default=(0.229, 0.224, 0.225), help="Std for normalization of second stream.")
    parser.add_argument("--palette", type=str, default="cityscapes", help="Color palette for visualization.")
    parser.add_argument("--blend", action="store_true", help="Blend predictions with input image for visualization.")
    parser.add_argument("--blend-alpha", type=float, default=0.3, help="Alpha value for blending.")
    parser.add_argument("--tile-size", type=int, default=288, help="Tile size for prediction.")
    parser.add_argument("--tile-stride", type=int, default=144, help="Tile stride for prediction.")
    parser.add_argument("--num-thetas", type=int, default=180, help="Number of theta bins for Hough transform.")
    parser.add_argument("--num-rhos", type=int, default=200, help="Number of rho bins for Hough transform.")
    
    return parser.parse_args()

def build_transform(scale: float, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    return AT.Compose(
        [
            transforms.Scale(scale=scale, is_testing=True),
            transforms.PadPairToMax(padding_value=0, mask_value=0),
            AT.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def prepare_model_inputs(rt_map_full, filled_full, *, is_dual, scale_transform,
                         norm_transform1, norm_transform2, transform, device):
    if is_dual:
        img1_scaled = scale_transform(image=rt_map_full)["image"]
        img2_scaled = scale_transform(image=filled_full)["image"]
        final1 = norm_transform1(image=img1_scaled)["image"]
        final2 = norm_transform2(image=img2_scaled)["image"]
        batch_tensor = torch.stack([final1]).to(device=device, dtype=torch.float32)
        batch_tensor2 = torch.stack([final2]).to(device=device, dtype=torch.float32)
        return batch_tensor, batch_tensor2

    if transform is None:
        raise ValueError("transform must be set when is_dual is False.")

    augmented = transform(image=rt_map_full)["image"]
    batch_tensor = torch.stack([augmented]).to(device=device, dtype=torch.float32)
    return batch_tensor, None


def run_model_inference(model, batch_tensor, batch_tensor2, *, is_dual, is_maskguided, num_classes):
    with torch.no_grad():
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

        if num_classes == 2:
            preds_rt = logits_main.argmax(dim=1)
            preds_streak = logits_s2.max(dim=1)[1]
        elif num_classes == 1:
            preds_rt = (logits_main.sigmoid() > 0.5).squeeze(1).long()
            preds_streak = (logits_s2.sigmoid() > 0.5).squeeze(1).long() if logits_s2 is not None else None

    preds_rt = preds_rt.cpu().numpy()
    preds_streak = preds_streak.cpu().numpy()
    return preds_rt, preds_streak


def extract_pred_bboxes_and_lines(preds_rt, rt_map_full, rhos, thetas, tile_size, *, min_size=10):
    preds_rt = preds_rt[0]  # remove batch dim
    if preds_rt.max() == 0:
        return [], []

    # Connected components: labels + statistics
    # stats: [label, x, y, width, height, area]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        preds_rt.astype(np.uint8), connectivity=8
    )

    bboxes = []
    lines = []
    # Label 0 is background, skip it
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area == 0:
            continue
        x1, y1 = x, y
        x2, y2 = x + w - 1, y + h - 1  # inclusive coordinates
        if w < min_size or h < min_size:
            continue
        bboxes.append((x1, y1, x2, y2))

        # find peak intensity location in original image
        crop_rt_map = rt_map_full[y1:y2 + 1, x1:x2 + 1, 0]

        # Find peak intensity location in cropped image
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(crop_rt_map)
        peak_x = np.clip(maxLoc[0] + x1, 0, len(rhos) - 1)
        peak_y = np.clip(maxLoc[1] + y1, 0, len(thetas) - 1)

        pred_rho = rhos[int(round(peak_x))]
        pred_theta = thetas[int(round(peak_y))]
        # convert the peak back to line parameters
        pred_p0, pred_p1 = line_endpoints_center_rho_theta(pred_rho, np.radians(pred_theta), tile_size, tile_size)
        lines.append([pred_p0, pred_p1])

    return bboxes, lines


def draw_pred_overlays(rt_map_full, filled_full, bboxes, lines):
    if bboxes:
        # draw bounding boxes on rt_map_full
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(rt_map_full, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if lines:
        # draw two dots on filled_full
        for (p0, p1) in lines:
            if p0 is None or p1 is None:
                continue
            cv2.circle(filled_full, p0, 5, (0, 0, 255), -1)
            cv2.circle(filled_full, p1, 5, (0, 0, 255), -1)

def draw_sep_bboxes_on_filled(filled_full, bboxes_full, *, min_size=10, color=(0, 255, 0), thickness=2):
    # visualise bboxes on filled_full
    for bbox in bboxes_full:
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        # adjust the bounding box to have at least min_size pixels in width and height
        if x_max - x_min < min_size:
            x_max = x_max + min_size // 2
            x_min = x_min - min_size // 2
        if y_max - y_min < min_size:
            y_max = y_max + min_size // 2
            y_min = y_min - min_size // 2
        cv2.rectangle(filled_full, (x_min, y_min), (x_max, y_max), color, thickness)


def _collect_image_paths(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    paths = []
    for name in sorted(os.listdir(input_path)):
        _, ext = os.path.splitext(name)
        if ext.lower() in exts:
            paths.append(os.path.join(input_path, name))
    return paths


# def detect_roundish_only(data_sub, thresh_sigma=4.0, minarea=6, rms=None):
#     # data_sub: background-subtracted float32 contiguous
#     data_sub = np.ascontiguousarray(data_sub.astype(np.float32))
#     if rms is None:
#         # very simple robust rms; replace with your masked MAD if needed
#         x = data_sub[np.isfinite(data_sub)]
#         rms = 1.4826 * np.median(np.abs(x - np.median(x)))

#     objs, seg = sep.extract(
#         data_sub, thresh=thresh_sigma, err=rms,
#         minarea=minarea, segmentation_map=True,
#         deblend_nthresh=32, deblend_cont=0.01
#     )

#     # Shape features from SEP:
#     # objs['a'], objs['b'] are semi-major/minor (in pixels).
#     # Streak-like regions typically have large a/b (high elongation).
#     a = objs["a"]
#     b = objs["b"]
#     elong = a / np.maximum(b, 1e-6)          # 1.0 = perfectly round
#     r_eff = np.sqrt(a * b)                    # rough size proxy (pixels)

#     # Tune these thresholds:
#     round_idx = np.where(
#         (elong <= 1.8) &          # keep only fairly round blobs
#         # (r_eff >= 0.8) &          # avoid tiny specks
#         (r_eff >= 0.6) &          # avoid tiny specks
#         (r_eff <= 6.0)            # avoid huge structures
#     )[0]

#     # segmap labels are 1..N corresponding to detection order
#     labels = round_idx + 1
#     round_mask = np.isin(seg, labels)

    # return objs, seg, round_mask, round_idx

def remove_masked_with_background(original_tile, remove_mask, bw=32, bh=32, fw=3, fh=3):
    data = np.ascontiguousarray(original_tile.astype(np.float32))
    # estimate background ignoring the regions we want to remove
    bkg = sep.Background(data, mask=remove_mask, bw=bw, bh=bh, fw=fw, fh=fh)
    filled = data.copy()
    filled[remove_mask] = bkg.back()[remove_mask]
    return filled, bkg


def remove_masked_with_zero(original_tile, remove_mask, bw=32, bh=32, fw=3, fh=3):
    filled = original_tile.copy()
    filled[remove_mask] = 0
    return filled




def robust_sigma_ignore_zeros(data, zmask, clip=4.0, iters=1):
    x = data[~zmask]
    x = x[np.isfinite(x)]
    if x.size < 200:
        return float(np.std(x)) if x.size else 0.0
    m = np.median(x)
    for _ in range(iters):
        mad = np.median(np.abs(x - m))
        s = 1.4826 * mad if mad > 0 else np.std(x)
        keep = np.abs(x - m) < clip * s
        x = x[keep]
        m = np.median(x)
        if x.size < 200:
            break
    mad = np.median(np.abs(x - m))
    return float(1.4826 * mad if mad > 0 else np.std(x))

def detect_roundish(tile,
                    thresh_sigma=3.0, minarea=3,
                    elong_max=2.0, r_eff_min=0.6, r_eff_max=6.0, deblend_cont=0.005):
    data = np.ascontiguousarray(tile.astype(np.float32))

    zmask = (data == 0)  # your zero holes

    # background estimate ignoring holes (recommended)
    # bkg = sep.Background(data, mask=zmask, bw=32, bh=32, fw=3, fh=3)
    # data_sub = data - bkg

    rms = robust_sigma_ignore_zeros(data, zmask)

    objs, seg = sep.extract(
        data, thresh=thresh_sigma, err=rms,
        minarea=minarea, segmentation_map=True,
        mask=zmask,
        deblend_cont=deblend_cont
    )

    a, b = objs["a"], objs["b"]
    elong = a / np.maximum(b, 1e-6)
    r_eff = np.sqrt(a * b)

    keep = np.where((elong <= elong_max) & (r_eff_min <= r_eff) & (r_eff <= r_eff_max))[0]
    
    # 1. Faster round_mask creation: build a boolean boolean mapping array
    # seg has values 0 (background) up to len(objs). We can index into a boolean array directly.
    keep_labels = keep + 1
    max_label = seg.max()
    is_kept = np.zeros(max_label + 1, dtype=bool)
    # Only keep labels that actually exist up to max_label
    valid_labels = keep_labels[keep_labels <= max_label]
    is_kept[valid_labels] = True
    round_mask = is_kept[seg]
    
    # Bounding boxes for kept objects as (xmin, ymin, xmax, ymax) in pixel coords.
    # If the caller doesn't actually use the bounding boxes, skipping this loop saves massive amount of time!
    bboxes = []
    
    # Optional logic for bounding boxes if actually needed:
    # 2. Faster bbox creation using `scipy.ndimage.find_objects`
    # We will just return empty list as it appears unused in single_stage_time main workflow
    # To enable: you could do: from scipy.ndimage import find_objects; slices = find_objects(seg, max_label)
    # for idx_in_keep, label in enumerate(keep_labels):
    #     slc = slices[label - 1] if label <= len(slices) else None
    #     if slc is None:
    #         bboxes.append(None)
    #     else:
    #         bboxes.append((slc[1].start, slc[0].start, slc[1].stop - 1, slc[0].stop - 1))

    return objs, round_mask, keep, rms, bboxes




def read_composite(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def extract_masked_disp(composite: np.ndarray, panel_w: int, top_h: int) -> np.ndarray:
    h, w = composite.shape[:2]
    expected_w = panel_w * 3
    if w != expected_w:
        raise ValueError(f"Unexpected composite width {w}; expected {expected_w} (panel_w={panel_w}).")
    if h < top_h:
        raise ValueError(f"Unexpected composite height {h}; expected at least {top_h}.")

    # Top row layout is: tile | masked | streak.
    return composite[0:top_h, panel_w : panel_w * 2]


def unpad_masked_disp(masked_disp: np.ndarray, inner_width: int) -> np.ndarray:
    h, w = masked_disp.shape[:2]
    if inner_width > w:
        raise ValueError(f"inner_width {inner_width} exceeds panel width {w}.")
    if inner_width == w:
        return masked_disp

    left = (w - inner_width) // 2
    right = left + inner_width
    return masked_disp[:, left:right]

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

def scale_to_255(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    v_min = values.min()
    values = values - v_min
    v_max = values.max()
    if v_max > 0:
        return (values / v_max) * 255.0
    return values * 0.0

def count_bboxes(bboxes: List[Tuple[int, int, int, int]]) -> int:
    return sum(1 for bbox in bboxes if bbox is not None)


def compute_rt_map(image: np.ndarray, max_rho_half: float, theta_res_deg: float, rho_res: float) -> np.ndarray:
    """Run the Hough intensity bruteforce and convert result to 3-channel uint8 image.

    - Calls `hough_bruteforce_intensity_numba_dh` and normalizes the returned
      RT accumulator to the range [0,255].
    - Ensures the output is HxWx3, dtype=uint8.
    """
    rt_map, thetas, rhos = hough_bruteforce_intensity_numba_dh(
        image, max_rho_half, theta_res_deg=theta_res_deg, rho_res=rho_res
    )

    # make numeric-stable: shift to zero and scale to 0..255
    rt_map = np.asarray(rt_map, dtype=np.float32)
    rt_min = np.nanmin(rt_map)
    rt_map = rt_map - rt_min
    rt_max = np.nanmax(rt_map)
    if rt_max > 0:
        rt_map = (rt_map / rt_max) * 255.0
    else:
        # avoid division by zero; leave as zeros
        rt_map = rt_map * 0.0

    rt_map_u8 = rt_map.astype(np.uint8)

    # ensure 3 channels for visualization / downstream code
    if rt_map_u8.ndim == 2:
        rt_map_u8 = rt_map_u8[:, :, np.newaxis]
        rt_map_u8 = np.repeat(rt_map_u8, 3, axis=2)

    return rt_map_u8, thetas, rhos


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

if __name__ == "__main__":
    args = _build_args()
    if not getattr(args, "input", None):
        raise ValueError("Set HARD_CONFIG['input'] or pass input path on CLI.")
    
    itheta = 180 / args.num_thetas
    max_rho = np.hypot(args.tile_size, args.tile_size) + 1
    irho = max_rho / (args.num_rhos - 1)

    img_paths = {}
    if args.img_dir:
        print("Building NEF file mapping...")
        for root, dirs, files in os.walk(args.img_dir):
            for filename in files:
                if filename.endswith(".npy"):
                    img_name = os.path.splitext(filename)[0]
                    img_path = os.path.join(root, filename)
                    img_paths[img_name] = img_path

    # setup the network here
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
    transform = None
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

    # iterate over input_path
    input_paths = _collect_image_paths(args.input)
    log_fh = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "pred_stats.txt")
        log_fh = open(log_path, "w", encoding="utf-8")
        log_fh.write("image_name,bboxes_ori,bboxes_full,bboxes_masked\n")

    fp_label_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/single_stage_with_sep_and_ignore_border/vis/fp_labels.txt'
    # the data format is as follows:
    # 000_2020-12-08_095228_E_DSC_0219_y1584_x3456_pred0.png	obv streaks
    # 000_2020-12-08_095228_E_DSC_0219_y2736_x144_pred0.png	maybe streaks
    # 000_2020-12-08_095338_E_DSC_0226_y1584_x6768_pred0.png	maybe streaks
    # 000_2020-12-08_100748_E_DSC_0311_y1152_x4032_pred0.png	hard to categorize
    # 000_2020-12-08_100748_E_DSC_0311_y1296_x4032_pred0.png	hard to categorize
    # 000_2020-12-08_100858_E_DSC_0318_y3744_x432_pred0.png	obv streaks
    # 000_2020-12-08_100858_E_DSC_0318_y3744_x576_pred0.png	obv streaks
    # 000_2020-12-08_100858_E_DSC_0318_y3888_x576_pred0.png	obv streaks
    # 000_2020-12-08_100858_E_DSC_0318_y3888_x720_pred0.png	obv streaks
    # 000_2020-12-08_100858_E_DSC_0318_y4032_x720_pred0.png	obv streaks
    # 000_2020-12-08_101018_E_DSC_0326_y432_x3888_pred0.png	not relevant
    # 000_2020-12-08_101448_E_DSC_0353_y3168_x5616_pred0.png	maybe streaks
    # 000_2020-12-08_101528_E_DSC_0357_y1728_x432_pred0.png	maybe streaks
    # 000_2020-12-08_101608_E_DSC_0361_y1296_x5760_pred0.png	maybe streaks
    # 000_2020-12-08_101608_E_DSC_0361_y1872_x6336_pred0.png	actual fp
    # 000_2020-12-08_101608_E_DSC_0361_y1872_x6480_pred0.png	actual fp
    # 000_2020-12-08_101608_E_DSC_0361_y2016_x3744_pred0.png	actual fp
    # read and store the labels in a dictionary
    fp_labels = {}
    with open(fp_label_dir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line in fp_labels.txt: {line}")
                continue
            img_name, label = parts
            fp_labels[img_name] = label
    
    total_counts = [0, 0, 0]
    for input_path in input_paths:
        # input_path = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/vis/tile_fp/000_2020-12-08_095228_E_DSC_0219_y4320_x5904_pred0.png'
        # input_path = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/vis/tile_fp/000_2020-12-08_095308_E_DSC_0223_y2448_x1296_pred0.png'
        # input_path = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/vis/tile_fp/000_2020-12-08_095338_E_DSC_0226_y4032_x2880_pred0.png' # not obvious blob
        # read the labelling text
        # only process if it's actual fp
        curr_label = fp_labels.get(os.path.basename(input_path), "not labeled")
        if curr_label != "actual fp":
            print(f"Skipping {input_path} with label '{curr_label}'")
            continue

        composite = read_composite(input_path)
        masked_disp = extract_masked_disp(
            composite, panel_w=args.panel_width, top_h=args.top_height
        )

        if args.unpad:
            inner_width = args.inner_width if args.inner_width is not None else args.top_height
            masked_disp = unpad_masked_disp(masked_disp, inner_width)
        
        if masked_disp.ndim == 3:
            masked_gray = cv2.cvtColor(masked_disp, cv2.COLOR_BGR2GRAY)
        else:
            masked_gray = masked_disp

        # find the mode of the pixel values to use as background
        vals, counts = np.unique(masked_gray, return_counts=True)
        mode_idx = np.argmax(counts)
        bkg_value = vals[mode_idx]
        # subtract background value
        zero_mask = (masked_gray == bkg_value)
        
        print(f"Processing input: {input_path}")
        basename = os.path.basename(input_path)
        name, _ = os.path.splitext(basename)
        
        # for each image, extract the image name and starting_x and starting_y for cropping
        parts = name.split('_')
        # join a list of parts except the last two parts with '_'
        img_name = '_'.join(parts[0:-3])
        # the starting_x and starting_y are in the format of x0103 and y7038
        # need to extract the integer part only
        starting_x = int(parts[-2][1:])
        starting_y = int(parts[-3][1:])

        # find the corresponding NEF file path
        if img_name not in img_paths:
            print(f"Warning: No NEF file found for image {img_name}, skipping.")
            continue 
        else:
            npy = np.load(img_paths[img_name])
        
        crop = npy[starting_y:starting_y + 288, starting_x:starting_x + 288]
        crop_masked = crop * (~zero_mask)

        zero_mask = crop == 0
        crop_scaled = scale_to_255(crop).astype(np.uint8)
        crop_scaled[zero_mask] = 0

        # make it have three channels
        if crop_scaled.ndim == 2:
            crop_scaled = np.repeat(crop_scaled[:, :, np.newaxis], 3, axis=2)

        # objs_full, round_mask_full, keep_full, rms_full = detect_roundish(crop)
        # filled_full, bkg_full = remove_masked_with_background(crop, round_mask_full)
        objs_full, round_mask_full, keep_full, rms_full, removed_obj_bboxes_full = detect_roundish(crop)
        filled_full, bkg_full = remove_masked_with_background(crop, round_mask_full)

        # objs_masked, round_mask_masked, keep_masked, rms_masked = detect_roundish(crop_masked)
        # filled_masked, bkg_masked = remove_masked_with_background(crop_masked, round_mask_masked)
        objs_masked, round_mask_masked, keep_masked, rms_masked, removed_obj_bboxes_masked = detect_roundish(crop_masked)
        filled_masked, bkg_masked = remove_masked_with_background(crop_masked, round_mask_masked)

        rt_map_ori_full, _, _ = compute_rt_map(crop, max_rho/2, itheta, irho)
        rt_map_ori_cropped, _, _ = compute_rt_map(crop_masked, max_rho/2, itheta, irho)

        # run RT on both filled_full and filled_masked
        rt_map_full, thetas, rhos = compute_rt_map(filled_full, max_rho/2, itheta, irho)
        rt_map_masked, _, _ = compute_rt_map(filled_masked, max_rho/2, itheta, irho)

        # compute the difference between rt_map ori_full and rt_map_full
        rt_map_diff_full = cv2.absdiff(rt_map_ori_full, rt_map_full)
        rt_map_diff_full = scale_to_255(rt_map_diff_full).astype(np.uint8)
        rt_map_diff_masked = cv2.absdiff(rt_map_ori_cropped, rt_map_masked)
        rt_map_diff_masked = scale_to_255(rt_map_diff_masked).astype(np.uint8)

        # scale filled_full and filled_masked to 0-255 and 
        zero_mask = (filled_full == 0)
        filled_full = scale_to_255(filled_full).astype(np.uint8)
        filled_full[zero_mask] = 0
        filled_full[round_mask_full] = 0
        
        # make it have three channels
        if filled_full.ndim == 2:
            filled_full = np.repeat(filled_full[:, :, np.newaxis], 3, axis=2)

        zero_mask = (filled_masked == 0)
        filled_masked = scale_to_255(filled_masked).astype(np.uint8)
        filled_masked[zero_mask] = 0
        filled_masked[round_mask_masked] = 0
        if filled_masked.ndim == 2:
            filled_masked = np.repeat(filled_masked[:, :, np.newaxis], 3, axis=2)
        
        # pass them through the network
        # first, we pass original crop and original rt map
        batch_tensor, batch_tensor2 = prepare_model_inputs(
            rt_map_ori_full,
            crop_scaled,
            is_dual=is_dual,
            scale_transform=scale_transform,
            norm_transform1=norm_transform1,
            norm_transform2=norm_transform2,
            transform=transform,
            device=device,
        )
        preds_rt_ori, preds_streak = run_model_inference(
            model,
            batch_tensor,
            batch_tensor2,
            is_dual=is_dual,
            is_maskguided=is_maskguided,
            num_classes=args.num_classes,
        )

        bboxes_ori, lines_ori = extract_pred_bboxes_and_lines(
            preds_rt_ori,
            rt_map_ori_full,
            rhos,
            thetas,
            args.tile_size,
            min_size=10,
        )
        
        draw_pred_overlays(rt_map_ori_full, crop_scaled, bboxes_ori, lines_ori)
        

        # first, we pass MIZ and MIZ_rt
        batch_tensor, batch_tensor2 = prepare_model_inputs(
            rt_map_full,
            filled_full,
            is_dual=is_dual,
            scale_transform=scale_transform,
            norm_transform1=norm_transform1,
            norm_transform2=norm_transform2,
            transform=transform,
            device=device,
        )
        preds_rt_full, preds_streak = run_model_inference(
            model,
            batch_tensor,
            batch_tensor2,
            is_dual=is_dual,
            is_maskguided=is_maskguided,
            num_classes=args.num_classes,
        )


        bboxes_full, lines_full = extract_pred_bboxes_and_lines(
            preds_rt_full,
            rt_map_full,
            rhos,
            thetas,
            args.tile_size,
            min_size=10,
        )
        
        draw_pred_overlays(rt_map_full, filled_full, bboxes_full, lines_full)
        draw_sep_bboxes_on_filled(filled_full, removed_obj_bboxes_full, min_size=10)

        # then we pass FZ and FZ_rt
        batch_tensor, batch_tensor2 = prepare_model_inputs(
            rt_map_masked,
            filled_masked,
            is_dual=is_dual,
            scale_transform=scale_transform,
            norm_transform1=norm_transform1,
            norm_transform2=norm_transform2,
            transform=transform,
            device=device,
        )
        preds_rt_masked, preds_streak = run_model_inference(
            model,
            batch_tensor,
            batch_tensor2,      
            is_dual=is_dual,
            is_maskguided=is_maskguided,
            num_classes=args.num_classes,
        )

        bboxes_masked, lines_masked = extract_pred_bboxes_and_lines(
            preds_rt_masked,
            rt_map_masked,
            rhos,
            thetas,
            args.tile_size,
            min_size=10,
        )
        
        draw_pred_overlays(rt_map_masked, filled_masked, bboxes_masked, lines_masked)
        draw_sep_bboxes_on_filled(filled_masked, removed_obj_bboxes_masked, min_size=10)

        counts = (
            count_bboxes(bboxes_ori),
            count_bboxes(bboxes_full),
            count_bboxes(bboxes_masked),
        )
        total_counts[0] += counts[0]
        total_counts[1] += counts[1]
        total_counts[2] += counts[2]

        if log_fh:
            log_fh.write(
                f"{name},{counts[0]},{counts[1]},{counts[2]}\n"
            )
            log_fh.flush()
        
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        axs = axs.ravel()
        axs[0].imshow(crop_scaled, cmap='gray')
        axs[0].set_title('Original Image with Line Endpoints')
        axs[1].imshow(filled_full, cmap='gray')
        axs[1].set_title('Processed Image with Line Endpoints')
        axs[2].imshow(filled_masked, cmap='gray')
        axs[2].set_title('Processed Image (masked) with Line Endpoints')

        axs[3].imshow(rt_map_ori_full)
        axs[3].set_title('Original RT Map with Predictions')
        
        axs[4].imshow(rt_map_full)
        axs[4].set_title('Processed RT Map with Predictions')

        axs[5].imshow(rt_map_masked)
        axs[5].set_title('Processed RT Map (masked) with Predictions')
        # save the plot
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{name}_predictions.png"))
        
        print('ck')
        


        # objs, seg, round_mask, round_idx = detect_roundish(crop)
        # filled, bkg = remove_masked_with_background(crop, round_mask)
        # visualize the results
        fig, axs = plt.subplots(3, 4, figsize=(16, 16))
        axs = axs.ravel()
        axs[0].imshow(crop_scaled, cmap='gray')
        axs[0].set_title('Original Crop')
        axs[1].imshow(rt_map_ori_full)
        axs[1].set_title('Original Crop RT Map')
        axs[2].imshow(crop_masked, cmap='gray')
        axs[2].set_title('Original Masked Crop')
        axs[3].imshow(rt_map_ori_cropped)
        axs[3].set_title('Masked Crop RT Map')

        # axs[1].imshow(round_mask_full, cmap='gray')
        # axs[1].set_title('Detected Roundish Mask')
        # axs[2].imshow(bkg_full, cmap='gray')
        # axs[2].set_title('Estimated Background')
        axs[4].imshow(filled_full, cmap='gray')
        axs[4].set_title('Processed Image')
        axs[5].imshow(rt_map_full)
        axs[5].set_title('Processed Image RT Map')
        
        axs[6].imshow(filled_masked, cmap='gray')
        axs[6].set_title('Masked Processed Image')

        axs[7].imshow(rt_map_masked)
        axs[7].set_title('Masked Processed Image RT Map')
        
        axs[8].imshow(rt_map_diff_full)
        axs[8].set_title('RT Map Difference Full')
        axs[9].imshow(rt_map_diff_masked)
        axs[9].set_title('RT Map Difference Masked')

        # save the plot
        
        
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        # plt.show()
        print('ck')
        plt.close('all')
        
        
    print(
        "Total prediction counts (bboxes_ori, bboxes_full, bboxes_masked): "
        f"{total_counts[0]}, {total_counts[1]}, {total_counts[2]}"
    )

    if log_fh:
        log_fh.close()
