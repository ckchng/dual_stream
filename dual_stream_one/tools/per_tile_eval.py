import os, json, math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
import ast
import re
from typing import List, Tuple, Any

from utils.HT_utils import _make_params_dh, _make_params


# ---------------- hard-coded ----------------
PRED_DIR = "/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_15_new_bg_w_blobs/synth_val_set_result/peak_bbox_output_larger_than10/"
GT_DIR      = "/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_288_snr_1_15_new_bg_w_blobs_set_2/poly_labels/train/"            # <image_id>.txt, one line, last 4 = normalized (xc,yc,w,h)
IMG_DIR     = "/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_288_snr_1_15_new_bg_w_blobs_set_2/actual_images/train/"             # where raw images live (fallback if file_name not usable)
RT_DIR      = "/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_288_snr_1_15_new_bg_w_blobs_set_2/images/train/"
OUT_DIR     = "/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_ft_1_25/last/synthdata_result/line_vis_larger_than10/"            # where to save visualizations
# RT_W = 192
# RT_H = 416
RT_W = 416
RT_H = 192
IMG_W = IMG_H = 288
FULL_IMG_W = 7380
FULL_IMG_H = 4928
NUM_ANGLES=192
NUM_RHOS=416

IOU_THR = 0.1
SCORE_THR = 0.0
CATEGORY_ID = 1
PRED_BBOX_FMT = "coco_xywh_pixels"
COL_PRED = (0, 255, 0)  # Green
COL_GT = (0, 0, 255)    # Blue
OUT_TP_DIR = OUT_DIR + "/tp/"
OUT_FP_DIR = OUT_DIR + "/fp/"
OUT_FN_DIR = OUT_DIR + "/fn/"



os.makedirs(OUT_DIR +'/img/', exist_ok=True)
os.makedirs(OUT_DIR +'/rt/', exist_ok=True)
os.makedirs(OUT_DIR +'/combined/', exist_ok=True)
os.makedirs(OUT_TP_DIR + '/img/', exist_ok=True)
os.makedirs(OUT_TP_DIR + '/rt/', exist_ok=True)
os.makedirs(OUT_TP_DIR + '/combined/', exist_ok=True)
os.makedirs(OUT_FP_DIR + '/img/', exist_ok=True)
os.makedirs(OUT_FP_DIR + '/rt/', exist_ok=True)
os.makedirs(OUT_FP_DIR + '/combined/', exist_ok=True)
os.makedirs(OUT_FN_DIR + '/img/', exist_ok=True)
os.makedirs(OUT_FN_DIR + '/rt/', exist_ok=True)
os.makedirs(OUT_FN_DIR + '/combined/', exist_ok=True)

# ---------------- helpers ----------------
def coco_xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def polygon_to_xyxy(poly):
    x1, y1, x2, y2, x3, y3, x4, y4 = poly
    xc = (x1 + x2 + x3 + x4) / 4.0
    yc = (y1 + y2 + y3 + y4) / 4.0
    w = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
    h = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
    return [xc - w/2, yc - h/2, xc + w/2, yc + h/2]

def cxcywh_norm_to_xyxy(b):
    cx, cy, w, h = b
    cx, cy, w, h = cx*RT_W, cy*RT_H, w*RT_W, h*RT_H
    # cx, cy, w, h = cx*IMG_W, cy*IMG_H, w*IMG_W, h*IMG_H
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    ub = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def read_det_xyxy(preds):
    pred_boxes = []
    for p in preds:
        pred_boxes.append(pred_to_xyxy(p))
    return pred_boxes

def read_det_xyxy_and_score(preds):
    pred_boxes = []
    pred_scores = []
    for p in preds:
        pred_boxes.append(pred_to_xyxy(p))
        pred_scores.append(p.get("score", 1.0))
    return pred_boxes, pred_scores


def read_gt_xyxy(gt_path):
    """Return list of xyxy boxes or None if empty/missing/malformed."""
    if not os.path.isfile(gt_path):
        raise ValueError(f"GT file not found: {gt_path}")

    boxes = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # expect: class_id x1 y1 x2 y2 x3 y3 x4 y4  -> 1 + 8 = 9 tokens
            if len(parts) < 9:
                continue

            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[1:9])
            xc = (x1 + x2 + x3 + x4) / 4.0
            yc = (y1 + y2 + y3 + y4) / 4.0
            w = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
            h = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

            xyxy = cxcywh_norm_to_xyxy([xc, yc, w, h])
            boxes.append(xyxy)

    if not boxes:
        return None
    return boxes

def pred_to_xyxy(p):
    if PRED_BBOX_FMT == "coco_xywh_pixels":
        return coco_xywh_to_xyxy(p["bbox"])
    elif PRED_BBOX_FMT == "cxcywh_norm":
        return cxcywh_norm_to_xyxy(p["bbox"])
    elif PRED_BBOX_FMT == "polygon_pixels":
        return polygon_to_xyxy(p["bbox"])
    else:
        raise ValueError("Unsupported PRED_BBOX_FMT")

def load_image(image_id, preds):
    # Prefer file_name from predictions if present; else fallback to <image_id>.png in IMG_DIR
    img_path = None
    if preds:
        fn = preds[0].get("file_name")
        if fn:
            # handle absolute vs relative
            img_path = fn if os.path.isfile(fn) else os.path.join(IMG_DIR, os.path.basename(fn))
    if not img_path:
        img_path = os.path.join(IMG_DIR, f"{image_id}.png")
        if not os.path.isfile(img_path):
            img_path = os.path.join(IMG_DIR, f"{image_id}.jpg")
    if not os.path.isfile(img_path):
        return None, None
    return cv2.imread(img_path), os.path.basename(img_path)

def draw_box(img, xyxy, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        tsize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tw, th = tsize
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


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
    cx = W / 2
    cy = H / 2
    # cx = (W - 1) / 2.0
    # cy = (H - 1) / 2.0


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


# ---------- tiny helpers ----------
def _make_params(h, w, theta_res_deg=1.0, rho_res=1.0):
    thetas_deg = np.arange(-90.0, 90.0, theta_res_deg, dtype=np.float64)
    thetas = np.deg2rad(thetas_deg)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    max_rho = np.hypot(h, w) / 2.0
    rhos = np.arange(-max_rho, max_rho + rho_res, rho_res, dtype=np.float64)
    return thetas_deg, thetas, cos_t, sin_t, rhos

def hough_bruteforce_intensity_numba(img, theta_res_deg=1.0, rho_res=1.0):
    h, w = img.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    theta_deg, theta_rad, cos_t, sin_t, rhos = _make_params(h, w, theta_res_deg, rho_res)

    # Ensure numeric type Numba likes
    img32 = img.astype(np.float32, copy=False)
    H = _hough_accumulate_intensity(img32, cos_t, sin_t, rhos, cx, cy, rho_res)
    return H, theta_deg, rhos


# ---------- core (numba) ----------
@njit(parallel=True, fastmath=True)
def _hough_accumulate_intensity(img, cos_t, sin_t, rhos, cx, cy, rho_res):
    h, w = img.shape
    T = cos_t.shape[0]
    R = rhos.shape[0]
    H = np.zeros((R, T), dtype=np.float32)

    # Parallelize over angles so each thread writes to its own column -> no races.
    for t_idx in prange(T):
        c = cos_t[t_idx]
        s = sin_t[t_idx]
        r0 = rhos[0]
        for y in range(h):
            Y = y - cy
            for x in range(w):
                v = img[y, x]
                if v == 0:
                    continue
                X = x - cx
                rho = X * c + Y * s
                r_idx = int((rho - r0) / rho_res + 0.5)   # round
                if 0 <= r_idx < R:
                    H[r_idx, t_idx] += v
    return H

def eval_detection(preds, gts, iou_thr=0.8):
    # preds = load_boxes(PRED_PATH)
    # gts   = load_boxes(GT_PATH)

     # Handle empty / missing cases explicitly
    if len(preds) == 0 and gts is None:
        return 0, 0, 0, [], [], []

    if len(preds) == 0:
        fn_gt_ids = list(range(len(gts)))
        return 0, 0, len(gts), [], [], fn_gt_ids

    if gts is None:
        fp_pred_ids = list(range(len(preds)))
        return 0, len(preds), 0, [], fp_pred_ids, []

    gt_matched = np.zeros(len(gts), dtype=bool)
    tp_pred_ids = []
    fp_pred_ids = []

    for pi, p in enumerate(preds):
        best_iou = 0.0
        best_j = -1
        for gj, g in enumerate(gts):
            if gt_matched[gj]:
                continue
            i = iou_xyxy(p, g)
            if i > best_iou:
                best_iou = i
                best_j = gj
        if best_iou >= iou_thr and best_j >= 0:
            tp_pred_ids.append([pi, best_j])
            gt_matched[best_j] = True
        else:
            fp_pred_ids.append([pi])

    fn_gt_ids = [j for j, m in enumerate(gt_matched) if not m]

    tp = len(tp_pred_ids)
    fp = len(fp_pred_ids)
    fn = len(fn_gt_ids)

    return tp, fp, fn, tp_pred_ids, fp_pred_ids, fn_gt_ids

def nms_keep_highest(preds, scores, iou_thr=0.5):
    """
    preds: np.ndarray of shape (N, 5) -> [x1,y1,x2,y2,score]
    Returns:
        kept_preds: filtered preds (same format)
        kept_indices: indices in the original preds array
    """
    if len(preds) == 0:
        return preds, []

    # sort by score descending
    order = np.argsort(-scores)
    preds = preds[order]
    scores = scores[order]

    kept = []
    kept_indices = []

    for i, p in enumerate(preds):
        keep = True
        for k in kept:
            if iou_xyxy(p[:4], k[:4]) >= iou_thr:
                keep = False
                break
        if keep:
            kept.append(p)
            kept_indices.append(order[i])  # map back to original index

    return np.vstack(kept), kept_indices

def read_boxes_and_peaks(txt_path: str) -> Tuple[List[Any], List[Any]]:
    """
    Read a text file where each line looks like:
        bbox: (x1, x2, x3, x4), peak: (y1, y2)
    or with lists, etc., and return lists of boxes and peaks.

    Returns:
        boxes: list of parsed bbox objects (tuples/lists/etc.)
        peaks: list of parsed peak objects
    """
    boxes = []
    peaks = []

    pattern = re.compile(r"bbox:\s*(.+?),\s*peak:\s*(.+)$")

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = pattern.match(line)
            if not m:
                # skip or raise, depending on how strict you want to be
                # raise ValueError(f"Line not in expected format: {line}")
                continue

            box_str, peak_str = m.groups()

            # Safely parse Python literals like (1, 2, 3, 4) or [1, 2, 3, 4]
            box = ast.literal_eval(box_str)
            peak = ast.literal_eval(peak_str)

            boxes.append(box)
            peaks.append(peak)

    return boxes, peaks

if __name__ == "__main__":
    # ---------------- load predictions ----------------
    gt_files = os.listdir(GT_DIR)

    itheta = 180 / NUM_ANGLES
    max_rho = np.hypot(IMG_W, IMG_H) + 1
    irho = max_rho / (NUM_RHOS - 1)
    #### prepare RT params
    # thetas_deg, thetas, cos_t, sin_t, rhos = _make_params(IMG_H, IMG_W, theta_res_deg=1.0, rho_res=1.0)
    thetas_deg, thetas, cos_t, sin_t, rhos = _make_params_dh(max_rho / 2, theta_res_deg=itheta, rho_res=irho)

    # ---------------- evaluate + visualize ----------------
    per_image = {}  # image_id -> (TP, FP, FN)
    tp_diags, fn_diags = [], []

    total_tp = 0
    total_fp = 0
    total_fn = 0
                
    for gt_file in gt_files:
        prefix = gt_file.split('.txt')[0]
        image_id = f"{prefix}"
        
        # tile_x = int(prefix.split('_')[-2])
        # tile_y = int(prefix.split('_')[-1])

        # correct tile_x and tile_y, if it plus IMG_W/H exceeds image dimensions, adjust tile_x/y to fit within bounds
        # start_x = tile_x if tile_x + IMG_W <= FULL_IMG_W else FULL_IMG_W - IMG_W
        # start_y = tile_y if tile_y + IMG_H <= FULL_IMG_H else FULL_IMG_H - IMG_H

        # prefix = prefix.split('_')
        # image_id = '_'.join(prefix[:-3])
        
        # preds_path = f"{image_id}_tile_{start_x}_{start_y}_preds.txt"
        preds_path = f"{prefix}_pred.txt"
        print('processing ' + prefix + '\n')
        
        
        # gt_path = os.path.join(GT_DIR, f"{image_id}.txt")
        # modify read_gt_xyxy to read polygon and convert to box
        gt_xyxy = read_gt_xyxy(os.path.join(GT_DIR, gt_file))
        pred_xyxy, pred_peak = read_boxes_and_peaks(os.path.join(PRED_DIR, preds_path))

        if gt_xyxy is None and len(pred_xyxy) == 0:
            continue
        # pred_xyxy, pred_score = read_det_xyxy_and_score(preds)
        # pred_xyxy, kept_ids = nms_keep_highest(np.array(pred_xyxy), np.array(pred_score), iou_thr=0.4)
        tp, fp, fn, tp_pred_ids, fp_pred_ids, fn_gt_ids = eval_detection(pred_xyxy, gt_xyxy, iou_thr=IOU_THR)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # -------- for visualization --------
        # pred_rho = 135.6669
        # pred_theta = 0.0
        # pred_p0, pred_p1 = line_endpoints_center_rho_theta(pred_rho, pred_theta, IMG_H, IMG_W)

        #then, visualie tp and save to tp_out_dir
        img, basename = load_image(image_id, [])
        rt_map = cv2.imread(os.path.join(RT_DIR, f"{image_id}.png"))

        if tp > 0:
            
            for tp_pred_id in tp_pred_ids:
                pred_idx, gt_idx = tp_pred_id
                pred_box = pred_xyxy[pred_idx]
                pred_peak_point = pred_peak[pred_idx]
                gt_box = gt_xyxy[gt_idx]
                draw_box(rt_map, gt_box, COL_GT, thickness=4)
                draw_box(rt_map, pred_box, COL_PRED)

                

                # draw two points on the image instead of a line
                x1, y1, x2, y2 = pred_box
                # clip center_x within 0 to 179
                # center_x = np.clip((x1 + x2) / 2, 0, len(thetas)-1)
                # clip center_y within 0 to 863
                # center_y = np.clip((y1 + y2) / 2, 0, len(rhos)-1)
                center_x = pred_peak_point[0]
                center_y = pred_peak_point[1]

                pred_rho = rhos[int(round(center_x))]
                pred_theta = thetas[int(round(center_y))]
                # pred_rho = rhos[int(round(center_y))]
                # pred_theta = thetas[int(round(center_x))]
                

                # determine the pixels coordinates that lies on the rho and theta
                pred_p0, pred_p1 = line_endpoints_center_rho_theta(pred_rho, pred_theta, IMG_H, IMG_W)

                x1, y1, x2, y2 = gt_box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2


                gt_rho = rhos[int(round(center_x))]
                gt_theta = thetas[int(round(center_y))]
                # gt_rho = rhos[int(round(center_y))]
                # gt_theta = thetas[int(round(center_x))]

                # determine the pixels coordinates that lies on the rho and theta
                gt_p0, gt_p1 = line_endpoints_center_rho_theta(gt_rho, gt_theta, IMG_H, IMG_W)

                # convert to rho and
                if pred_p0 is not None and pred_p1 is not None:
                    cv2.circle(img, (pred_p0[0], pred_p0[1]), 5, COL_PRED, -1)
                    cv2.circle(img, (pred_p1[0], pred_p1[1]), 5, COL_PRED, -1)
                
                if gt_p0 is not None and gt_p1 is not None:
                    cv2.circle(img, (gt_p0[0], gt_p0[1]), 3, COL_GT, -1)
                    cv2.circle(img, (gt_p1[0], gt_p1[1]), 3, COL_GT, -1)

                

        if fp > 0:
            # img, basename = load_image(image_id, preds)
            # rt_map = cv2.imread(os.path.join(RT_DIR, f"{image_id}.png"))
            for fp_pred_id in fp_pred_ids:
                pred_idx = fp_pred_id
                pred_idx = np.squeeze(pred_idx)
                pred_box = pred_xyxy[pred_idx]
                pred_peak_point = pred_peak[pred_idx]
                draw_box(rt_map, pred_box, COL_PRED)

                # draw two points on the image instead of a line
                x1, y1, x2, y2 = pred_box
                # center_x = np.clip((x1 + x2) / 2, 0, len(thetas)-1)
                # # clip center_y within 0 to 863
                # center_y = np.clip((y1 + y2) / 2, 0, len(rhos)-1)

                center_x = pred_peak_point[0]
                center_y = pred_peak_point[1]

                pred_rho = rhos[int(round(center_x))]
                pred_theta = thetas[int(round(center_y))]
                # pred_rho = rhos[int(round(center_y))]
                # pred_theta = thetas[int(round(center_x))]

                # determine the pixels coordinates that lies on the rho and theta
                pred_p0, pred_p1 = line_endpoints_center_rho_theta(pred_rho, pred_theta, IMG_H, IMG_W)

                if pred_p0 is not None and pred_p1 is not None:
                    cv2.circle(img, (pred_p0[0], pred_p0[1]), 5, COL_PRED, -1)
                    cv2.circle(img, (pred_p1[0], pred_p1[1]), 5, COL_PRED, -1)

        if fn > 0:
            # img, basename = load_image(image_id, preds)
            # rt_map = cv2.imread(os.path.join(RT_DIR, f"{image_id}.png"))
            for fn_gt_id in fn_gt_ids:
                gt_idx = fn_gt_id
                gt_box = gt_xyxy[gt_idx]
                draw_box(rt_map, gt_box, COL_GT, thickness=4)
                

                # draw two points on the image instead of a line
                x1, y1, x2, y2 = gt_box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                gt_rho = rhos[int(round(center_x))]
                gt_theta = thetas[int(round(center_y))]
                # gt_rho = rhos[int(round(center_y))]
                # gt_theta = thetas[int(round(center_x))]

                # determine the pixels coordinates that lies on the rho and theta
                gt_p0, gt_p1 = line_endpoints_center_rho_theta(gt_rho, gt_theta, IMG_H, IMG_W)

                if gt_p0 is not None and gt_p1 is not None:
                    cv2.circle(img, (gt_p0[0], gt_p0[1]), 3, COL_GT, -1)
                    cv2.circle(img, (gt_p1[0], gt_p1[1]), 3, COL_GT, -1)

        # open a text file, write them
        curr_stats_dir = '/stats_per_image.txt'
        with open(OUT_DIR + curr_stats_dir, 'a') as f:
            f.write(prefix + ' ' + str(tp) + ' ' + str(fp) + ' ' + str(fn) + '\n')

        # write on RT_map the count of TP/FP/FN
        cv2.putText(rt_map, f'TP: {tp}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, COL_PRED, 2, cv2.LINE_AA)
        cv2.putText(rt_map, f'FP: {fp}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, COL_PRED, 2, cv2.LINE_AA)
        cv2.putText(rt_map, f'FN: {fn}', (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, COL_PRED, 2, cv2.LINE_AA)

        # combine img and rt_map side by side horizontally
        # pad img and rt_map to have same height if needed
        if img.shape[0] != rt_map.shape[0]:
            max_h = max(img.shape[0], rt_map.shape[0])
            if img.shape[0] < max_h:
                pad_h = max_h - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            if rt_map.shape[0] < max_h:
                pad_h = max_h - rt_map.shape[0]
                rt_map = cv2.copyMakeBorder(rt_map, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        combined = np.hstack((img, rt_map))
        if fp > 0:
            cv2.imwrite(os.path.join(OUT_FP_DIR + '/combined/', f"{image_id}.png"), combined)

        if tp > 0:
            cv2.imwrite(os.path.join(OUT_TP_DIR + '/combined/', f"{image_id}.png"), combined)

        if fn > 0:
            cv2.imwrite(os.path.join(OUT_FN_DIR + '/combined/', f"{image_id}.png"), combined)
        # cv2.imwrite(os.path.join(OUT_DIR + '/rt/', f"{image_id}.png"), rt_map)
        # cv2.imwrite(os.path.join(OUT_DIR + '/img/', f"{image_id}.png"), img)

    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}")    