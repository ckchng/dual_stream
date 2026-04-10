import os
import cv2
import numpy as np
from typing import Tuple, Dict, List

def get_bboxes_from_mask(
    mask_path: str,
    target_value: Tuple[int, int, int],
    connectivity: int = 8
) -> List[Tuple[int, int, int, int]]:
    """
    Given an RGB mask image, find all connected components where
    pixel == target_value and return tight bounding boxes.

    Args:
        mask_path: Path to the RGB mask image.
        target_value: (R, G, B) tuple for the region of interest.
        connectivity: 4 or 8 for connectedComponentsWithStats.

    Returns:
        List of bounding boxes as (x1, y1, x2, y2) in pixel coordinates.
    """
    # Read image as BGR
    img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {mask_path}")

    # Convert BGR -> RGB so target_value is in RGB as requested
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Build binary mask: 1 where pixel == target_value, 0 otherwise
    target_arr = np.array(target_value, dtype=img_rgb.dtype)
    binary_mask = np.all(img_rgb == target_arr, axis=-1).astype(np.uint8)

    # If nothing matches, return empty list
    if binary_mask.max() == 0:
        return []

    # Connected components: labels + statistics
    # stats: [label, x, y, width, height, area]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=connectivity
    )

    bboxes = []
    # Label 0 is background, skip it
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area == 0:
            continue
        x1, y1 = x, y
        x2, y2 = x + w - 1, y + h - 1  # inclusive coordinates
        # filter out small boxe
        if w < 10 or h < 10:
            continue
        bboxes.append((x1, y1, x2, y2))

    return bboxes


def get_bboxes_from_dir(
    mask_dir: str,
    target_value: Tuple[int, int, int],
    connectivity: int = 8
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """
    Process all images in a directory and extract bounding boxes
    for connected regions matching target_value.

    Args:
        mask_dir: Directory containing RGB mask images.
        target_value: (R, G, B) tuple for the region of interest.
        connectivity: 4 or 8.

    Returns:
        Dict mapping filename -> list of (x1, y1, x2, y2) boxes.
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    results: Dict[str, List[Tuple[int, int, int, int]]] = {}

    for fname in sorted(os.listdir(mask_dir)):
        if 's2' in fname:
            continue
        if not fname.lower().endswith(valid_exts):
            continue
        
        if 'blend' in fname:
            continue

        fpath = os.path.join(mask_dir, fname)
        try:
            bboxes = get_bboxes_from_mask(fpath, target_value, connectivity)
            results[fname] = bboxes
        except Exception as e:
            print(f"[WARNING] Skipping {fpath}: {e}")

    return results


if __name__ == "__main__":
    # Example usage
    pred_mask_dir = "/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_15_new_bg_w_blobs/synth_wo_blobs_val_set_result/vis/"
    # Example: foreground encoded as pure red in RGB
    target_value = (244, 35, 232)

    all_pred_boxes = get_bboxes_from_dir(pred_mask_dir, target_value)
    # for fname, boxes in all_pred_boxes.items():
    #     print(f"{fname}:")
    #     for box in boxes:
    #         x1, y1, x2, y2 = box
    #         print(f"  bbox: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

    # find the corresponding image, crop the bbox region and save the coordinates of the peak intensity
    gt_img_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_272_50_bg/images/train/'
    gt_img_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/testing_gray_rt_288_subset/images/'
    gt_img_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_288_snr_1_15_new_bg_w_blobs_set_2/images/train/'
    gt_img_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_288_blobs_only/images/train/'
    # gt_img_dir = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_288_snr_1_25_longer_wider/images/train/'
    output_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_15_new_bg_w_blobs/synth_wo_blobs_val_set_result/peak_bbox_output_larger_than10/'

    os.makedirs(output_dir, exist_ok=True)

    for fname, boxes in all_pred_boxes.items():
        gt_img_path = os.path.join(gt_img_dir, fname)
        gt_img = cv2.imread(gt_img_path)
        gt_img = gt_img[:,:,0]  # assuming grayscale, take one channel

        peak_loc = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop_img = gt_img[y1:y2+1, x1:x2+1]  # inclusive

            # Find peak intensity location in cropped image
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(crop_img)
            peak_x = x1 + maxLoc[0]
            peak_y = y1 + maxLoc[1]
            peak_loc.append([peak_x, peak_y])

            # print(f"{fname} - BBox {idx}: Peak Intensity at (x={peak_x}, y={peak_y}), Value={maxVal}")

        # Save the bboxes and peak locations to a text file
        save_path = os.path.join(output_dir, fname.replace('.png', '_pred.txt'))

        with open(save_path, 'w') as f:
            for box, peak in zip(boxes, peak_loc):
                f.write(f"bbox: {box}, peak: {peak}\n")
