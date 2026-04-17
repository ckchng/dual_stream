import os
import argparse
import math
import numpy as np
import cv2
import json
from typing import List, Tuple
from glob import glob
# from tqdm import tqdm 

USE_ARGPARSE = False
HARD_CONFIG = dict(
    # input_dir="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/rt_only/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/txt/",
    # input_dir="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25/two_classes/both/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/txt/",
    # input_dir="/home/ckchng/Documents/dual_stream/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/both_run5/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/txt/",
    # input_dir="/home/ckchng/Documents/dual_stream/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/multi_model_eval/both_run5/full_frame_result/txt/",
    input_dir="/home/ckchng/Documents/dual_stream_two/save/bg_50_no_crop/snr_1_32_len_200/single_class/multi_model_eval/both_run2_sep_3_6_6_0.6_6_0.1/full_frame_result/txt/",
    max_angle=5,    # degrees
    max_dist=10.0,    # pixels
    eps=1e-2,
    img_dir= "/media/ckchng/internal2TB/FILTERED_IMAGES/",
    anno_json= "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_and_rt_two_stage_labeled_merged_labels.json",
    # output_dir="/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/rt_only/single_stage/vis/params_3.0_6_6.0_0.6_6.0_0.1/full_frame_result/merged_txt",
    # output_dir="/home/ckchng/Documents/dual_stream/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/multi_model_eval/both_run5/full_frame_result/merged_txt",
    output_dir="/home/ckchng/Documents/dual_stream_two/save/bg_50_no_crop/snr_1_32_len_200/single_class/multi_model_eval/both_run2_sep_3_6_6_0.6_6_0.1/full_frame_result/merged_txt",
)

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

def merge_connected_segments_2d(segments: List[Segment], max_angle_deg: float = 2.0, max_dist_px: float = 5.0, eps: float = 1e-6):
    """
    Merge 2D line segments that lie on the same line and either overlap or touch.
    Each segment is ((x1, y1), (x2, y2)).
    
    Args:
        segments: List of ((x1, y1), (x2, y2))
        max_angle_deg: Maximum angle difference in degrees for segments to be considered parallel.
        max_dist_px: Maximum perpendicular distance in pixels for segments to be considered collinear.
        eps: Epsilon for floating point comparisons (e.g., gap between segments to consider touching).
    """

    def dist(p: Point, q: Point) -> float:
        return math.hypot(p[0] - q[0], p[1] - q[1])

    def points_equal(p: Point, q: Point) -> bool:
        return dist(p, q) <= eps

    def are_colinear(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
        """
        Check if segments a1-a2 and b1-b2 lie on the same infinite line
        within angle and distance thresholds.
        """
        # Vector 1
        vx, vy = a2[0] - a1[0], a2[1] - a1[1]
        mag_v = math.hypot(vx, vy)
        
        # If segment is too short, direction is unstable. handle if needed, or assume valid segments.
        # Here we just protect against division by zero.
        if mag_v < eps:
            return False 
            
        # Normalize Vector 1
        ux, uy = vx / mag_v, vy / mag_v

        # Vector 2
        wx, wy = b2[0] - b1[0], b2[1] - b1[1]
        mag_w = math.hypot(wx, wy)
        if mag_w < eps:
            return False

        # Normalize Vector 2
        zx, zy = wx / mag_w, wy / mag_w

        # 1. Check direction parallelism (dot product)
        # cos(theta)
        dot_val = ux * zx + uy * zy
        
        # Threshold for cos(angle). 
        # If angle is small, cos(angle) is close to 1.
        # We check abs(dot_val) because segments can be anti-parallel (opposite direction) 
        # but still on the same line.
        cos_thres = math.cos(math.radians(max_angle_deg))
        
        if abs(dot_val) < cos_thres:
            return False 

        # 2. Check overlap/distance to line
        # Vector w1 from a1 to b1
        rx, ry = b1[0] - a1[0], b1[1] - a1[1]
        
        # Perpendicular distance check using dot product projection:
        # Distance of point b1 from line defined by a1->a2
        # |rejection|^2 = |r|^2 - (r . u)^2
        
        r_mag_sq = rx*rx + ry*ry
        proj = rx * ux + ry * uy
        perp_dist_sq = r_mag_sq - proj * proj
        
        # Ensure non-negative due to float errors
        if perp_dist_sq < 0:
            perp_dist_sq = 0
            
        if perp_dist_sq > (max_dist_px ** 2):
            return False
            
        return True

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

    def bbox_overlap(s1: Segment, s2: Segment) -> bool:
        """
        Quick check if bounding boxes overlap with some margin.
        Margin is determined by max_dist_px (lateral) and eps (longitudinal gap).
        We use the larger of the two to be safe.
        """
        margin = max(max_dist_px, eps)
        
        x1_min = min(s1[0][0], s1[1][0])
        x1_max = max(s1[0][0], s1[1][0])
        y1_min = min(s1[0][1], s1[1][1])
        y1_max = max(s1[0][1], s1[1][1])
        
        x2_min = min(s2[0][0], s2[1][0])
        x2_max = max(s2[0][0], s2[1][0])
        y2_min = min(s2[0][1], s2[1][1])
        y2_max = max(s2[0][1], s2[1][1])
        
        # Check X overlap
        if x1_max < x2_min - margin or x2_max < x1_min - margin:
            return False
            
        # Check Y overlap
        if y1_max < y2_min - margin or y2_max < y1_min - margin:
            return False
            
        return True

    def try_merge(s1: Segment, s2: Segment):
        a1, a2 = s1
        b1, b2 = s2
        
        # 0) Pre-check: Bounding box overlap
        if not bbox_overlap(s1, s2):
            return None

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

    # Normalize input: allow each input item to be either
    #  - a Segment: ((x1,y1),(x2,y2)) in which case label=None
    #  - a tuple (Segment, label)
    segs = []
    for s in segments:
        if s is None:
            continue
        # If passed as (seg, label)
        if isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], tuple) and (isinstance(s[0][0], (int, float)) or isinstance(s[0][0], tuple)) is False:
            # fallthrough to other checks
            pass
        if isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], tuple) and isinstance(s[0][0], (int, float)):
            # s looks like a Segment ((x1,y1),(x2,y2))
            segs.append((s, None))
        elif isinstance(s, tuple) and len(s) == 3:
            # ( (x1,y1),(x2,y2), label )
            segs.append((s[0:2], s[2]))
        elif isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], tuple):
            # (segment, label) expected
            segs.append((s[0], s[1]))
        else:
            # try to be flexible for lists
            try:
                # assume list-like [x1,y1,x2,y2,label?]
                if len(s) >= 4:
                    x1, y1, x2, y2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
                    label = s[4] if len(s) > 4 else None
                    segs.append((((x1, y1), (x2, y2)), label))
                else:
                    # unknown format, skip
                    continue
            except Exception:
                continue
    changed = True
    # Counter used to create unique labels when merging different labels
    merge_unique_counter = 0

    while changed:
        changed = False
        n = len(segs)
        for i in range(n):
            if changed:
                break
            for j in range(i + 1, n):
                s1, l1 = segs[i]
                s2, l2 = segs[j]
                merged_seg = try_merge(s1, s2)
                if merged_seg is not None:
                    # determine merged label
                    merged_label = None
                    if l1 is None and l2 is None:
                        merged_label = None
                    elif l1 is None:
                        merged_label = l2
                    elif l2 is None:
                        merged_label = l1
                    else:
                        # both labels present
                        if l1 == l2:
                            merged_label = l1
                        else:
                            # assign a unique label to indicate a merged-different-labels case
                            # merge_unique_counter += 1
                            merged_label = "to_be_determined"

                    # remove i, j and insert merged
                    new_segs = []
                    for k, s in enumerate(segs):
                        if k not in (i, j):
                            new_segs.append(s)
                    new_segs.append((merged_seg, merged_label))
                    segs = new_segs
                    changed = True
                    break

    # Return list of (segment, label)
    return segs

def parse_line_string(line_str: str) -> Segment:
    """Parses a line string in 'x1,y1,x2,y2' format."""
    parts = line_str.strip().split(',')
    if len(parts) != 4:
         # Try split by space if comma fails, simple fallback
        parts = line_str.strip().split()
        if len(parts) != 4:
            return None
    
    try:
        x1, y1 = float(parts[0]), float(parts[1])
        x2, y2 = float(parts[2]), float(parts[3])
        return ((x1, y1), (x2, y2))
    except ValueError:
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Merge connected line segments in text files.")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Input directory containing .txt files.")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory to save merged files.")
    parser.add_argument("--max-angle", type=float, default=2.0, help="Max angle difference in degrees.")
    parser.add_argument("--max-dist", type=float, default=5.0, help="Max perpendicular distance in pixels.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for floating point comparisons.")
    parser.add_argument("--img-dir", type=str, default=None, help="Directory containing .npy image files.")
    parser.add_argument("--anno-json", type=str, default=None, help="Path to annotation JSON file.")
    return parser.parse_args()

def run(args):

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
    
    # Load GT annotations if provided
    gt_annotations = {}
    if args.anno_json and os.path.exists(args.anno_json):
        print("Loading GT annotations...")
        with open(args.anno_json, 'r') as f:
            data_json = json.load(f)
            
        # Create a mapping from image file name (basename) to annotations
        # Adjust logic to match how you map filenames
        image_id_map = {}
        for img_entry in data_json['images']:
             # Assuming file_name in json matches keys we use
             fname = os.path.splitext(os.path.basename(img_entry['file_name']))[0]
             image_id_map[img_entry['id']] = fname
        
        for anno in data_json['annotations']:
            img_id = anno['image_id']
            if img_id in image_id_map:
                fname = image_id_map[img_id]
                if fname not in gt_annotations:
                    gt_annotations[fname] = []
                gt_annotations[fname].append(anno['xyxy'])


    # Ensure output directory exists
    output_dir = args.output_dir + f'_ang{args.max_angle}_dist{args.max_dist}_eps{args.eps}'
    os.makedirs(output_dir, exist_ok=True)

    txt_dir = output_dir + '/txt/'
    os.makedirs(txt_dir, exist_ok=True)
    
    if args.img_dir:
        vis_dir = os.path.join(output_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Find all text files
    input_files = glob(os.path.join(args.input_dir, "*.txt"))
    print(f"Found {len(input_files)} text files in {args.input_dir}")
    
    count = 0
    total_lines_before = 0
    total_lines_after = 0

    for file_path in input_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(txt_dir, filename)
        
        segments = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if not line.strip():
                        continue
                    seg = parse_line_string(line)
                    if seg:
                        segments.append(seg)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        # Merge segments
        lines_before = len(segments)
        # Assuming args has max_angle and max_dist, defaulting if not present in Namespace 
        # (for HARD_CONFIG backward compatibility if keys missing, though we updated HARD_CONFIG)
        ma = getattr(args, 'max_angle', 2.0)
        md = getattr(args, 'max_dist', 5.0)
        
        merged_result = merge_connected_segments_2d(segments, max_angle_deg=ma, max_dist_px=md, eps=args.eps)
        # merged_result is list of (segment, label)
        merged_segments = [s for s, _ in merged_result]
        merged_labels = [l for _, l in merged_result]
        lines_after = len(merged_segments)

        total_lines_before += lines_before
        total_lines_after += lines_after

        print(f"{filename}: {lines_before} -> {lines_after}")
        
        # Write back
        try:
            with open(output_path, 'w') as f:
                for seg in merged_segments:
                    (x1, y1), (x2, y2) = seg
                    f.write(f"{x1},{y1},{x2},{y2}\n")
            count += 1
        except Exception as e:
             print(f"Error writing {output_path}: {e}")

        # Visualization
        # if args.img_dir:
        #     # Determine image key
        #     txt_basename = os.path.splitext(os.path.basename(file_path))[0]
        #     # Try to find matching image, txt file usually end with '_lines'
        #     img_key = txt_basename
        #     if img_key.endswith('_lines'):
        #         img_key = img_key[:-6] # remove _lines
            
        #     if img_key in img_paths:
        #         img_path = img_paths[img_key]
        #         try:
        #             img = np.load(img_path)
                    
        #             # Normalize and convert to BGR for visualization
        #             img_disp = img.copy()
                    
        #             # Simple min-max normalization to 0-255
        #             v_min, v_max = img_disp.min(), img_disp.max()
        #             if v_max > v_min:
        #                 img_disp = (img_disp - v_min) / (v_max - v_min) * 255.0
        #             else:
        #                 img_disp = np.zeros_like(img_disp)
                    
        #             img_disp = img_disp.astype(np.uint8)
                    
        #             # If single channel, make it 3 channel
        #             if len(img_disp.shape) == 2:
        #                 img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
        #             elif len(img_disp.shape) == 3 and img_disp.shape[2] == 1:
        #                 img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
                    
        #             # Draw merged lines
        #             for seg in merged_segments:
        #                 p1 = (int(round(seg[0][0])), int(round(seg[0][1])))
        #                 p2 = (int(round(seg[1][0])), int(round(seg[1][1])))
        #                 cv2.line(img_disp, p1, p2, (255, 0, 0), 2) # Blue lines
                    
        #             # Draw GT lines if available
        #             if img_key in gt_annotations:
        #                  for gt_line in gt_annotations[img_key]:
        #                      # gt_line is [x1, y1, x2, y2]
        #                      g1 = (int(round(gt_line[0])), int(round(gt_line[1])))
        #                      g2 = (int(round(gt_line[2])), int(round(gt_line[3])))
        #                      cv2.line(img_disp, g1, g2, (0, 0, 255), 2) # Red lines for GT

        #             # Save
        #             vis_path = os.path.join(vis_dir, f"{txt_basename}.png")
        #             # cv2.imwrite(vis_path, img_disp)
                    
        #         except Exception as e:
        #             print(f"Error visualizing {img_key}: {e}")


    print(f"Successfully processed {count} files.")
    print(f"Total lines before merging: {total_lines_before}")
    print(f"Total lines after merging: {total_lines_after}")



if __name__ == "__main__":
    if USE_ARGPARSE:
        args = parse_args()
    else:
        args = argparse.Namespace(**HARD_CONFIG)
    run(args)
