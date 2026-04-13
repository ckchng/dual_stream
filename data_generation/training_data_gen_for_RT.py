import cv2
import numpy as np
import os
import copy
from numba import njit

import random
import math
import matplotlib.pyplot as plt
import warnings
from itertools import chain

from ht_utils import (
    _compute_rt_kernel,
    _make_params,
    endpoints_to_rho_theta_mod,
    rho_theta_to_indices,
    compute_rt_map,
)

# Turn *all* NumPy RuntimeWarnings into errors
warnings.filterwarnings("error", category=RuntimeWarning)

def clip_line_to_tile(tile_x, tile_y, tile_width, tile_height, line):
    """
    Clip the line to the tile boundaries and adjust the line's coordinates to be relative to the tile.
    """
    # Define the tile boundaries
    left, right, top, bottom = tile_x, tile_x + tile_width, tile_y, tile_y + tile_height

    # Unpack line endpoints
    # (x1, y1), (x2, y2) = line
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]

    # Function to check if a point is inside the tile
    def is_inside_tile(x, y):
        return (left - 1) <= x <= (right + 1) and (top - 1) <= y <= (bottom + 1)

    # Function to find intersection with a tile edge
    def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        # Line AB represented as a1x + b1y = c1
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1

        # Line CD represented as a2x + b2y = c2
        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3

        determinant = a1 * b2 - a2 * b1

        if determinant != 0:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            return x, y
        return None

    def on_segment(p, q, r):
        """Given three colinear points p, q, r, check if point q lies on line segment 'pr'."""
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    def orientation(p, q, r):
        """Find orientation of ordered triplet (p, q, r).
        Returns 0 if p, q and r are colinear, 1 if Clockwise, 2 if Counterclockwise."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Colinear
        return 1 if val > 0 else 2  # Clock or counterclock wise

    def do_lines_intersect(p1, q1, p2, q2):
        """Returns True if the line segments 'p1q1' and 'p2q2' intersect."""
        # Find the four orientations needed for general and special cases
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        # p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        # p1, q1 and q2 are colinear and q2 lies on segment p1q1
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        # p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        # p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        # Doesn't fall in any of the above cases
        return False

    def distance(point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def furthest_points(points):
        """Returns the two furthest points from a list of 3 or 4 points."""
        if not 3 <= len(points) <= 4:
            raise ValueError("The function requires either 3 or 4 points.")

        max_distance = 0
        furthest_pair = (None, None)

        # Compare each pair of points
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = distance(points[i], points[j])
                if dist > max_distance:
                    max_distance = dist
                    furthest_pair = (points[i], points[j])

        return list(furthest_pair)

    points = []
    # if not is_inside_tile(x1, y1) and not is_inside_tile(x2, y2):
    #     return None
    # Find intersection points with the tile edges
    # Adjust or clip the points based on whether they are inside the tile
    if not is_inside_tile(x1, y1):
        # find which edge is x1, y1 closest to
        # closest_edge = closest_edge_to_point((tile_x, tile_y, tile_width, tile_height), (x1, y1))

        top_edge = [(left, top), (right, top)]
        bottom_edge = [(left, bottom), (right, bottom)]
        left_edge = [(left, top), (left, bottom)]
        right_edge = [(right, top), (right, bottom)]

        for edge in [top_edge, bottom_edge, left_edge, right_edge]:
            p1 = [x1, y1]
            q1 = [x2, y2]
            p2 = [edge[0][0], edge[0][1]]
            q2 = [edge[1][0], edge[1][1]]
            if do_lines_intersect(p1, q1, p2, q2):
                pt = intersection(p1[0], p1[1], q1[0], q1[1], p2[0], p2[1], q2[0], q2[1])
                if pt and is_inside_tile(*pt):
                    # x1, y1 = pt
                    points.append(pt)
                    # break
    else:
        points.append((x1, y1))

    if not is_inside_tile(x2, y2):
        # find which edge is x1, y1 closest to
        # check for all intersection, if there is one that's inside the tile, that's it

        top_edge = [(left, top), (right, top)]
        bottom_edge = [(left,bottom), (right, bottom)]
        left_edge = [(left, top), (left, bottom)]
        right_edge = [(right, top), (right, bottom)]

        for edge in [top_edge, bottom_edge, left_edge, right_edge]:
            p1 = [x1, y1]
            q1 = [x2, y2]
            p2 = [edge[0][0], edge[0][1]]
            q2 = [edge[1][0], edge[1][1]]
            if do_lines_intersect(p1, q1, p2, q2):
                pt = intersection(p1[0], p1[1], q1[0], q1[1], p2[0], p2[1], q2[0], q2[1])
                if pt and is_inside_tile(*pt):
                    # x1, y1 = pt
                    points.append(pt)
                    # break
    else:
        points.append((x2, y2))

    # in case the line intersects the whole tile
    if len(points) > 2:
        # return two furthest points
        points = furthest_points(points)

    # Return the two points that are inside or on the boundary of the tile, adjusted to tile coordinates
    if len(points) >= 2 and (np.linalg.norm(np.array([points[0][0], points[0][1]]) - np.array([points[1][0], points[1][1]])) > 10):
        return np.array([points[0][0] - tile_x, points[0][1] - tile_y, points[1][0] - tile_x, points[1][1] - tile_y])

    return None


def generate_random_line(min_x0, max_x0, min_y0, max_y0, length_range, length_ratio1, length_ratio2):
    # Randomly select start point, angle, and length
    x0 = random.uniform(min_x0, max_x0)
    y0 = random.uniform(min_y0, max_y0)

    len_rand = random.random()
    if len_rand <= length_ratio1:
        length = random.uniform(length_range[0, 0], length_range[0, 1])  # Assuming theta_range is in radians
    elif len_rand <= length_ratio2:
        length = random.uniform(length_range[1, 0], length_range[1, 1])  # Assuming theta_range is in radians
    else:
        length = random.uniform(length_range[2, 0], length_range[2, 1])  # Assuming theta_range is in radians

    theta = random.uniform(0, 2   * np.pi)

    # Calculate end point
    x1 = x0 + length * math.cos(theta)
    y1 = y0 + length * math.sin(theta)

    return [x0, y0, x1, y1]

def generate_and_trim_lines(tile_width, tile_height, offset, length_range, length_ratio1, length_ratio2, max_num_streak, length_min=200):
    x0_min = 0 - offset
    x0_max = tile_width + offset
    y0_min = 0 - offset
    y0_max = tile_height + offset

    # ensure all the adjusted_lines are not zero-length, if it is, regenerate
    adjusted_lines = []
    while True:
        lines = generate_streaks_and_remove_intersections(x0_min, x0_max, y0_min, y0_max, length_range, length_ratio1, length_ratio2, max_num_streak)

        ##### trim it to within the box
        adjusted_lines = []
        for line in lines:
            line = [(line[0], line[1]), (line[2], line[3])]
            adjusted_line = clip_line_to_tile(0, 0, tile_width, tile_height, line)


            if adjusted_line is not None:
                # check the lenght of the adjusted_line, if it's too small, discard it
                pt1 = np.array([adjusted_line[0], adjusted_line[1]])
                pt2 = np.array([adjusted_line[2], adjusted_line[3]])
                if np.linalg.norm(pt1 - pt2) < length_min:
                    adjusted_line = None
                else:    
                    adjusted_lines.append(adjusted_line)

        # note, this is assuming we have one line only.
        if adjusted_line is not None and (np.abs(adjusted_line[0] - adjusted_line[2]) > 5) and (np.abs(adjusted_line[1] - adjusted_line[3]) > 5):
            # print("Regenerating lines due to zero-length after adjustment.")
            break

    return adjusted_lines

def generate_streaks_and_remove_intersections(min_x0, max_x0, min_y0, max_y0, length_range, length_ratio1, length_ratio2, max_num_streak):
    lines = []
    for i in range(max_num_streak):
        lines.append(generate_random_line(min_x0, max_x0, min_y0, max_y0, length_range, length_ratio1, length_ratio2))

        # lines.append(generate_line_within_box(height, width))

    non_intersecting_lines = remove_intersecting_lines(lines)
    return non_intersecting_lines


def lines_to_synth_patch(tile, lines, snr_range, sigma_range, lc_width, max_phi,
                         tile_width, tile_height, snr_ratio, sigma_ratio, scale_flag,
                         zero_mask, ori_min,
                         num_angles, num_rhos, max_rho, irho, itheta,
                         rho_min_cap=None, rho_max_cap=None, debug=False):
    synth_patch = copy.deepcopy(tile)

    remaining_lines = []
    remaining_poly = []
    len_stack = []
    snr_stack = []
    sigma_stack = []

    cx = (tile_width) / 2.0
    cy = (tile_height) / 2.0
    
    
    theta_min = 0
    
    for line_idx, line in enumerate(lines):
        all_xy = bresenham(*line)
        length = np.linalg.norm(all_xy[0, :] - all_xy[-1, :])

        snr_rand_float = random.random()
        if snr_rand_float <= snr_ratio:
            snr = random.uniform(snr_range[0, 0], snr_range[0, 1])
        else:
            snr = random.uniform(snr_range[1, 0], snr_range[1, 1])


        sigma_rand_float = random.random()
        if sigma_rand_float <= sigma_ratio:
            sigma = random.uniform(sigma_range[0, 0], sigma_range[0, 1])
        else:
            sigma = random.uniform(sigma_range[1, 0], sigma_range[1, 1])


        # ensure all_xy is within the tile
        all_xy[:, 0] = np.clip(all_xy[:, 0], 0, tile_width - 1)
        all_xy[:, 1] = np.clip(all_xy[:, 1], 0, tile_height - 1)

        x_min, y_min = np.min(all_xy[:, 0]), np.min(all_xy[:, 1])
        x_max, y_max = np.max(all_xy[:, 0]), np.max(all_xy[:, 1])
        # compute the median for pixels not in the zero_mask
        try:
            median_bg = np.abs(np.median(synth_patch[y_min:y_max, x_min:x_max][zero_mask[y_min:y_max, x_min:x_max] == 0]))
        except:
            continue
        

        phi = (snr * median_bg) - median_bg
        # r_target = 1.25  # desired apparent half-width in pixels — tune this
        # T = 1
        # if snr > T:
        #     sigma = np.sqrt(r_target / (2 * np.log(snr / T)))
        #     sigma = np.clip(sigma, sigma_range[0, 0], sigma_range[0, 1])
        

        patch, min_xy, max_xy = patch_gen(all_xy, tile_width, tile_height, sigma=sigma, lc_width=lc_width)
        
        patch = patch / np.max(patch)
        patch = patch * phi # phi is determined by snr

        # composite the streak onto synth_patch — must happen unconditionally
        # so the image and its GT stay in sync
        synth_patch[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]] = np.clip(
            synth_patch[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]] + patch, 0, max_phi)

        p0 = all_xy[0, :]
        p1 = all_xy[-1, :]

        # convert to rho and theta
        try:
            rho, theta_deg = endpoints_to_rho_theta_mod(p0[0], p0[1], p1[0], p1[1], cx, cy)
        except Exception:
            # cannot compute RT coords — undo the composite to keep image/GT in sync
            synth_patch[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]] = np.clip(
                synth_patch[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]] - patch, 0, max_phi)
            continue

        # Skip lines whose rho falls outside the capped range —
        # they will not appear in the RT map, so they must not get labels.
        # Also undo the composite so the image stays clean.
        _rlo = rho_min_cap if rho_min_cap is not None else -(max_rho / 2)
        _rhi = rho_max_cap if rho_max_cap is not None else  (max_rho / 2)
        if not (_rlo <= rho <= _rhi):
            synth_patch[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]] = np.clip(
                synth_patch[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]] - patch, 0, max_phi)
            continue

        r_id, t_id = rho_theta_to_indices(rho, theta_deg, theta_min, itheta, _rlo, irho)

        # compute a bounding box for the rho_id and theta_id
        x1 = np.minimum(np.maximum((r_id - 10) / num_rhos, 0), 1)
        y1 = np.minimum(np.maximum((t_id - 10) / num_angles, 0), 1)
        x2 = np.minimum(np.maximum((r_id + 10) / num_rhos, 0), 1)
        y2 = np.minimum(np.maximum((t_id - 10) / num_angles, 0), 1)
        x3 = np.minimum(np.maximum((r_id + 10) / num_rhos, 0), 1)
        y3 = np.minimum(np.maximum((t_id + 10) / num_angles, 0), 1)
        x4 = np.minimum(np.maximum((r_id - 10) / num_rhos, 0), 1)
        y4 = np.minimum(np.maximum((t_id + 10) / num_angles, 0), 1)

        remaining_poly.append((x1, y1, x2, y2, x3, y3, x4, y4))
        remaining_lines.append((tuple(p0), tuple(p1)))
        len_stack.append(length)
        snr_stack.append(snr)
        sigma_stack.append(sigma)
    
    # zero out synth_patch with the zero_mask
    synth_patch = synth_patch * (1 - zero_mask)
    synth_patch = np.minimum(synth_patch, max_phi)

    # this step is important, we want to compute the rt_map in the original range, with zero-mean
    synth_patch_ori_range = synth_patch.copy()
    synth_patch_ori_range = synth_patch_ori_range + ori_min
    synth_patch_ori_range[zero_mask] = 0
    
    rt_map, _, _ = compute_rt_map(synth_patch_ori_range.astype(np.float32), max_rho/2,
                                  theta_res_deg=itheta,
                                  rho_res=irho,
                                  rho_min_cap=rho_min_cap,
                                  rho_max_cap=rho_max_cap)
    
    rt_map = rt_map - rt_map.min()
    rt_map = rt_map / rt_map.max() * 255
    rt_map = rt_map.astype(np.uint8)

    # visualise remaining_poly on rt_map for debugging
    if debug:
        for poly in remaining_poly:
            pts = np.array([[poly[0] * num_rhos, poly[1] * num_angles],
                            [poly[2] * num_rhos, poly[3] * num_angles],
                            [poly[4] * num_rhos, poly[5] * num_angles],
                            [poly[6] * num_rhos, poly[7] * num_angles]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(rt_map, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    synth_patch = synth_patch / np.max(synth_patch) * max_phi
    synth_patch = synth_patch.astype(np.uint8)

    if debug:
        for line in remaining_lines:
            pt0 = (int(round(line[0][0])), int(round(line[0][1])))
            pt1 = (int(round(line[1][0])), int(round(line[1][1])))
            cv2.line(synth_patch, pt0, pt1, color=255, thickness=1)

    return synth_patch, synth_patch_ori_range, rt_map, remaining_lines, remaining_poly, len_stack, snr_stack, sigma_stack

def _make_line_mask(line_txt_path, img_w, img_h, line_thickness=1):
    """
    Rasterise line-label txt (format: x0 y0 x1 y1 W H per row) into a
    binary {0,1} uint8 mask of size (img_h, img_w).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    with open(line_txt_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) != 6:
                continue
            x0, y0, x1, y1 = map(float, parts[:4])
            pt0 = (int(round(x0)), int(round(y0)))
            pt1 = (int(round(x1)), int(round(y1)))
            cv2.line(mask, pt0, pt1, color=255, thickness=line_thickness, lineType=cv2.LINE_AA)
    return (mask > 0).astype(np.uint8)


def _make_poly_mask(poly_txt_path, img_w, img_h):
    """
    Rasterise poly-label txt (YOLO format: class_id x0 y0 x1 y1 … per row,
    coords normalised [0,1]) into a binary {0,1} uint8 mask of size (img_h, img_w).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    with open(poly_txt_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 3:
                continue
            coords = list(map(float, parts[1:]))  # skip class_id
            if len(coords) % 2 != 0:
                continue
            xs = np.array(coords[0::2])
            ys = np.array(coords[1::2])
            px = np.clip(np.round(xs * (img_w - 1)), 0, img_w - 1).astype(np.int32)
            py = np.clip(np.round(ys * (img_h - 1)), 0, img_h - 1).astype(np.int32)
            pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color=1)
    return (mask > 0).astype(np.uint8) * 255


def write_data_to_file(txt_dir, img_name, lengths_array, sigmas_array, snrs_array):
    """
    Writes the provided data to a text file in the specified format.

    :param filename: The name of the file to write to.
    :param img_name: The image file name.
    :param lengths_array: An array of lengths with 3 elements.
    :param sigmas_array: An array of sigmas with 3 elements.
    :param snrs_array: An array of SNRs with 3 elements.
    """
    try:
        with open(txt_dir, 'a') as file:
            file.write(f"img_name: '{img_name}', lengths: {lengths_array[0]}, {lengths_array[1]}, {lengths_array[2]}, "
                       f"sigmas: {sigmas_array[0]}, {sigmas_array[1]}, {sigmas_array[2]}, "
                       f"snrs: {snrs_array[0]}, {snrs_array[1]}, {snrs_array[2]}\n")
    except Exception as e:
        print(f"An error occurred: {e}")


def paste_synth_streak_on_real_bg(filenames, out_img_dir, out_rt_dir,
                                  out_line_mask_dir, out_poly_mask_dir,
                                  snr_range, sigma_range, length_range,
                                  lc_width, stats_dir,
                                  snr_ratio, sigma_ratio, length_ratio_1, length_ratio_2,
                                  max_num_streak, scale_flag,
                                  num_angles, num_rhos,
                                  rho_min_cap=None, rho_max_cap=None, debug=False,
                                  line_mask_thickness=1):
    """
    Process each image by tiling and adjusting annotations.
    """

    for filename in filenames:
        max_phi = 255

        # the name is formatted as imgName_x(tileNum)_y(tileNum).png
        # we want to extract the corresponding image in filtered_paths with imgName.png
        # And then crop the tile based on the x and y tileNum with a fixed tile size
        # img_name = filename.split('/')[-1].split('.')[0].split('_')
        # img_name = str.join('_', img_name[0:-2])

        # x0 = filename.split('/')[-1].split('.')[0].split('_')[-2]
        # x0 = int(x0[1:])  # Convert to integer

        # y0 = filename.split('/')[-1].split('.')[0].split('_')[-1]
        # # extract the number after 'y'
        # y0 = int(y0[1:])  # Convert to integer

        
        # if img_name in filtered_paths:
        #     curr_img_path = filtered_paths[img_name]
        # else:
        #     continue

        curr_img_path = filename

        lengths_array = np.zeros([3])
        sigmas_array = np.zeros([3])
        snrs_array = np.zeros([3])

        try:
            if os.path.exists(curr_img_path):
                img = np.load(curr_img_path)
                img = img[:, :, 0]
            else:
                continue
        except Exception as e:
            continue

        filename = curr_img_path.split('/')
        stem = filename[-1].split('.')[0]  # already contains _xN_yN from the source filename
        
        
        zero_mask = img == 0
        itheta = 180 / num_angles
        max_rho = np.hypot(img.shape[0], img.shape[1]) + 1

        # When a rho cap is active, spread num_rhos bins over the capped range
        # instead of the full diagonal, keeping irho consistent with _make_params.
        if rho_min_cap is not None and rho_max_cap is not None:
            _effective_half_rho = (rho_max_cap - rho_min_cap) / 2.0
        elif rho_max_cap is not None:
            _effective_half_rho = rho_max_cap
        elif rho_min_cap is not None:
            _effective_half_rho = abs(rho_min_cap)
        else:
            _effective_half_rho = max_rho / 2.0
        irho = (2.0 * _effective_half_rho) / (num_rhos - 1)

        # check if it's dark
        # use deepcopy here to avoid modifying the original image
        tmp_img = copy.deepcopy(img)
        tmp_img = tmp_img - tmp_img.min()
        tmp_img[zero_mask] = 0

        tmp_img = tmp_img / tmp_img.max() * 255
        if np.mean(tmp_img) <= 5:
            continue
        
        streak_exists = random.random()

        # print(streak_exists)
        if streak_exists >= 0.5:  # 50% chance of having no streaks
            rt_map, theta_deg, rhos = compute_rt_map(img.astype(np.float32), max_rho/2,
                                                      theta_res_deg=itheta, rho_res=irho,
                                                      rho_min_cap=rho_min_cap,
                                                      rho_max_cap=rho_max_cap)
            
            rt_map = rt_map - rt_map.min()
            rt_map = rt_map / rt_map.max() * 255
            rt_map = rt_map.astype(np.uint8)

            # unscaled_img = copy.deepcopy(img)
            img_min = img.min()
            img = img - img_min
            # set the masked areas back to 0
            img[zero_mask] = 0
            
            img_max = img.max()
            img = img / img_max * 255
            img = img.astype(np.uint8)
            cv2.imwrite(f"{out_img_dir}{stem}.png", np.array(img).astype(np.uint8))
            cv2.imwrite(f"{out_rt_dir}{stem}.png", rt_map)

            # generate empty binary masks (no streak)
            h_img, w_img = img.shape[:2]
            h_rt, w_rt = rt_map.shape[:2]
            line_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            poly_mask = np.zeros((h_rt, w_rt), dtype=np.uint8)
            cv2.imwrite(f"{out_line_mask_dir}{stem}.png", line_mask)
            cv2.imwrite(f"{out_poly_mask_dir}{stem}.png", poly_mask)

            write_data_to_file(stats_dir, stem, lengths_array, sigmas_array, snrs_array)
            continue

        # scale img here
        img_min = img.min()
        img = img - img_min
        img[zero_mask] = 0

        height, width = img.shape
        # width, height = img.size
        lines = generate_and_trim_lines(width, height, 100, length_range, length_ratio_1, length_ratio_2, max_num_streak, length_min=args.length_min_1)

        synth_patch, _, rt_map, lines, polys, lengths, snrs, sigmas = lines_to_synth_patch(np.array(img), lines, snr_range, sigma_range, lc_width, max_phi,
                                                  width, height, snr_ratio, sigma_ratio, scale_flag, zero_mask, img_min,
                                                  num_angles, num_rhos, max_rho, irho, itheta,
                                                  rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap, debug=debug)
        
        if np.mean(synth_patch) <= 5: # ignore dark patch
            continue

        cv2.imwrite(f"{out_img_dir}{stem}.png", synth_patch.astype(np.uint8))
        cv2.imwrite(f"{out_rt_dir}{stem}.png", rt_map.astype(np.uint8))

        # generate binary masks directly from lines/polys (no txt files needed)
        h_img, w_img = synth_patch.shape[:2]
        h_rt, w_rt = rt_map.shape[:2]

        line_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for line in lines:
            pt0 = (int(round(line[0][0])), int(round(line[0][1])))
            pt1 = (int(round(line[1][0])), int(round(line[1][1])))
            cv2.line(line_mask, pt0, pt1, color=255, thickness=line_mask_thickness, lineType=cv2.LINE_AA)
        line_mask = (line_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(f"{out_line_mask_dir}{stem}.png", line_mask)

        poly_mask = np.zeros((h_rt, w_rt), dtype=np.uint8)
        for poly in polys:
            xs = np.array(poly[0::2])
            ys = np.array(poly[1::2])
            px = np.clip(np.round(xs * (w_rt - 1)), 0, w_rt - 1).astype(np.int32)
            py = np.clip(np.round(ys * (h_rt - 1)), 0, h_rt - 1).astype(np.int32)
            pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
            cv2.fillPoly(poly_mask, [pts], color=1)
        poly_mask = (poly_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(f"{out_poly_mask_dir}{stem}.png", poly_mask)
        

        # save with the way i've been saving in visualise_labels_on_raw.pyif image_channel == 3:
        # for visualising purposes
        # plt.imshow(synth_patch, vmin=synth_patch.min(), vmax=synth_patch.max())
        # plt.axis('off')
        # plt.savefig(f"{out_img_dir}{filename[-1].split('.')[0]}_x{x0}_y{y0}.png", bbox_inches='tight', pad_inches=0)
        # plt.close()

        # write stats here
        for i in range(len(lengths)):
            lengths_array[i] = lengths[i]
            sigmas_array[i] = sigmas[i]
            snrs_array[i] = snrs[i]

        write_data_to_file(stats_dir, stem, lengths_array, sigmas_array, snrs_array)
        # print('ck')



def bresenham(x1, y1, x2, y2):
    x1, x2 = round(x1), round(x2)
    y1, y2 = round(y1), round(y2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = abs(dy) > abs(dx)

    if steep:
        t = dx
        dx = dy
        dy = t

    if dy == 0:
        q = np.zeros(dx + 1)
    else:
        A = np.floor(dx / 2)
        B = np.arange(A, -dy * dx + A, -dy)
        C = B.reshape(-1, 1)
        D = np.mod(C, dx)
        E = np.diff(D, axis=0)
        F = E >= 0
        q = np.vstack((0, F))
        # q = np.append([0], np.diff(np.mod(np.floor(dx / 2) - np.arange(dy, -dy * dx + np.floor(dx / 2), -dy), dx)) >= 0)

    if steep:
        if y1 <= y2:
            y = np.arange(y1, y2)
        else:
            y = np.arange(y1, y2, -1)
        if x1 <= x2:
            x = x1 + np.cumsum(q)
        else:
            x = x1 - np.cumsum(q)
    else:
        if x1 <= x2:
            x = np.arange(x1, x2)
        else:
            x = np.arange(x1, x2, -1)
        if y1 <= y2:
            y = y1 + np.cumsum(q)
        else:
            y = y1 - np.cumsum(q)

    x, y = equalize_arrays(x, y)

    return np.int64(np.array([x, y]).T)


def equalize_arrays(arr1, arr2):
    # Check the lengths of the arrays
    len_arr1 = len(arr1)
    len_arr2 = len(arr2)

    # If arr1 is longer, trim it to match the length of arr2
    if len_arr1 > len_arr2:
        arr1 = arr1[:len_arr2]

    # If arr2 is longer, trim it to match the length of arr1
    elif len_arr2 > len_arr1:
        arr2 = arr2[:len_arr1]

    return arr1, arr2


@njit
def _gaussian_streak_kernel(all_x, all_y, min_y, max_y, min_x, max_x, patch,
                             sigma_sqr, sigma_sqrt_pi, step, patch_min_y, patch_min_x,
                             alpha=0.001):
    """
    Numba-JIT kernel: accumulates Gaussian intensity contributions from each
    streak pixel (all_x[i], all_y[i]) into the output patch.
    """
    for x in range(patch.shape[1]):
        for y in range(patch.shape[0]):
            y_id = y + patch_min_y
            x_id = x + patch_min_x
            for i in range(step):
                if min_y[i] <= y_id < max_y[i] and min_x[i] <= x_id < max_x[i]:
                    dx = x_id - all_x[i]
                    dy = y_id - all_y[i]
                    dist = (dx * dx + dy * dy) ** 0.5
                    patch[y, x] += alpha / sigma_sqrt_pi * math.exp(-0.5 * dist / sigma_sqr)
    return patch


def patch_gen(xy, tile_width, tile_height, sigma, lc_width=10):
    """
    Generate a Gaussian-blurred intensity patch for a streak defined by pixel
    coordinates xy (shape N×2, columns x and y).

    Returns
    -------
    patch   : 2-D float64 array  — intensity patch over the streak bounding box
    min_xy  : (2,) int array     — (x_min, y_min) of the patch in tile coords
    max_xy  : (2,) int array     — (x_max, y_max) of the patch in tile coords
    """
    all_x = xy[:, 0]
    all_y = xy[:, 1]

    # Vectorised AOE (area-of-effect) bounds per streak pixel
    min_y = np.maximum(0,           np.round(all_y - lc_width)).astype(np.float64)
    max_y = np.minimum(tile_height, np.round(all_y + lc_width)).astype(np.float64)
    min_x = np.maximum(0,           np.round(all_x - lc_width)).astype(np.float64)
    max_x = np.minimum(tile_width,  np.round(all_x + lc_width)).astype(np.float64)

    patch_min_y = int(min_y.min())
    patch_max_y = int(max_y.max())
    patch_min_x = int(min_x.min())
    patch_max_x = int(max_x.max())

    patch = np.zeros((patch_max_y - patch_min_y, patch_max_x - patch_min_x))

    sigma_sqr     = sigma * sigma
    sigma_sqrt_pi = sigma * np.sqrt(2 * np.pi)

    patch = _gaussian_streak_kernel(
        all_x, all_y, min_y, max_y, min_x, max_x, patch,
        sigma_sqr, sigma_sqrt_pi, len(min_y), patch_min_y, patch_min_x
    )

    return patch, np.array([patch_min_x, patch_min_y]), np.array([patch_max_x, patch_max_y])



def on_segment(p, q, r):
    """Check if point q lies on line segment 'pr'"""
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def orientation(p, q, r):
    """Find orientation of ordered triplet (p, q, r).
    Returns 0 if p, q and r are colinear, 1 if clockwise, 2 if counterclockwise"""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # colinear
    return 1 if val > 0 else 2  # clock or counterclock wise

def do_lines_intersect(line1, line2):
    """Check if two line segments intersect"""
    p1 = (line1[0], line1[1])
    q1 = (line1[2], line1[3])
    p2 = (line2[0], line2[1])
    q2 = (line2[2], line2[3])

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False


def remove_intersecting_lines(lines):
    """
    Remove lines that intersect with others from a list of lines.
    """
    if not lines:
        return []

    # Start with the first line in the non_intersecting_lines
    non_intersecting_lines = [lines[0]]

    # Iterate over the remaining lines
    for line in lines[1:]:
        if not any(do_lines_intersect(line[0:4], existing_line[0:4]) for existing_line in non_intersecting_lines):
            non_intersecting_lines.append(line)

    return non_intersecting_lines

def clip_line_to_image(p0, p1, image_shape):
    """
    Clip a line segment to the image rectangle [0, W-1] x [0, H-1].

    Parameters
    ----------
    p0 : tuple (x0, y0)
        Start point of the line (can be float).
    p1 : tuple (x1, y1)
        End point of the line (can be float).
    image_shape : tuple (H, W)
        Image shape.

    Returns
    -------
    (q0, q1) or None
        q0, q1 are the clipped endpoints as (x, y) floats.
        Returns None if the line is completely outside the image.
    """
    H, W = image_shape
    x0, y0 = p0
    x1, y1 = p1

    dx = x1 - x0
    dy = y1 - y0

    t0, t1 = 0.0, 1.0  # parametric interval

    def _clip(p, q, t0, t1):
        if p == 0:
            # Line is parallel to this boundary
            if q < 0:
                return None  # completely outside
            else:
                return t0, t1  # no change
        r = q / p
        if p < 0:
            # entering boundary
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            # leaving boundary
            if r < t0:
                return None
            if r < t1:
                t1 = r
        return t0, t1

    # Left: x >= 0  →  -dx * t <= x0
    out = _clip(-dx, x0 - 0, t0, t1)
    if out is None: return None
    t0, t1 = out

    # Right: x <= W-1  →  dx * t <= (W-1) - x0
    out = _clip(dx, (W - 1) - x0, t0, t1)
    if out is None: return None
    t0, t1 = out

    # Top: y >= 0
    out = _clip(-dy, y0 - 0, t0, t1)
    if out is None: return None
    t0, t1 = out

    # Bottom: y <= H-1
    out = _clip(dy, (H - 1) - y0, t0, t1)
    if out is None: return None
    t0, t1 = out

    # Compute clipped endpoints
    q0 = (x0 + t0 * dx, y0 + t0 * dy)
    q1 = (x0 + t1 * dx, y0 + t1 * dy)
    return q0, q1


import argparse

# =========================
USE_ARGPARSE = True  # set True to use CLI; False to use CONFIG below

CONFIG = {
    # IO
    "img_dir": "/home/ckchng/Documents/SDA_ODA/LMA_data/background_patches_with_new_model",              # directory containing source images
    "output_dir": "/home/ckchng/Documents/SDA_ODA/LMA_data/tmp/",            # base output directory
    # Range selection
    "starting_id": 100,
    "step": 100,                                # number of files to take starting from starting_id
    "num_angles": 192,                            # number of angles for Hough Transform
    #"num_rhos": 416,                              # number of rho values for Hough Transform
    "num_rhos": 288,
    "rho_min_cap": -144,
    "rho_max_cap": 143,
    #"rho_min_cap": None,
    #"rho_max_cap": None,
    # Hyperparameters
    "snr_min_1": 1.32, "snr_max_1": 1.35,
    "snr_min_2": 1.25, "snr_max_2": 2.0,

    "sigma_min_1": 0.75, "sigma_max_1": 1.3,
    "sigma_min_2": 1.25, "sigma_max_2": 2.0,

    "length_min_1": 200, "length_max_1": 400,
    "length_min_2": 50, "length_max_2": 600,
    "length_min_3": 601, "length_max_3": 1000,

    "snr_ratio": 1.0,          # probability to sample from range 1 vs range 2
    "sigma_ratio": 1.0,
    "length_ratio_1": 1.0,     # choose range1 if r <= ratio1; else if r <= ratio2 choose range2; else range3
    "length_ratio_2": 0.66,

    "lc_width": 3,
    "max_num_streak": 1,
    "scale_flag": 1,            # 1 = normalize to 0-255 before compositing
    # Set to a float to clamp the rho axis and ignore border lines, or None for full range
    "debug": False,             # True = draw label polys on RT map and lines on synth patch
    "line_mask_thickness": 5,   # cv2.line thickness when rasterising line masks
}

def _parse_args_or_config():
    if not USE_ARGPARSE:
        class _A: pass
        a = _A()
        for k, v in CONFIG.items():
            setattr(a, k, v)
        return a


    parser = argparse.ArgumentParser(description="Script to process data.")
    parser.add_argument("--img_dir", required=True, help="Directory path")
    parser.add_argument("--output_dir", required=True, help="Directory path")
    
    parser.add_argument("--starting_id", type=int, required=True, help="Testing path")
    parser.add_argument("--num_angles", type=int, required=True, help="Testing path")
    parser.add_argument("--num_rhos", type=int, required=True, help="Testing path")
    parser.add_argument("--step", type=int, required=True, help="Testing path")
    parser.add_argument("--snr_min_1", type=float, required=True, help="Testing path")
    parser.add_argument("--snr_max_1", type=float, required=True, help="Testing path")
    parser.add_argument("--snr_min_2", type=float, required=True, help="Testing path")
    parser.add_argument("--snr_max_2", type=float, required=True, help="Testing path")
    parser.add_argument("--sigma_min_1", type=float, required=True, help="Testing path")
    parser.add_argument("--sigma_max_1", type=float, required=True, help="Testing path")
    parser.add_argument("--sigma_min_2", type=float, required=True, help="Testing path")
    parser.add_argument("--sigma_max_2", type=float, required=True, help="Testing path")
    parser.add_argument("--length_min_1", type=float, required=True, help="Testing path")
    parser.add_argument("--length_max_1", type=float, required=True, help="Testing path")
    parser.add_argument("--length_min_2", type=float, required=True, help="Testing path")
    parser.add_argument("--length_max_2", type=float, required=True, help="Testing path")
    parser.add_argument("--length_min_3", type=float, required=True, help="Testing path")
    parser.add_argument("--length_max_3", type=float, required=True, help="Testing path")

    parser.add_argument("--snr_ratio", type=float, required=True, help="Testing path")
    parser.add_argument("--sigma_ratio", type=float, required=True, help="Testing path")
    parser.add_argument("--length_ratio_1", type=float, required=True, help="Testing path")
    parser.add_argument("--length_ratio_2", type=float, required=True, help="Testing path")
    parser.add_argument("--lc_width", type=int, required=True, help="Testing path")

    parser.add_argument("--max_num_streak", type=int, required=True, help="Testing path")
    parser.add_argument("--scale_flag", type=int, required=True, help="Testing path")
    parser.add_argument("--rho_min_cap", type=float, default=None, help="Lower rho cap (None = full range)")
    parser.add_argument("--rho_max_cap", type=float, default=None, help="Upper rho cap (None = full range)")
    parser.add_argument("--debug", action="store_true", default=False, help="Visualise label polys on RT map and lines on synth patch")
    parser.add_argument("--line_mask_thickness", type=int, default=1, help="Line thickness when rasterising line masks")

    args = parser.parse_args()
    return parser.parse_args()

# ...existing code...
if __name__ == "__main__":
    # parser/config
    args = _parse_args_or_config()
    
    output_dir = args.output_dir
    
    starting_id = args.starting_id
    img_dir = args.img_dir
    step = args.step
    ending_id = starting_id + step
    snr_min_1 = args.snr_min_1
    snr_max_1 = args.snr_max_1
    snr_min_2 = args.snr_min_2
    snr_max_2 = args.snr_max_2

    sigma_min_1 = args.sigma_min_1
    sigma_max_1 = args.sigma_max_1
    sigma_min_2 = args.sigma_min_2
    sigma_max_2 = args.sigma_max_2

    length_min_1 = args.length_min_1
    length_max_1 = args.length_max_1
    length_min_2 = args.length_min_2
    length_max_2 = args.length_max_2
    length_min_3 = args.length_min_3
    length_max_3 = args.length_max_3

    snr_ratio = args.snr_ratio
    sigma_ratio = args.sigma_ratio
    length_ratio_1 = args.length_ratio_1
    length_ratio_2 = args.length_ratio_2

    max_num_streak = args.max_num_streak
    scale_flag = args.scale_flag
    num_angles = args.num_angles
    num_rhos = args.num_rhos
    lc_width = args.lc_width
    rho_min_cap = args.rho_min_cap
    rho_max_cap = args.rho_max_cap
    debug = args.debug
    line_mask_thickness = args.line_mask_thickness
    

    print(starting_id, ending_id)
    stats_out_dir = output_dir + '/'+str(starting_id)+'_stats.txt'
    output_img_dir = output_dir + '/actual_images/train/'
    output_rt_dir = output_dir + '/images/train/'
    output_line_mask_dir  = output_dir + '/line_masks/train/'
    output_poly_mask_dir  = output_dir + '/poly_masks/train/'

    if not os.path.exists(output_img_dir):
        # Create the directory if it doesn't exist
        os.makedirs(output_img_dir, exist_ok=True)
    
    if not os.path.exists(output_rt_dir):
        # Create the directory if it doesn't exist
        os.makedirs(output_rt_dir, exist_ok=True)

    os.makedirs(output_line_mask_dir, exist_ok=True)
    os.makedirs(output_poly_mask_dir, exist_ok=True)

    # image_dirs = read_file_lines(img_dir_txt_file)
    # get image_dirs with os.listdir from a given directory
    image_dirs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    snr_range = np.array([[snr_min_1, snr_max_1], [snr_min_2, snr_max_2]])
    sigma_range = np.array([[sigma_min_1, sigma_max_1], [sigma_min_2, sigma_max_2]])

    length_range = np.array([[length_min_1, length_max_1],
                            [length_min_2, length_max_2],
                            [length_min_3, length_max_3]])

    ending_id = np.minimum(len(image_dirs), ending_id)

    sub_image_dirs = image_dirs[starting_id:ending_id]

    # need another function to crop the filtered image version
    paste_synth_streak_on_real_bg(sub_image_dirs, output_img_dir, output_rt_dir,
                                  output_line_mask_dir, output_poly_mask_dir,
                                  snr_range, sigma_range, length_range, lc_width, stats_out_dir,
                                  snr_ratio, sigma_ratio, length_ratio_1, length_ratio_2,
                                  max_num_streak, scale_flag,
                                  num_angles, num_rhos,
                                  rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap, debug=debug,
                                  line_mask_thickness=line_mask_thickness)
    


