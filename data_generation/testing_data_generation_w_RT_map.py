from itertools import chain
import cv2
import numpy as np
import os
import pickle
import bz2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from numba import njit, prange
import random
import math

import rawpy, cv2

from ht_utils import (
    compute_rt_map,
    endpoints_to_rho_theta_mod,
    rho_theta_to_indices,
)

def load_nef_as_bgr_uint8(path: str,
                          use_camera_wb: bool = True,
                          output_bps: int = 8,
                          no_auto_bright: bool = True) -> np.ndarray:
    """
    Returns a demosaiced BGR image (uint8) suitable for OpenCV/YOLO.
    """
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=use_camera_wb,     # use in-camera WB (good default)
            no_auto_bright=no_auto_bright,   # keep exposure linear-ish
            output_bps=output_bps,           # 8 or 16
            gamma=(2.222, 4.5),              # sRGB-ish gamma
            output_color=rawpy.ColorSpace.sRGB,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # or DCB/LMMSE for quality
            bright=1.0
        )

    # rawpy gives RGB; convert to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if output_bps == 16:
        # If you chose 16-bit, you can downscale safely to 8-bit here for YOLO
        bgr = (bgr / 256).astype(np.uint8)
    return bgr

def normalize_bbox(bbox, tile_width, tile_height):
    """
    Normalize the bounding box coordinates based on the tile dimensions.
    """
    bx, by, bwidth, bheight = bbox
    x_center = bx + bwidth / 2
    y_center = by + bheight / 2
    return [x_center / tile_width, y_center / tile_height, bwidth / tile_width, bheight / tile_height]

def run_test_cases():
    bbox = [50, 100, 150, 150]  # Bounding box in XYWH format

    # Test Case 1: Line does not intersect or is inside the box
    line1 = [(10, 20), (30, 40)]
    print("Test Case 1:", check_line_and_bbox(line1, bbox))  # Expected: (False, False)

    # Test Case 2: Line is inside the box
    line2 = [(60, 110), (100, 150)]
    print("Test Case 2:", check_line_and_bbox(line2, bbox))  # Expected: (True, True)

    # Test Case 3: Only one point of the line is in the box
    line3 = [(40, 90), (120, 170)]
    print("Test Case 3:", check_line_and_bbox(line3, bbox))  # Expected: (True, False)

    # Test Case 4: Line intersects the box but both endpoints are outside the box
    line4 = [(40, 90), (300, 300)]
    print("Test Case 4:", check_line_and_bbox(line4, bbox))  # Expected: (True, False)


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

    # Adjust or clip the points based on whether they are inside the tile
    if not is_inside_tile(x1, y1):

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
        

def check_line_and_bbox(line, bbox):
    """
    Check various conditions for a line and a bounding box.
    Returns four booleans for four conditions.
    """
    # Check if each point is inside the bbox
    point1_inside = is_point_inside_bbox(line[0], bbox)
    point2_inside = is_point_inside_bbox(line[1], bbox)

    # Condition 1: Both points are inside the bbox
    fully_inside = point1_inside and point2_inside

    # Condition 2: Only point1 is inside the bbox
    only_point1_inside = point1_inside and not point2_inside

    # Condition 3: Only point2 is inside the bbox
    only_point2_inside = point2_inside and not point1_inside

    # Condition 4: Neither point is inside, but the line intersects the bbox
    intersects = does_line_intersect_bbox(line, bbox)
    neither_inside_but_intersects = not point1_inside and not point2_inside and intersects

    return fully_inside, only_point1_inside, only_point2_inside, neither_inside_but_intersects


def is_point_inside_bbox(point, bbox):
    """
    Check if a point (x, y) is inside a bounding box [x_min, y_min, width, height].
    """
    x, y = point
    x_min, y_min, width, height = bbox
    return x_min <= x <= x_min + width and y_min <= y <= y_min + height


def line_segment_intersection(line1, line2):
    """
    Check if two line segments intersect.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = line1
    C, D = line2

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def does_line_intersect_bbox(line, bbox):
    """
    Check if a line intersects a bounding box.
    """
    point1_inside = is_point_inside_bbox(line[0], bbox)
    point2_inside = is_point_inside_bbox(line[1], bbox)

    # If either point is inside the bbox, the line intersects
    if point1_inside or point2_inside:
        return True

    # Check if the line intersects any of the edges of the bbox
    x_min, y_min, width, height = bbox
    top_edge = [(x_min, y_min), (x_min + width, y_min)]
    bottom_edge = [(x_min, y_min + height), (x_min + width, y_min + height)]
    left_edge = [(x_min, y_min), (x_min, y_min + height)]
    right_edge = [(x_min + width, y_min), (x_min + width, y_min + height)]

    for edge in [top_edge, bottom_edge, left_edge, right_edge]:
        if line_segment_intersection(line, edge):
            # compute the intersection point
            return True

    return False

def line_to_bbox_with_buffer_and_clipping(line, buffer_size, img_width, img_height):
    """
    Convert a line to a bounding box with a buffer and clip it to the image boundaries.

    :param line: The line represented by two endpoints [(x1, y1), (x2, y2)].
    :param buffer_size: The buffer size to expand the bounding box.
    :param img_width: The width of the image.
    :param img_height: The height of the image.
    :return: The bounding box represented by [x_min, y_min, width, height].
    """
    x1, y1, x2, y2 = line

    # Determine the min and max coordinates with buffer
    x_min = max(min(x1, x2) - buffer_size, 0)
    y_min = max(min(y1, y2) - buffer_size, 0)
    x_max = min(max(x1, x2) + buffer_size, img_width)
    y_max = min(max(y1, y2) + buffer_size, img_height)

    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]


def visualize_box_and_line(img, box, line, box_color=(0, 255, 0), line_color=(255, 0, 0), thickness=2, point_radius=5):
    """
    Visualize a bounding box and a line on an image.

    :param img: The image on which to draw (numpy array).
    :param box: The bounding box as [x, y, width, height].
    :param line: The line as [(x1, y1), (x2, y2)].
    :param box_color: Color of the bounding box (BGR).
    :param line_color: Color of the line (BGR).
    :param thickness: Thickness of the lines.
    """
    # Draw the bounding box
    x, y, w, h = box

    top_left = (x, y)
    bottom_right = (x + w, y + h)
    cv2.rectangle(img, top_left, bottom_right, box_color, thickness)

    # Draw the line
    # cv2.line(img, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), line_color, thickness)
    cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), line_color, thickness)

    point1_color = (255, 0, 0)
    point2_color = (255, 0, 255)
    # Draw the endpoints
    # cv2.circle(img, (int(line[0][0]), int(line[0][1])), point_radius, point1_color, -1)  # Filled circle for point1
    # cv2.circle(img, (int(line[1][0]), int(line[1][1])), point_radius, point2_color, -1)  # Filled circle for point2

    cv2.circle(img, (int(line[0]), int(line[1])), point_radius, point1_color, -1)  # Filled circle for point1
    cv2.circle(img, (int(line[2]), int(line[3])), point_radius, point2_color, -1)  # Filled circle for point2

    pass

# def filename_to_img_nef(filename, img_path, img_name):
#     nef_image = load_nef_as_bgr_uint8(img_path)

#     #check if filename exists
#     # img = None
#     # if os.path.exists(filename):
#     #     if '.pkl' in filename:
#     #         with open(filename, 'rb') as f:
#     #             img = pickle.load(f)
#     #     elif '.bz2' in filename:
#     #         # Open the .bz2 file
#     #         with bz2.open(filename, 'rb') as f:
#     #             # Load the .pkl file
#     #             img = pickle.load(f)
#     #     else:
#     #         # for filename in os.listdir(img_dir):
#     #         with fits.open(filename) as hdu_list:
#     #             ccd = CCDData(hdu_list[0].data, unit='adu', header=hdu_list[0].header)
#     #             img = ccd.data

#     # else:
#     #     tmp = img_name.split('.')
#     #     if os.path.exists(img_path + tmp[0] + '.pkl'):
#     #         with open(img_path + tmp[0] + '.pkl', 'rb') as f:
#     #             img = pickle.load(f)
#     #     elif os.path.exists(img_path + tmp[0] + '.bz2'):
#     #         with bz2.open(img_path + tmp[0] + '.bz2', 'rb') as f:
#     #             # Load the .pkl file
#     #             img = pickle.load(f)

#     return img

def line_medians(image, p0, p1, width):
    """
    Compute the median intensity along a given line and all its parallel
    lines within a given perpendicular width on both sides.

    Parameters
    ----------
    image : (H, W) ndarray
        Grayscale image.
    p0 : tuple (x0, y0)
        First endpoint of the main line (in pixel coordinates).
    p1 : tuple (x1, y1)
        Second endpoint of the main line (in pixel coordinates).
    width : int
        Maximum perpendicular offset in pixels. All offsets from -width
        to +width (inclusive) are used, so there are 2*width+1 lines.

    Returns
    -------
    medians : (2*width+1,) ndarray
        Median intensity for each parallel line.
        medians[i] corresponds to offset offsets[i].
    offsets : (2*width+1,) ndarray
        Integer perpendicular offsets in pixels, ranging from -width to +width.
    """
    H, W = image.shape

    x0, y0 = p0
    x1, y1 = p1

    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)

    if length == 0:
        raise ValueError("Endpoints must define a non-zero length line.")

    # Unit direction vector along the line
    ux = dx / length
    uy = dy / length

    # Unit normal vector (perpendicular to the line)
    nx = -uy
    ny = ux

    # Sample points along the central line
    num_samples = int(np.ceil(length)) + 1
    t = np.linspace(0.0, 1.0, num_samples)
    base_x = x0 + t * dx
    base_y = y0 + t * dy

    medians = []
    offsets = []

    for k in range(-width, width + 1):
        # Offset the line by k pixels along the normal
        line_x = base_x + k * nx
        line_y = base_y + k * ny

        # Convert to integer pixel indices
        ix = np.round(line_x).astype(int)
        iy = np.round(line_y).astype(int)

        # Keep points inside the image
        mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix_valid = ix[mask]
        iy_valid = iy[mask]

        if ix_valid.size == 0:
            med = np.nan  # or 0, or skip; up to your use case
        else:
            values = image[iy_valid, ix_valid]
            med = float(np.median(values))

        medians.append(med)
        offsets.append(k)

    return np.array(medians), np.array(offsets)

def parallel_line(p0, p1, offset):
    """
    p0, p1: (x, y) endpoints of the original line
    offset: perpendicular distance in pixels (positive to one side, negative to the other)
    """
    x0, y0 = p0
    x1, y1 = p1

    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)
    if length == 0:
        raise ValueError("Zero-length line")

    # unit normal (perpendicular to the line)
    nx = -dy / length
    ny =  dx / length

    # shift both endpoints by offset * normal
    p0_off = (x0 + offset * nx, y0 + offset * ny)
    p1_off = (x1 + offset * nx, y1 + offset * ny)
    return p0_off, p1_off

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


# ---------- core (numba) ----------
@njit(parallel=True, fastmath=True)
def _hough_accumulate_intensity(img, cos_t, sin_t, rhos, cx, cy, rho_res):
    h, w = img.shape
    T = cos_t.shape[0]
    R = rhos.shape[0]
    H = np.zeros((T, R), dtype=np.float32)
    r0 = rhos[0] / rho_res
    # Parallelize over angles so each thread writes to its own column -> no races.
    for t_idx in prange(T):
        c = cos_t[t_idx]
        s = sin_t[t_idx]
        
        for y in range(h):
            Y = y - cy
            for x in range(w):
                v = img[y, x]
                if v == 0:
                    continue
                X = x - cx
                rho = X * c + Y * s
                # r_idx = int((rho - r0) / rho_res + 0.5)   # round
                r_idx = int((round(rho / rho_res)) - r0)
                if 0 <= r_idx < R:
                    H[t_idx, r_idx] += v
    return H

@njit(fastmath=True)
def _compute_t_bounds(h, w, cx, cy, rho, c, s, seg_res):
    """
    Compute the valid t-range [t_min, t_max] for the line X*c + Y*s = rho,
    clipped to the image rectangle, using centered coords (X=x-cx, Y=y-cy).
    Returns (t_min, t_max). If no intersection, t_min >= t_max.
    """
    # Centered coordinate bounds
    Xmin = -cx
    Xmax = (w - 1.0) - cx
    Ymin = -cy
    Ymax = (h - 1.0) - cy

    eps = 1e-12
    # Start with the whole real line for t, then intersect with constraints
    t_lo = -1e18
    t_hi =  1e18

    # Constraint: X = c*rho - s*t in [Xmin, Xmax]
    if abs(s) > eps:
        lo_x = (c * rho - Xmax) / s
        hi_x = (c * rho - Xmin) / s
        if lo_x > hi_x:
            tmp = lo_x
            lo_x = hi_x
            hi_x = tmp
        if lo_x > t_lo:
            t_lo = lo_x
        if hi_x < t_hi:
            t_hi = hi_x
    else:
        # s ~ 0 => X ≈ c*rho is constant; must lie within [Xmin, Xmax]
        Xc = c * rho
        if Xc < Xmin - 1e-9 or Xc > Xmax + 1e-9:
            return 0.0, -1.0  # no intersection

    # Constraint: Y = s*rho + c*t in [Ymin, Ymax]
    if abs(c) > eps:
        lo_y = (Ymin - s * rho) / c
        hi_y = (Ymax - s * rho) / c
        if lo_y > hi_y:
            tmp = lo_y
            lo_y = hi_y
            hi_y = tmp
        if lo_y > t_lo:
            t_lo = lo_y
        if hi_y < t_hi:
            t_hi = hi_y
    else:
        # c ~ 0 => Y ≈ s*rho is constant; must lie within [Ymin, Ymax]
        Yc = s * rho
        if Yc < Ymin - 1e-9 or Yc > Ymax + 1e-9:
            return 0.0, -1.0  # no intersection

    # Add a small margin so rounding to bins includes edge pixels
    margin = 0.5 * seg_res
    return t_lo - margin, t_hi + margin

@njit(fastmath=True)
def _hough_segment_accumulate_intensity(img, rho, theta, starts, lengths, cx, cy, rho_res, segment_res):
    h, w = img.shape
    seg_res = segment_res if segment_res > 0.0 else 1.0

    c = np.cos(theta)
    s = np.sin(theta)

    # Compute t-range specifically for this (rho, theta) by clipping to the image
    t_min, t_max = _compute_t_bounds(h, w, cx, cy, rho, c, s, seg_res)
    if not (t_max > t_min):
        # No intersection of the line with the image => empty results
        return (np.zeros((starts.shape[0], lengths.shape[0]), dtype=np.float64),
                np.zeros((starts.shape[0], lengths.shape[0]), dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64))

    # Discretize t-axis for accumulation between [t_min, t_max]
    t0 = t_min
    span = t_max - t_min
    num_bins = int(np.floor(span / seg_res)) + 1
    if num_bins < 1:
        num_bins = 1

    profile = np.zeros(num_bins, dtype=np.float64)       # sum of intensities
    count_profile = np.zeros(num_bins, dtype=np.float64) # count of contributing pixels

    tol = 0.5 * rho_res

    # Accumulate pixels near the line (within rho tolerance) onto t-bins
    for y in range(h):
        Y = y - cy
        for x in range(w):
            v = img[y, x]
            if v == 0.0:
                continue
            X = x - cx
            rho_px = X * c + Y * s
            if abs(rho_px - rho) > tol:
                continue
            t = -X * s + Y * c
            idx = int(np.round((t - t0) / seg_res))
            if 0 <= idx < num_bins:
                profile[idx] += v
                count_profile[idx] += 1.0

    # Prefix sums (inclusive scan)
    prefix = np.zeros(num_bins + 1, dtype=np.float64)
    count_prefix = np.zeros(num_bins + 1, dtype=np.float64)
    for i in range(num_bins):
        prefix[i + 1] = prefix[i] + profile[i]
        count_prefix[i + 1] = count_prefix[i] + count_profile[i]

    num_starts = starts.shape[0]
    num_lengths = lengths.shape[0]
    votes = np.zeros((num_starts, num_lengths), dtype=np.float64)
    counts = np.zeros((num_starts, num_lengths), dtype=np.float64)

    # Only accept segments that fall within [t_min, t_max]
    for si in range(num_starts):
        start = starts[si]
        if (start < t_min) or (start >= t_max):
            continue
        start_idx = int(np.round((start - t0) / seg_res))
        if start_idx < 0 or start_idx >= num_bins:
            continue
        for li in range(num_lengths):
            seg_len = lengths[li]
            if seg_len <= 0.0:
                continue
            end_t = start + seg_len
            if end_t > t_max:
                continue
            seg_bins = int(np.round(seg_len / seg_res))
            if seg_bins <= 0:
                seg_bins = 1
            end_idx = start_idx + seg_bins
            if end_idx > num_bins:
                continue
            sum_int = prefix[end_idx] - prefix[start_idx]
            sum_cnt = count_prefix[end_idx] - count_prefix[start_idx]
            votes[si, li] = sum_int
            counts[si, li] = sum_cnt  # may be zero

    # Physical t-axis (for diagnostics/plotting)
    t_axis = np.zeros(num_bins, dtype=np.float64)
    for i in range(num_bins):
        t_axis[i] = t0 + i * seg_res

    return votes, counts, profile, count_profile, t_axis


# ---------- tiny helpers ----------
def _make_params(max_rho, theta_res_deg=1.0, rho_res=1.0):

    thetas_deg = np.arange(0, 180, theta_res_deg, dtype=np.float64)
    thetas = np.deg2rad(thetas_deg)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    rhos = np.arange(-max_rho, max_rho + rho_res, rho_res, dtype=np.float64)
    return thetas_deg, thetas, cos_t, sin_t, rhos

def hough_bruteforce_intensity_numba(img, max_rho, theta_res_deg=1.0, rho_res=1.0):
    h, w = img.shape
    cy = h / 2.0
    cx = w / 2.0
    
    theta_deg, theta_rad, cos_t, sin_t, rhos = _make_params(max_rho, theta_res_deg, rho_res)

    # Ensure numeric type Numba likes
    img32 = img.astype(np.float32, copy=False)
    H = _hough_accumulate_intensity(img32, cos_t, sin_t, rhos, cx, cy, rho_res)
    return H, theta_deg, rhos

def hough_line_segment_votes(img, rho, theta, starts, lengths, rho_res=1.0, segment_res=1.0, normalize=True):
    h, w = img.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    img64 = np.ascontiguousarray(img.astype(np.float64, copy=False))
    starts64 = np.ascontiguousarray(np.asarray(starts, dtype=np.float64))
    lengths64 = np.ascontiguousarray(np.asarray(lengths, dtype=np.float64))

    votes, counts, profile, count_profile, t_axis = _hough_segment_accumulate_intensity(
        img64, rho, theta, starts64, lengths64, cx, cy, rho_res, segment_res
    )
    if normalize:
        # Avoid divide-by-zero: only divide where counts>0
        norm_votes = np.zeros_like(votes)
        for i in range(votes.shape[0]):
            for j in range(votes.shape[1]):
                cval = counts[i, j]
                if cval > 0.0:
                    norm_votes[i, j] = votes[i, j] / cval
        return norm_votes, votes, counts, profile, count_profile, t_axis
    else:
        return votes, votes, counts, profile, count_profile, t_axis

def endpoints_to_rho_theta(x1, y1, x2, y2, cx, cy):
    """
    (x1, y1), (x2, y2) are image coords (0..w-1, 0..h-1)
    Returns (rho, theta_deg) compatible with your Hough code:
        - theta_deg in [-90, 90)
        - rho computed in centered coordinates.
    """
    # Image center (same as in hough_bruteforce_intensity_numba)
    # cx = (w - 1) / 2.0
    # cy = (h - 1) / 2.0

    # Shift endpoints to centered coords
    X1, Y1 = x1 - cx, y1 - cy
    X2, Y2 = x2 - cx, y2 - cy

    # Direction vector of the line in centered coords
    dx = X2 - X1
    dy = Y2 - Y1

    if dx == 0 and dy == 0:
        raise ValueError("Degenerate line: the two endpoints are identical")

    # A normal vector to the line
    nx = dy
    ny = -dx

    # Unit normal -> cos(theta), sin(theta)
    norm_n = np.hypot(nx, ny)
    cos_t = nx / norm_n
    sin_t = ny / norm_n

    # Angle of the normal
    theta = np.arctan2(sin_t, cos_t)   # in (-pi, pi]

    # Wrap theta to your range [-pi/2, pi/2)
    if theta < -np.pi / 2:
        theta += np.pi
    elif theta >= np.pi / 2:
        theta -= np.pi

    theta_deg = np.rad2deg(theta)
    comp_cos_t = np.cos(theta)
    comp_sin_t = np.sin(theta)
    
    # rho in centered coordinates (same formula as your accumulator)
    # rho = X1 * cos_t + Y1 * sin_t
    rho = X1 * comp_cos_t + Y1 * comp_sin_t
    return rho, theta_deg


def endpoints_to_rho_theta_mod(x1, y1, x2, y2, cx, cy):
    """
    (x1, y1), (x2, y2) are image coords (0..w-1, 0..h-1)
    Returns (rho, theta_deg) with theta in [0, 180) to match line2hough.
    """
    # Image center (same as in hough_bruteforce_intensity_numba)
    # cx = (w - 1) / 2.0
    # cy = (h - 1) / 2.0

    # Shift endpoints to centered coords
    X1, Y1 = x1 - cx, y1 - cy
    X2, Y2 = x2 - cx, y2 - cy

    # Direction vector of the line in centered coords
    dx = X2 - X1
    dy = Y2 - Y1

    if dx == 0 and dy == 0:
        raise ValueError("Degenerate line: the two endpoints are identical")

    # A normal vector to the line
    nx = dy
    ny = -dx

    # Unit normal -> cos(theta), sin(theta)
    norm_n = np.hypot(nx, ny)
    cos_t = nx / norm_n
    sin_t = ny / norm_n

    # Angle of the normal
    theta = np.arctan2(sin_t, cos_t)   # in (-pi, pi]

    # Wrap theta to [-pi/2, pi/2)
    if theta < -np.pi / 2:
        theta += np.pi
    elif theta >= np.pi / 2:
        theta -= np.pi
    
    comp_cos_t = np.cos(theta)
    comp_sin_t = np.sin(theta)
    
    # rho in centered coordinates (same formula as your accumulator)
    rho = X1 * comp_cos_t + Y1 * comp_sin_t

    # Move theta to [0, pi) to match line2hough; flip rho when rotating 180°
    theta_deg = np.rad2deg(theta)
    if theta_deg < 0:
        theta_deg += 180.0
        rho = -rho
    return rho, theta_deg


def rho_theta_to_indices(rho, theta_deg, theta_min, theta_res, rho_min, rho_res):
    t_idx = int(round((theta_deg - theta_min) / theta_res))
    r_idx = int(round((rho - rho_min) / rho_res))

    return r_idx, t_idx



def line_endpoints_center_rho_theta(rho, theta, H, W, eps=1e-10):
    """Return (x0,y0),(x1,y1) where the line X*cos(theta)+Y*sin(theta)=rho
    (origin at image centre cx=W/2, cy=H/2) intersects an HxW image."""
    cx = W / 2.0
    cy = H / 2.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    points = []
    if abs(sin_t) > eps:
        y = cy + (rho + cx * cos_t) / sin_t
        if -0.5 <= y <= H - 0.5:
            points.append((0.0, float(y)))
        y = cy + (rho - (W - 1 - cx) * cos_t) / sin_t
        if -0.5 <= y <= H - 0.5:
            points.append((W - 1.0, float(y)))
    if abs(cos_t) > eps:
        x = cx + (rho + cy * sin_t) / cos_t
        if -0.5 <= x <= W - 0.5:
            points.append((float(x), 0.0))
        x = cx + (rho - (H - 1 - cy) * sin_t) / cos_t
        if -0.5 <= x <= W - 0.5:
            points.append((float(x), H - 1.0))
    uniq = []
    for p in points:
        if not any(abs(p[0] - q[0]) < 0.5 and abs(p[1] - q[1]) < 0.5 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None, None
    if len(uniq) > 2:
        best, max_d = (uniq[0], uniq[1]), -1.0
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                d2 = (uniq[i][0]-uniq[j][0])**2 + (uniq[i][1]-uniq[j][1])**2
                if d2 > max_d:
                    max_d, best = d2, (uniq[i], uniq[j])
        p0, p1 = best
    else:
        p0, p1 = uniq[0], uniq[1]
    x0 = min(max(int(round(p0[0])), 0), W - 1)
    y0 = min(max(int(round(p0[1])), 0), H - 1)
    x1 = min(max(int(round(p1[0])), 0), W - 1)
    y1 = min(max(int(round(p1[1])), 0), H - 1)
    return (x0, y0), (x1, y1)


def _make_rt_gaussian_mask(rt_centers, num_rhos, num_angles,
                           sigma_rho=5.0, sigma_theta=5.0):
    heatmap = np.zeros((num_angles, num_rhos), dtype=np.float32)
    if not rt_centers:
        return heatmap.astype(np.uint8)
    rho_idx = np.arange(num_rhos, dtype=np.float32)
    theta_idx = np.arange(num_angles, dtype=np.float32)
    grid_rho, grid_theta = np.meshgrid(rho_idx, theta_idx)
    for r_id, t_id in rt_centers:
        blob = np.exp(
            -0.5 * ((grid_rho - r_id) ** 2 / (sigma_rho ** 2) +
                    (grid_theta - t_id) ** 2 / (sigma_theta ** 2))
        )
        heatmap = np.maximum(heatmap, blob)
    return (heatmap * 255).clip(0, 255).astype(np.uint8)


def tile_images(img_path, annotations, out_img_dir, out_rt_dir,
                out_line_mask_dir, out_poly_mask_dir,
                step_size, tile_width, tile_height, buffer_size,
                num_angles, num_rhos,
                rho_min_cap=None, rho_max_cap=None,
                gt_mask_mode='box',
                gaussian_sigma_rho=5.0, gaussian_sigma_theta=5.0,
                line_mask_thickness=1):
    img = np.load(img_path)
    if img is None:
        return

    height, width = img.shape
    no_image_produced_flag = True
    theta_min = 0
    itheta = 180 / num_angles
    max_rho = np.hypot(tile_width, tile_height) + 1
    cx = tile_width / 2.0
    cy = tile_height / 2.0
    base = os.path.basename(img_path).split('.')[0]

    # irho and rho_min consistent with training_data_gen_for_RT
    if rho_min_cap is not None and rho_max_cap is not None:
        _effective_half_rho = (rho_max_cap - rho_min_cap) / 2.0
    elif rho_max_cap is not None:
        _effective_half_rho = float(rho_max_cap)
    elif rho_min_cap is not None:
        _effective_half_rho = abs(float(rho_min_cap))
    else:
        _effective_half_rho = max_rho / 2.0
    irho = (2.0 * _effective_half_rho) / (num_rhos - 1)
    _rlo = rho_min_cap if rho_min_cap is not None else -(max_rho / 2)

    def _compute_rt(tile_data):
        rt, _, _ = compute_rt_map(tile_data.astype(np.float32), max_rho / 2,
                                  theta_res_deg=itheta, rho_res=irho,
                                  rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap)
        rt = rt - rt.min()
        rt = rt / rt.max() * 255
        return rt.astype(np.uint8)

    def _normalize_tile(tile_data):
        zero_mask = tile_data == 0
        t = (tile_data - tile_data.min()).astype(np.float32)
        t /= t.max()
        t *= 255
        t[zero_mask] = 0
        return t.astype(np.uint8)

    bg_candidates = []

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            start_x = x if x + tile_width <= width else width - tile_width
            start_y = y if y + tile_height <= height else height - tile_height

            polys = []
            rt_centers = []
            lines_in_tile = []
            streak_flag = False

            for line in annotations:
                line = [(line[0], line[1]), (line[2], line[3])]
                adjusted_line = clip_line_to_tile(start_x, start_y, tile_width, tile_height, line)
                if adjusted_line is None:
                    continue
                line_len = math.sqrt((adjusted_line[2] - adjusted_line[0])**2 +
                                     (adjusted_line[3] - adjusted_line[1])**2)
                if line_len <= 50:
                    continue

                rho, theta_deg = endpoints_to_rho_theta_mod(
                    adjusted_line[0], adjusted_line[1],
                    adjusted_line[2], adjusted_line[3], cx, cy)
                r_id, t_id = rho_theta_to_indices(rho, theta_deg, theta_min, itheta, _rlo, irho)

                x1 = np.minimum(np.maximum((r_id - 10) / num_rhos, 0), 1.0)
                y1 = np.minimum(np.maximum((t_id - 10) / num_angles, 0), 1.0)
                x2 = np.minimum(np.maximum((r_id + 10) / num_rhos, 0), 1.0)
                y2 = np.minimum(np.maximum((t_id - 10) / num_angles, 0), 1.0)
                x3 = np.minimum(np.maximum((r_id + 10) / num_rhos, 0), 1.0)
                y3 = np.minimum(np.maximum((t_id + 10) / num_angles, 0), 1.0)
                x4 = np.minimum(np.maximum((r_id - 10) / num_rhos, 0), 1.0)
                y4 = np.minimum(np.maximum((t_id + 10) / num_angles, 0), 1.0)

                polys.append((x1, y1, x2, y2, x3, y3, x4, y4))
                rt_centers.append((r_id, t_id))
                lines_in_tile.append(adjusted_line)
                streak_flag = True

            if not streak_flag:
                bg_candidates.append((x, y, start_x, start_y))
                continue

            # streak tile
            tile_crop = img[start_y:start_y + tile_height, start_x:start_x + tile_width].copy()
            rt_map = _compute_rt(tile_crop)
            tile_norm = _normalize_tile(tile_crop)
            h_rt, w_rt = rt_map.shape[:2]

            line_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
            for adj_line in lines_in_tile:
                pt0 = (int(round(adj_line[0])), int(round(adj_line[1])))
                pt1 = (int(round(adj_line[2])), int(round(adj_line[3])))
                cv2.line(line_mask, pt0, pt1, color=255,
                         thickness=line_mask_thickness, lineType=cv2.LINE_AA)
            line_mask = (line_mask > 0).astype(np.uint8) * 255

            if gt_mask_mode == 'gaussian':
                poly_mask = _make_rt_gaussian_mask(rt_centers, num_rhos, num_angles,
                                                   sigma_rho=gaussian_sigma_rho,
                                                   sigma_theta=gaussian_sigma_theta)
            else:
                poly_mask = np.zeros((h_rt, w_rt), dtype=np.uint8)
                for poly in polys:
                    xs = np.array(poly[0::2])
                    ys = np.array(poly[1::2])
                    px = np.clip(np.round(xs * (w_rt - 1)), 0, w_rt - 1).astype(np.int32)
                    py = np.clip(np.round(ys * (h_rt - 1)), 0, h_rt - 1).astype(np.int32)
                    pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
                    cv2.fillPoly(poly_mask, [pts], color=1)
                poly_mask = (poly_mask > 0).astype(np.uint8) * 255

            stem = f"{base}_tile_{x}_{y}"
            cv2.imwrite(f"{out_img_dir}{stem}.png", tile_norm)
            cv2.imwrite(f"{out_rt_dir}{stem}.png", rt_map)
            cv2.imwrite(f"{out_line_mask_dir}{stem}.png", line_mask)
            cv2.imwrite(f"{out_poly_mask_dir}{stem}.png", poly_mask)
            no_image_produced_flag = False

    # Pass 2: randomly sample up to 70 background tiles
    random.shuffle(bg_candidates)
    for (x, y, start_x, start_y) in bg_candidates[:70]:
        tile_crop = img[start_y:start_y + tile_height, start_x:start_x + tile_width].copy()
        rt_map = _compute_rt(tile_crop)
        tile_norm = _normalize_tile(tile_crop)
        h_rt, w_rt = rt_map.shape[:2]

        line_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
        poly_mask = np.zeros((h_rt, w_rt), dtype=np.uint8)

        stem = f"{base}_tile_{x}_{y}"
        cv2.imwrite(f"{out_rt_dir}{stem}.png", rt_map)
        cv2.imwrite(f"{out_img_dir}{stem}.png", tile_norm)
        cv2.imwrite(f"{out_line_mask_dir}{stem}.png", line_mask)
        cv2.imwrite(f"{out_poly_mask_dir}{stem}.png", poly_mask)
        no_image_produced_flag = False

    return no_image_produced_flag


def create_image_list_file(img_dir, text_dir):
    # Get all the image names in the img_dir
    image_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # Open the text file for writing
    with open(text_dir, 'w') as file:
        for image_name in image_names:
            # Write each image name in the specified format
            file.write(f"{img_dir}/{image_name}\n")


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


# =========================
USE_ARGPARSE = False  # set True to use CLI; False to use CONFIG below

CONFIG = {
    # IO
    "output_dir": "/home/ckchng/Documents/SDA_ODA/LMA_data/val_gaussian/",
    "img_dir": "/media/ckchng/internal2TB/FILTERED_IMAGES/",
    "anno_dir": "/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label.pkl",

    "starting_id": 0,
    "step": 126,
    "tile_width": 288,
    "tile_height": 288,
    "step_size": 144,
    "num_angles": 192,
    "num_rhos": 416,
    "rho_min_cap": None,
    "rho_max_cap": None,
    "line_mask_thickness": 3,
    "gt_mask_mode": "gaussian",        # 'box' or 'gaussian'
    "gaussian_sigma_rho": 5.0,
    "gaussian_sigma_theta": 5.0,
}

def _parse_args_or_config():
    if not USE_ARGPARSE:
        class _A: pass
        a = _A()
        for k, v in CONFIG.items():
            setattr(a, k, v)
        return a

    parser = argparse.ArgumentParser(description="Script to process data.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--anno_dir", required=True)
    
    parser.add_argument("--starting_id", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--tile_width", type=int, required=True)
    parser.add_argument("--tile_height", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)
    parser.add_argument("--num_angles", type=int, default=192)
    parser.add_argument("--num_rhos", type=int, default=416)
    parser.add_argument("--rho_min_cap", type=float, default=None)
    parser.add_argument("--rho_max_cap", type=float, default=None)
    parser.add_argument("--line_mask_thickness", type=int, default=3)
    parser.add_argument("--gt_mask_mode", type=str, default="box", choices=["box", "gaussian"])
    parser.add_argument("--gaussian_sigma_rho", type=float, default=5.0)
    parser.add_argument("--gaussian_sigma_theta", type=float, default=5.0)

    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args_or_config()
    output_dir = args.output_dir
    img_dir = args.img_dir

    anno_dir = args.anno_dir
    starting_id = args.starting_id
    step = args.step
    ending_id = starting_id + step
    
    tile_width = args.tile_width
    tile_height = args.tile_height
    step_size = args.step_size
    num_angles = args.num_angles
    num_rhos = args.num_rhos
    rho_min_cap = args.rho_min_cap
    rho_max_cap = args.rho_max_cap
    line_mask_thickness = args.line_mask_thickness
    gt_mask_mode = args.gt_mask_mode
    gaussian_sigma_rho = args.gaussian_sigma_rho
    gaussian_sigma_theta = args.gaussian_sigma_theta

    output_img_dir      = output_dir + '/actual_images/'
    output_rt_dir       = output_dir + '/images/'
    output_line_mask_dir = output_dir + '/line_masks/'
    output_poly_mask_dir = output_dir + '/poly_masks/'

    print("Building NEF file mapping...")
    img_paths = {}
    for root, dirs, files in os.walk(img_dir):
        for filename in files:
            if filename.endswith(".npy"):
                img_name = os.path.splitext(filename)[0]
                img_path = os.path.join(root, filename)
                img_paths[img_name] = img_path

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_rt_dir, exist_ok=True)
    os.makedirs(output_line_mask_dir, exist_ok=True)
    os.makedirs(output_poly_mask_dir, exist_ok=True)

    with open(anno_dir, 'rb') as f:
        data_dict = pickle.load(f)

    image_dirs = data_dict['imgPath']
    annotations = data_dict['XY']
    # print(len(image_dirs))

    #for each image_dirs, find the corresponding in img_paths, and replace image_dirs with the path in image_paths
    for i in range(len(image_dirs)):
        img_name = os.path.splitext(os.path.basename(image_dirs[i]))[0]
        img_name = img_name.split('.')[0]  # in case there are multiple dots
        
        if img_name in img_paths:
            image_dirs[i] = img_paths[img_name]
        else:
            print(f"Image {img_name} not found in img_paths.")

    # ending_id = np.minimum(len(image_dirs), ending_id)
    ending_id = len(image_dirs)
    ending_id = args.starting_id + args.step

    no_image_produced_id = []

    # find the int_id for a particular image_id in image_dirs
    
    
    for int_id in range(starting_id, ending_id):
        # int_img_id = '000_2020-12-08_104028_E_DSC_0507'
        # int_id = np.where(np.array(image_dirs) == img_paths[int_img_id])[0][0]
        # print(f"int_id for {int_img_id} is {int_id}")   
        # print(image_dirs[int_id])
        no_image_produced_flag = tile_images(
            image_dirs[int_id], annotations[int_id][0],
            output_img_dir, output_rt_dir,
            output_line_mask_dir, output_poly_mask_dir,
            step_size=step_size, tile_width=tile_width, tile_height=tile_height, buffer_size=10,
            num_angles=num_angles, num_rhos=num_rhos,
            rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap,
            gt_mask_mode=gt_mask_mode,
            gaussian_sigma_rho=gaussian_sigma_rho, gaussian_sigma_theta=gaussian_sigma_theta,
            line_mask_thickness=line_mask_thickness)
        no_image_produced_id.append(image_dirs[int_id])

    # check how many of the no_image_produced_id has zero len annotations
    # count = 0
    # for img_path in no_image_produced_id:
    #     img_name = os.path.splitext(os.path.basename(img_path))[0]
    #     if len(annotations[image_dirs.index(img_path)][0]) == 0:
    #         count += 1
        
    img_list_dir = output_dir + '/' + str(starting_id) +'.txt'
    create_image_list_file(output_img_dir, img_list_dir)
