import numpy as np
from numba import njit, prange

######################### for deep hough layers' convention ###########################3\
@njit(parallel=True, fastmath=True)
def _hough_accumulate_intensity_dh(img, cos_t, sin_t, rhos, cx, cy, rho_res):
    h, w = img.shape
    T = cos_t.shape[0]
    R = rhos.shape[0]
    H = np.zeros((T, R), dtype=np.float32)
    # Use the same integer offset as rho_theta_to_indices_dh:
    #   r_idx = round(rho / rho_res) - round(rhos[0] / rho_res)
    # rhos[0] may not be an exact multiple of rho_res (np.arange drift), so we
    # round it once here rather than using the raw float, which caused a
    # systematic bin offset whenever rho_res != 1.
    r0 = int(round(rhos[0] / rho_res))

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
                r_idx = int(round(rho / rho_res)) - r0
                if 0 <= r_idx < R:
                    H[t_idx, r_idx] += v
    return H

def endpoints_to_rho_theta_dh(x1, y1, x2, y2, cx, cy):
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


def rho_theta_to_indices_dh(rho, theta_deg, theta_min, theta_res, rho_min, rho_res):
    t_idx = int(round((theta_deg - theta_min) / theta_res))
    r_idx = int(round((rho - rho_min) / rho_res))
    return r_idx, t_idx


# ---------- tiny helpers ----------
# def _make_params_dh(max_rho, theta_res_deg=1.0, rho_res=1.0):

#     thetas_deg = np.arange(0, 180, theta_res_deg, dtype=np.float64)
#     thetas = np.deg2rad(thetas_deg)
#     cos_t = np.cos(thetas)
#     sin_t = np.sin(thetas)
    
#     rhos = np.arange(-max_rho, max_rho + rho_res, rho_res, dtype=np.float64)
#     return thetas_deg, thetas, cos_t, sin_t, rhos

def _make_params_dh(max_rho, theta_res_deg=1.0, rho_res=1.0,
                 rho_min_cap=None, rho_max_cap=None):
    """Build Hough parameter arrays.

    rho_min_cap / rho_max_cap (float or None):
        If provided, the rho axis is clamped to [rho_min_cap, rho_max_cap]
        instead of the full [-max_rho, +max_rho] range.
    """
    thetas_deg = np.arange(0, 180, theta_res_deg, dtype=np.float64)
    thetas = np.deg2rad(thetas_deg)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    rho_lo = rho_min_cap if rho_min_cap is not None else -max_rho
    rho_hi = rho_max_cap if rho_max_cap is not None else  max_rho
    rhos = np.arange(rho_lo, rho_hi, rho_res, dtype=np.float64)
    return thetas_deg, thetas, cos_t, sin_t, rhos

# def hough_bruteforce_intensity_numba_dh(img, max_rho, theta_res_deg=1.0, rho_res=1.0):
#     h, w = img.shape
#     cy = h / 2.0
#     cx = w / 2.0
    
#     theta_deg, theta_rad, cos_t, sin_t, rhos = _make_params_dh(max_rho, theta_res_deg, rho_res)

#     # Ensure numeric type Numba likes
#     img32 = img.astype(np.float32, copy=False)
#     H = _hough_accumulate_intensity_dh(img32, cos_t, sin_t, rhos, cx, cy, rho_res)
#     return H, theta_deg, rhos

def hough_bruteforce_intensity_numba_dh(img, max_rho, num_rhos, num_degs, theta_res_deg=1.0, rho_res=1.0,
                                     rho_min_cap=None, rho_max_cap=None):
    """Compute the Hough accumulator for img.

    rho_min_cap / rho_max_cap (float or None):
        Cap the rho axis to [rho_min_cap, rho_max_cap].  Lines whose
        tile-local rho falls outside this range are silently ignored by
        the accumulator (they contribute no votes to the RT map).
        Pass None (default) to use the full [-max_rho, +max_rho] range.
    """
    h, w = img.shape
    cy = h / 2.0
    cx = w / 2.0

    theta_deg, theta_rad, cos_t, sin_t, rhos = _make_params_dh(
        max_rho, theta_res_deg, rho_res,
        rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap)

    # Ensure numeric type Numba likes
    img32 = img.astype(np.float32, copy=False)
    H = _hough_accumulate_intensity(img32, cos_t, sin_t, cx, cy, rho_res, num_rhos, num_degs, rhos[0])

    return H, theta_deg, rhos


######################### for original convention ###########################3\
# ---------- core (numba) ----------
# @njit(parallel=True, fastmath=True)
# def _hough_accumulate_intensity(img, cos_t, sin_t, rhos, cx, cy, rho_res):
#     h, w = img.shape
#     T = cos_t.shape[0]
#     R = rhos.shape[0]
#     H = np.zeros((R, T), dtype=np.float64)

#     # Parallelize over angles so each thread writes to its own column -> no races.
#     for t_idx in prange(T):
#         c = cos_t[t_idx]
#         s = sin_t[t_idx]
#         r0 = rhos[0]
#         for y in range(h):
#             Y = y - cy
#             for x in range(w):
#                 v = img[y, x]
#                 if v == 0:
#                     continue
#                 X = x - cx
#                 rho = X * c + Y * s
#                 r_idx = int((rho - r0) / rho_res + 0.5)   # round 
#                 if 0 <= r_idx < R:
#                     H[r_idx, t_idx] += v
#     return H


# ---------- core (numba) ----------
@njit(parallel=True, fastmath=True)
def _hough_accumulate_intensity(img, cos_t, sin_t, cx, cy, rho_res, num_rhos, num_deg, min_rho):
    h, w = img.shape
    # T = cos_t.shape[0]
    # R = rhos.shape[0]

    H = np.zeros((num_deg, num_rhos), dtype=np.float32)
    r0 = int(round(min_rho / rho_res))

    # Parallelize over angles so each thread writes to its own column -> no races.
    for t_idx in prange(num_deg):
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
                
                if 0 <= r_idx < num_rhos:
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


def _make_params(h, w, theta_res_deg=1.0, rho_res=1.0):
    # Use normal angles in [0, 180) to match endpoints_to_rho_theta_mod / line2hough
    thetas_deg = np.arange(0.0, 180.0, theta_res_deg, dtype=np.float64)
    thetas = np.deg2rad(thetas_deg)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Rho range centered at image center; step = rho_res
    max_rho = np.hypot(h - 1, w - 1) / 2.0
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

def rho_theta_to_indices(rho, theta_deg, theta_min, theta_res, rho_min, rho_res):
    t_idx = int(round((theta_deg - theta_min) / theta_res))
    r_idx = int(round(rho / rho_res)) - rho_min

    return r_idx, t_idx