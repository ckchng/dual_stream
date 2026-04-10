"""
utils.py – shared Hough-transform utilities.

Functions
---------
_compute_rt_kernel         – numba-JIT inner accumulator (parallel)
_make_params               – build theta/rho grids (supports rho capping)
endpoints_to_rho_theta_mod – line endpoints  →  (rho, theta_deg)
rho_theta_to_indices       – (rho, theta_deg) → (r_idx, t_idx)
compute_rt_map             – full RT map (supports rho capping)
"""

import math
import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Internal kernel (numba, parallel)
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def _compute_rt_kernel(img, cos_t, sin_t, rhos, cx, cy, rho_res):
    h, w = img.shape
    T = cos_t.shape[0]
    R = rhos.shape[0]
    H = np.zeros((T, R), dtype=np.float32)
    r0 = rhos[0] / rho_res

    # Parallelise over angles so each thread writes to its own row → no races.
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
                r_idx = int((round(rho / rho_res)) - r0)
                if 0 <= r_idx < R:
                    H[t_idx, r_idx] += v
    return H


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _make_params(max_rho, theta_res_deg=1.0, rho_res=1.0,
                 rho_min_cap=None, rho_max_cap=None):
    """Return (thetas_deg, thetas_rad, cos_t, sin_t, rhos) grids.

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
    rhos = np.arange(rho_lo, rho_hi + rho_res, rho_res, dtype=np.float64)
    # rhos = np.arange(rho_lo, rho_hi, rho_res, dtype=np.float64)
    return thetas_deg, thetas, cos_t, sin_t, rhos


def endpoints_to_rho_theta_mod(x1, y1, x2, y2, cx, cy):
    """
    Convert a line defined by two image-coordinate endpoints to
    (rho, theta_deg) in the convention used by the Hough accumulator.

    Parameters
    ----------
    x1, y1 : float  – first endpoint (image coords)
    x2, y2 : float  – second endpoint (image coords)
    cx, cy  : float – image centre (e.g. (w-1)/2, (h-1)/2)

    Returns
    -------
    rho       : float  – perpendicular distance from the centred origin
    theta_deg : float  – normal angle in [0, 180)
    """
    X1, Y1 = x1 - cx, y1 - cy
    X2, Y2 = x2 - cx, y2 - cy

    dx = X2 - X1
    dy = Y2 - Y1

    if dx == 0 and dy == 0:
        raise ValueError("Degenerate line: the two endpoints are identical")

    # Normal vector to the line
    nx, ny = dy, -dx
    norm_n = np.hypot(nx, ny)
    cos_t = nx / norm_n
    sin_t = ny / norm_n

    theta = np.arctan2(sin_t, cos_t)          # (-pi, pi]

    # Wrap to [-pi/2, pi/2)
    if theta < -np.pi / 2:
        theta += np.pi
    elif theta >= np.pi / 2:
        theta -= np.pi

    comp_cos_t = np.cos(theta)
    comp_sin_t = np.sin(theta)

    rho = X1 * comp_cos_t + Y1 * comp_sin_t

    # Shift to [0, 180); flip rho when rotating by 180°
    theta_deg = np.rad2deg(theta)
    if theta_deg < 0:
        theta_deg += 180.0
        rho = -rho

    return rho, theta_deg


def rho_theta_to_indices(rho, theta_deg, theta_min, theta_res, rho_min, rho_res):
    """
    Map (rho, theta_deg) to integer accumulator indices.

    Returns
    -------
    r_idx, t_idx : int
    """
    t_idx = int(round((theta_deg - theta_min) / theta_res))
    r_idx = int(round((rho - rho_min) / rho_res))
    return r_idx, t_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rt_map(img, max_rho, theta_res_deg=1.0, rho_res=1.0,
                   rho_min_cap=None, rho_max_cap=None):
    """
    Compute the intensity-weighted Hough (Radon) transform of *img*.

    Parameters
    ----------
    img          : 2-D numpy array (float32 preferred)
    max_rho      : float  – half the rho range; rhos span [-max_rho, +max_rho]
    theta_res_deg: float  – angular resolution in degrees (default 1°)
    rho_res      : float  – rho resolution in pixels (default 1)
    rho_min_cap  : float or None – clamp rho axis lower bound (default: -max_rho)
    rho_max_cap  : float or None – clamp rho axis upper bound (default: +max_rho)

    Returns
    -------
    H         : (num_angles, num_rhos) float32 accumulator
    theta_deg : 1-D array of angle values
    rhos      : 1-D array of rho values
    """
    h, w = img.shape
    cy = h / 2.0
    cx = w / 2.0

    theta_deg, _theta_rad, cos_t, sin_t, rhos = _make_params(
        max_rho, theta_res_deg, rho_res,
        rho_min_cap=rho_min_cap, rho_max_cap=rho_max_cap)

    img32 = img.astype(np.float32, copy=False)
    H = _compute_rt_kernel(img32, cos_t, sin_t, rhos, cx, cy, rho_res)
    return H, theta_deg, rhos
