"""
Created on Wed Dec 17 22:03:47 2025
@author: santaro



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

from pathlib import Path

class MyFitting:
    def __init__(self, points):
        self._points = points
        self._cache = []
    @property
    def points(self):
        return self._points
    @property
    def cache(self):
        return self._cache

    def lsm_for_line(self, allow_nan=False):
        points = self.points
        if points.shape[1] != 2:
            raise ValueError("argument of points must be shape of (N, 2).")
        has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
        if not allow_nan and has_nan_or_inf:
            return np.full(3, np.nan), None
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        valid_id = np.where(valid_mask)[0]
        x = points[valid_id, 0]
        y = points[valid_id, 1]
        num_points = len(x)
        M1 = np.vstack([x, np.ones(num_points)]).T
        ab, residuals, rank, s = np.linalg.lstsq(M1, y, rcond=None) # ab is slope and intercept
        info = {
            "rss": float(residuals[0]) if residuals.size > 0 else None,
            "rank": int(rank),
            "singular_values": s,
            "num_points": len(x),
            "valid_indices": valid_id
        }
        # M2 = np.array(y)
        # ab = np.linalg.inv(M1.T @ M1) @ M1.T @ M2
        _cache = {
            "method": "lsm_for_line",
            "shape": "line",
            "ab": ab,
            "info": info,
        }
        self._cache.append(_cache)
        return ab, info

    def check(self, cache_id=0):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.grid()
        _cache = self.cache[cache_id]
        xrange = (np.nanmin(self.points.T[0]), np.nanmax(self.points.T[0]))
        xmin = xrange[0] - (xrange[1] - xrange[0])*0.2
        xmax = xrange[1] + (xrange[1] - xrange[0])*0.2
        t = np.linspace(xmin, xmax, 100, endpoint=True)
        if _cache["shape"] == "line":
            ab = _cache["ab"]
            x = t
            y = ab[0] * x + ab[1]
            ax.plot(x, y, lw=1, c='b')
            ax.scatter(self.points[:, 0], self.points[:, 1], c='k', s=20)
        plt.show()


#### least squares fitting of line
def lsm_for_line(points, allow_nan=False):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(3, np.nan), None
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_id = np.where(valid_mask)[0]
    x = points[valid_id, 0]
    y = points[valid_id, 1]
    num_points = len(x)
    M1 = np.vstack([x, np.ones(num_points)]).T
    ab, residuals, rank, s = np.linalg.lstsq(M1, y, rcond=None) # ab is slope and intercept
    info = {
        "rss": float(residuals[0]) if residuals.size > 0 else None,
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "valid_indices": valid_id
    }
    # M2 = np.array(y)
    # ab = np.linalg.inv(M1.T @ M1) @ M1.T @ M2
    return ab, info

#### least squares fitting of circles
def lsm_for_circle(points, allow_nan=False):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(3, np.nan), None
    valid_points_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_points_id = np.where(valid_points_mask)[0]
    if len(valid_points_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    x = points[valid_points_id, 0]
    y = points[valid_points_id, 1]
    num_points = len(x)
    M1 = np.vstack([x, y, np.ones(num_points)]).T
    M2 = -(x**2 + y**2)
    coef, residuals, rank, s = np.linalg.lstsq(M1, M2, rcond=None)
    A, B, C = coef
    # A, B, C = np.dot(np.linalg.inv(np.dot(M1.T, M1)), np.dot(M1.T, M2))
    cx, cy = -A/2, -B/2
    r = (cx**2 + cy**2 - C)**0.5
    xyr = np.array([cx, cy, r])
    d_center = np.sqrt((x - cx)**2 +(y - cy)**2)
    geom_err = np.abs(d_center - r)
    info = {
        "rss": float(residuals[0]) if residuals.size > 0 else None,
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "num_valid_points": len(valid_points_id),
        "valid_points_indices": valid_points_id,
        "radii": d_center,
        "geom_error_mean": float(np.mean(geom_err)),
        "geom_error_std": float(np.std(geom_err)),
        "geom_error_max": float(np.max(geom_err)),
    }
    return xyr, info

def lsm_for_circles(points):
    """
    Least squares fitting of circles for sequential frames.
    it doesn't use linalg.lstsq of numpy to avoid for loops for frames, is useful when the number of frames is greater than that of points.

    """
    if points.shape[2] != 2:
        raise ValueError("argument of points must be shape of (M, N, 2).")
    valid_frames_mask = (~np.isnan(points).any(axis=2) & ~np.isinf(points).any(axis=2)).all(axis=1)
    valid_frames_id = np.where(valid_frames_mask)[0]
    if len(valid_frames_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    num_frames = points.shape[0]
    num_points = points.shape[1]
    xyrs = np.full((num_frames, 3), np.nan)
    d_center = np.full((num_frames, num_points), np.nan)
    xs = points[valid_frames_id, :, 0]
    ys = points[valid_frames_id, :, 1]
    num_frames_valid = xs.shape[0]
    M1 = np.stack([xs, ys, np.ones((num_frames_valid, num_points))], axis=2)
    M1T = M1.transpose(0, 2, 1)
    M2 = -(xs**2 + ys**2)[:, :, np.newaxis]
    inv_M1T_M1 = np.linalg.inv(M1T @ M1)
    # inv_M1T_M1 = np.linalg.pinv(M1T @ M1)
    M1T_M2 = M1T @ M2
    ABC = (inv_M1T_M1 @ M1T_M2).T.squeeze()
    # A, B, C = ABC[0, 0], ABC[0, 1], ABC[0, 2]
    # print(ABC.shape)
    A, B, C = ABC[0], ABC[1], ABC[2]
    # Compute the circle centers (cx, cy) and radius (r)
    cx = -A / 2
    cy = -B / 2
    r = np.sqrt(cx**2 + cy**2 - C)  # Shape (num_frames,)
    xyrs[valid_frames_id, :] = np.column_stack([cx, cy, r])
    d_center[valid_frames_id, :] = np.sqrt((xs - cx[:, np.newaxis])**2 + (ys - cy[:, np.newaxis])**2)
    info = {
        "num_points": num_points,
        "num_frames": num_frames,
        "num_valid_frames": num_frames_valid,
        "valid_indices": valid_frames_id,
        "radii": d_center,
    }
    return xyrs, info

def lsm_for_ellipse(points, allow_nan=False):
    """
    Least squares fitting of ellipse
    points: shape of (num_points, 2)

    """
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(5, np.nan), None
    valid_points_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_points_id = np.where(valid_points_mask)[0]
    if len(valid_points_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    x = points[valid_points_id, 0]
    y = points[valid_points_id, 1]
    num_points, _ = points.shape
    num_points_valid = len(valid_points_id)
    M1 = np.vstack([x*y, y**2, x, y, np.ones(num_points_valid)]).T
    M2 = -x**2
    coef, residuals, rank, s = np.linalg.lstsq(M1, M2, rcond=None)
    A, B, C, D, E = coef
    cx = (A*D - 2*B*C) / (4*B - A**2)
    cy = (A*C - 2*D) / (4*B - A**2)
    if abs(A/(1-B)) > 10**14: # first aid to avoid singular error
        a , b = np.nan, np.nan
        # a , b = 0, 0
        theta = np.radians(45)
    else:
        theta = np.arctan(A/(1-B)) / 2
        # theta = np.atan2(A, (1-B)) / 2
        sin = np.sin(theta)
        cos = np.cos(theta)
        a = np.sqrt(
            (cx*cos + cy*sin)**2 - E*cos**2
                - ((cx*sin - cy*cos)**2 - E*sin**2)
                    *(sin**2 - B*cos**2) / (cos**2 - B*sin**2)
            )
        b = np.sqrt(
            (cx*sin - cy*cos)**2 - E*sin**2
                - ((cx*cos + cy*sin)**2 - E*cos**2)
                    *(cos**2 - B*sin**2) / (sin**2 - B*cos**2)
            )
    result = [cx, cy, a, b, theta]
    d_center = np.sqrt((x - cx)**2 +(y - cy)**2)
    geom_err = np.full(num_points, np.nan)
    sin = np.sin(theta)
    cos = np.cos(theta)
    ux = points[:, 0] - cx
    uy = points[:, 1] - cy
    x_local = ux * cos + uy * sin
    y_local = -ux * sin + uy * cos
    for i in range(num_points):
        geom_err[i], _ = calc_mindist_p2ellipse(np.array([x_local[i], y_local[i]]), a, b)
    info = {
        "rss": float(residuals[0]) if residuals.size > 0 else None,
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "num_valid_points": num_points_valid,
        "valid_points_indices": valid_points_id,
        "radii": d_center,
        "geom_error_mean": float(np.mean(geom_err)),
        "geom_error_std": float(np.std(geom_err)),
        "geom_error_max": float(np.max(geom_err)),
    }
    return result, info

def fitzgibbon_ellipse(points, allow_nan=False):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    has_nan_or_inf = np.isnan(points).any() or np.isinf(points).any()
    if not allow_nan and has_nan_or_inf:
        return np.full(5, np.nan), None
    valid_points_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_points_id = np.where(valid_points_mask)[0]
    if len(valid_points_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    x = points[valid_points_id, 0]
    y = points[valid_points_id, 1]
    num_points, _ = points.shape
    num_points_valid = len(valid_points_id)
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    S = np.dot(D.T, D)
    C = np.zeros((6, 6))
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2
    eigvals, eigvecs = linalg.eig(S, C)
    pos_idx = np.where((eigvals > 0) & (np.isfinite(eigvals)))[0]
    abcdef = eigvecs[:, pos_idx].real.flatten()
    return abcdef

def abcdef2xyabtheta(abcdef):
    a, b, c, d, e, f = abcdef
    delta = b**2 - 4*a*c
    cx = (b*e - 2*c*d) / (-delta)
    cy = (b*d - 2*a*e) / (-delta)
    theta = 0.5 * np.arctan2(b, a-c)
    up = 2 * (a * cx**2 + b * cx * cy + c * cy**2 -f)
    down1 = a + c - np.sqrt((a - c)**2 + b**2)
    down2 = a + c + np.sqrt((a - c)**2 + b**2)
    semi_major = np.sqrt(max(0, up/down1))
    semi_minor = np.sqrt(max(0, up/down2))
    return {
        "center": np.array([cx, cy]),
        # "axes": np.array([semi_minor, semi_major]),
        "axes": np.array([min(semi_minor, semi_major), max(semi_minor, semi_major)]),
        "angle": theta
    }

#### calculate elliptical deformation
def calc_elliptical_deformation(points, points_ref):
    """
    evaluate elliptical deformation; roundness, major and minor axis, ect
    parameter:
    points: in the shape of (num_frames, num_points, 2)
    points_ref: in the shape of (num_points, 2)

    """
    if points.ndim == 2:
        points = points[np.newaxis, :, :]
        num_frames, num_points, _ = points.shape
    elif points.ndim == 3:
        num_frames, num_points, _ = points.shape
    if points_ref.ndim == 2:
        points_ref = points_ref[np.newaxis, :, :]
    num_axes = num_points // 2
    #### calclate diameters
    diameters_vct = points[:, :num_axes, :] - points[:, num_axes:, :]
    diameters_norm = np.linalg.norm(diameters_vct, axis=2)
    diameters_theta = np.arctan2(diameters_vct[:, :, 1], diameters_vct[:, :, 0])
    diameters_ref_vct = points_ref[:, :num_axes, :] - points_ref[:, num_axes:, :]
    diameters_ref_norm = np.linalg.norm(diameters_ref_vct, axis=2)
    delta_diameters = diameters_norm - diameters_ref_norm
    roundness = (np.amax(delta_diameters, axis=1) - np.amin(delta_diameters, axis=1)) / 2
    direction_id = np.array([np.argmin(delta_diameters, axis=1), np.argmax(delta_diameters, axis=1)]).T
    direction = np.array([diameters_theta[np.arange(num_frames), direction_id[:, 0]], diameters_theta[np.arange(num_frames), direction_id[:, 1]]]).T
    deformation_angle = np.abs(direction[:, 1] - direction[:, 0]) % np.pi
    results = {
        "diameters_norm": diameters_norm,
        "delta_diameters": delta_diameters,
        "roundness": roundness,
        "direction_id": direction_id,
        "direction": direction,
        "deformation_angle": deformation_angle,
    }
    return results

def calc_mindist_p2ellipse(point, a, b, mode="newton", tol=1e-12, max_iter=100):
    """
    Calculates the minimum distance from a point (p, q) to an ellipse with semi-axes a and b.
    this is invalid in certain parameters like (0.001, 0.001), 1, 2 because of inner_sqrt become negative.

    """
    # Avoid division by zero
    p, q = point
    if p == 0 and q == 0:
        theta = 0 if a <= b else np.radians(90)
        return min(a, b), theta
    if b == 0 or q == 0:
        return None, None
    if mode == "newton":
        is_converge = False
        sp = 1 if p >= 0 else -1
        sq = 1 if q >= 0 else -1
        P = abs(p)
        Q = abs(q)
        theta = np.arctan2(Q*a, P*b) # initial value
        for _ in range(max_iter):
            ct = np.cos(theta)
            st = np.sin(theta)
            x = a * ct
            y = b * st
            dx = x - P
            dy = y - Q
            g = dx * (-a * st) + dy * (b * ct)
            gp = (-a * ct) * (-a * st) + dx * (-a * ct) + (b * st)*(b * ct) + dy * (-b * st)
            if abs(gp) < 1e-14:
                is_converge = True
                break
            step = g / gp
            theta_new = theta - step
            if abs(step) < tol:
                theta = theta_new
                is_converge = True
                break
            theta = theta_new
        if not is_converge:
            return None, None
        x = a * np.cos(theta) * sp
        y = b * np.sin(theta) * sq
        r = np.sqrt((x-p)**2 + (y-q)**2)
        theta = np.arctan2(y/b, x/a)
    elif mode == "algebra":
        # Calculate intermediate variables
        A = (-a**2 - a*p + b**2) / (b * q)
        B = (-a**2 + a*p + b**2) / (b * q)
        D = (a**2 - b**2)**2 - (a**2 * p**2) - (b**2 * q**2)
        term1 = -432 * a * b * p * q * (a**2 - b**2)
        term2 = 12 * D
        inner_sqrt = np.sqrt(term1**2 - 4 * (term2**3))
        C = np.cbrt(term1 + inner_sqrt)
        cbrt2 = np.cbrt(2)
        denom_bqC = b * q * C
        denom_bq_cbrt = 3 * cbrt2 * b * q
        term_D = (4 * cbrt2 * D) / denom_bqC
        term_C = C / denom_bq_cbrt
        sqrt_part1 = np.sqrt(A**2 + term_D + term_C)
        inner_numerator = 2 * A**3 - 4 * B
        nested_radical = np.sqrt(
            2 * A**2 - term_D - term_C - (inner_numerator / sqrt_part1)
        )
        # soleve
        theta_val = (A / 2) - (0.5 * sqrt_part1) + (0.5 * nested_radical)
        theta = 2 * np.arctan(theta_val)
        r = np.sqrt((a * np.cos(theta) - p)**2 + (b * np.sin(theta) - q)**2)
    return r, theta


# def calc_cumulative_angles(angles, threshold=300, unit='deg'):
    full_angle = 360 if unit=='deg' else 2*np.pi
    if np.ndim(angles) == 1:
        d_angles = angles[1:] - angles[:-1]
        flag_forward = np.hstack([0, np.where(d_angles<-threshold, 1, 0)])
        flag_backward = np.hstack([0, np.where(d_angles>threshold, -1, 0)])
        flag = np.cumsum(flag_forward) + np.cumsum(flag_backward)
        angles_corrected = angles + flag * full_angle
    elif np.ndim(angles) > 1:
        d_angles = angles[:, 1:] - angles[:, :-1]
        flag_forward = np.hstack([np.zeros((len(angles), 1)), np.where(d_angles<np.full((len(angles), 1), -threshold), 1, 0)])
        flag_backward = np.hstack([np.zeros((len(angles), 1)), np.where(d_angles>np.full((len(angles), 1), threshold), -1, 0)])
        flag = np.cumsum(flag_forward, axis=1) + np.cumsum(flag_backward, axis=1)
        angles_corrected = angles + flag * full_angle
    return angles_corrected, flag

if __name__ == '__main__':
    print('---- test ----')
    #### line
    # t = np.linspace(0, 10, 20)
    # rng = np.random.default_rng(seed=0)
    # x = t + rng.uniform(-1, 1, 20)
    # y = x*2 + rng.uniform(-1, 1, 20)
    # fitter = MyFitting(np.vstack([x, y]).T)
    # fitter.lsm_for_line()
    # fitter.check()

    # ps = np.array([0.001, 10])
    # d, theta = calc_mindist_p2ellipse(ps, 2, 1, max_iter=100000)
    # print(d, np.degrees(theta))

    duration = 0.2
    fps = 10000
    num_frames = int(duration * fps) + 1
    rng = np.random.default_rng(seed=0)
    import sampledata_generator
    cage = sampledata_generator.SimpleCage(name='', PCD=50, ID=48, OD=52, width=10, num_pockets=8, num_markers=8, num_mesh=100, Dp=6.25, Dw=5.953)
    t = np.arange(num_frames) / fps
    # a = (1 - 0.2 * np.sin(2*np.pi*t*10)) * cage.PCD/2
    a = 0.9 * cage.PCD/2
    # a = 1 * cage.PCD/2
    # b = (1 + 0.1 * np.sin(2*np.pi*t*10)) * cage.PCD/2
    b = 2 * cage.PCD/2
    # b = 1 * cage.PCD/2
    cage.time_series_data2(fps=fps, duration=duration, omega_rot=40*np.pi, omega_rev=40*np.pi, r_rev=0, a=a, b=b, omega_deform=0, noise_type="normal", noise_max=0.01, p0_angle=np.pi/15)
    from mymods import mycoord
    transformer = mycoord.CoordTransformer3D(coordsys_name="cage_coordsys", local_origin=np.zeros((1, 3)), euler_angles=np.zeros(3), rot_order='zyx')
    points_zero = cage.make_cage_points(num_frames=1, num_points=8, x_value=5, a=25, b=25, deform_angle=0, transformer=transformer)[0][:, :, 1:]
    cut = 0
    markers= cage.p_markers_noise[:, cut:, 1:]
    markers_defect = cage.p_markers_noise[:, :, 1:].copy()
    markers_defect[500:1000, 0, 0] = np.nan

    xyr = np.zeros((num_frames, 3))
    for i in range(num_frames):
        xyr[i], info = lsm_for_circle(markers[i])
    # xyr_defect = np.zeros((num_frames, 3))
    # for i in range(num_frames):
    #     xyr_defect[i], info = lsm_for_circle(markers_defect[i], allow_nan=False)

    xyabtheta = np.zeros((num_frames, 5))
    for i in range(num_frames):
        xyabtheta[i], info = lsm_for_ellipse(markers[i], allow_nan=False)
    # xyabtheta_defect = np.zeros((num_frames, 5))
    # for i in range(num_frames):
    #     xyabtheta_defect[i], info = lsm_for_ellipse(markers_defect[i], allow_nan=False)

    # xyr2, info = lsm_for_circles(markers)
    # xyr2_defect, info = lsm_for_circles(markers_defect)
    # results = calc_elliptical_deformation(markers, points_zero)

    abcdef = np.zeros((num_frames, 6))
    xyabtheta_fitzgibbon = np.zeros((num_frames, 5))
    for i in range(num_frames):
        abcdef[i] = fitzgibbon_ellipse(markers[i], allow_nan=False)
        _xyabtheta = abcdef2xyabtheta(abcdef[i])
        xyabtheta_fitzgibbon[i] = _xyabtheta["center"][0], _xyabtheta["center"][1], _xyabtheta["axes"][0], _xyabtheta["axes"][1], _xyabtheta["angle"]


    datalist = [
        {"id": 0, "data": xyr[:, 0], "lw": 4, "c": 'g', "alpha": 0.2},
        {"id": 1, "data": xyr[:, 1], "lw": 4, "c": 'g', "alpha": 0.2},
        {"id": 2, "data": xyr[:, 2], "lw": 4, "c": 'g', "alpha": 0.4},
        # {"id": 0, "data": xyr_defect[:, 0], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 1, "data": xyr_defect[:, 1], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 2, "data": xyr_defect[:, 2], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 0, "data": xyr2[:, 0], "lw": 2, "c": 'r', "alpha": 0.8},
        # {"id": 1, "data": xyr2[:, 1], "lw": 2, "c": 'r', "alpha": 0.8},
        # {"id": 2, "data": xyr2[:, 2], "lw": 2, "c": 'r', "alpha": 0.8},
        # {"id": 0, "data": xyr2_defect[:, 0], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 1, "data": xyr2_defect[:, 1], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 2, "data": xyr2_defect[:, 2], "lw": 1, "c": 'b', "alpha": 0.8},

        {"id": 0, "data": xyabtheta[:, 0], "lw": 1, "c": 'b', "alpha": 1},
        {"id": 1, "data": xyabtheta[:, 1], "lw": 1, "c": 'b', "alpha": 1},
        {"id": 2, "data": xyabtheta[:, 2], "lw": 1, "c": 'b', "alpha": 1},
        # {"id": 2, "data": xyabtheta[:, 3], "lw": 1, "c": 'g', "alpha": 1},
        # {"id": 0, "data": xyabtheta_defect[:, 0], "lw": 8, "c": 'm', "alpha": 0.2},
        # {"id": 1, "data": xyabtheta_defect[:, 1], "lw": 8, "c": 'm', "alpha": 0.2},
        # {"id": 2, "data": xyabtheta_defect[:, 2], "lw": 8, "c": 'm', "alpha": 0.2},
        # {"id": 2, "data": xyabtheta_defect[:, 3], "lw": 8, "c": 'm', "alpha": 0.2},

        {"id": 0, "data": xyabtheta_fitzgibbon[:, 0], "lw": 1, "c": 'r', "alpha": 1},
        {"id": 1, "data": xyabtheta_fitzgibbon[:, 1], "lw": 1, "c": 'r', "alpha": 1},
        {"id": 2, "data": xyabtheta_fitzgibbon[:, 2], "lw": 1, "c": 'r', "alpha": 1},
        {"id": 2, "data": xyabtheta_fitzgibbon[:, 3], "lw": 1, "c": 'r', "alpha": 1},



        # {"id": 0, "data": results["diameters_norm"][:, 0], "lw": 1, "c": 'r', "alpha": 0.8},
        # {"id": 0, "data": results["diameters_norm"][:, 1], "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 0, "data": results["diameters_norm"][:, 2], "lw": 1, "c": 'g', "alpha": 0.8},
        # {"id": 1, "data": results["roundness"], "lw": 1, "c": 'r', "alpha": 0.8},
        # {"id": 2, "data": np.degrees(results["direction"][:, 0]), "lw": 1, "c": 'r', "alpha": 0.8},
        # {"id": 2, "data": np.degrees(results["direction"][:, 1]), "lw": 1, "c": 'b', "alpha": 0.8},
        # {"id": 2, "data": np.degrees(results["deformation_angle"]), "lw": 1, "c": 'k', "alpha": 0.8},
    ]
    f = np.arange(num_frames)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    # axs[0].set_ylim(40, 60)
    # axs[1].set_ylim(-1, 10)
    # axs[2].set_ylim(-200, 200)
    # axs[2].set_ylim(0, 30)
    for i in range(3):
        axs[i].set_xlim(0, num_frames)
    for i in range(len(datalist)):
        _d = datalist[i]
        axs[_d["id"]].plot(f, _d["data"], lw=_d["lw"], c=_d["c"], alpha=_d["alpha"])

    diff_center = xyr[:, :2] - xyabtheta_fitzgibbon[:, :2]
    error_center_circle = xyr[:, :2] - cage.p_cage[:, 1:]
    error_center_fitz = xyabtheta_fitzgibbon[:, :2] - cage.p_cage[:, 1:]
    error_a_circle = xyr[:, 2] - min(a, b)
    error_b_circle = xyr[:, 2] - max(a, b)
    error_a_fitz = xyabtheta_fitzgibbon[:, 2] - min(a, b)
    error_b_fitz = xyabtheta_fitzgibbon[:, 3] - max(a, b)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    # axs[0].plot(f, diff_center[:, 0], lw=1, c='k', alpha=0.4)
    axs[0].plot(f, error_center_circle[:, 0], lw=1, c='g', alpha=0.4)
    axs[0].plot(f, error_center_fitz[:, 0], lw=1, c='b', alpha=0.4)

    # axs[1].plot(f, diff_center[:, 1], lw=1, c='k', alpha=0.4)
    axs[1].plot(f, error_center_circle[:, 1], lw=1, c='g', alpha=0.4)
    axs[1].plot(f, error_center_fitz[:, 1], lw=1, c='b', alpha=0.4)

    axs[2].plot(f, error_a_circle, lw=1, c='g', alpha=0.4)
    axs[2].plot(f, error_b_circle, lw=1, c='g', alpha=0.4)
    axs[2].plot(f, error_a_fitz, lw=1, c='b', alpha=0.4)
    axs[2].plot(f, error_b_fitz, lw=1, c='b', alpha=0.4)

    axs[0].set(ylim=(-0.1, 0.1))
    axs[1].set(ylim=(-0.1, 0.1))
    axs[2].set(ylim=(-40, 40))


    f = 0
    node = np.linspace(0, 1, 100)
    x_circle = xyr[f, 2] * np.cos(2*np.pi*node) + xyr[f, 0]
    y_circle = xyr[f, 2] * np.sin(2*np.pi*node) + xyr[f, 1]



    def rotate_points(p, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])[np.newaxis, :, :]
        p = p[:, :, np.newaxis]
        p_rotated = R @ p
        return p_rotated

    x_ellipse = xyabtheta[f, 2] * np.cos(2*np.pi*node)
    y_ellipse = xyabtheta[f, 3] * np.sin(2*np.pi*node)
    ellipse = np.vstack([x_ellipse, y_ellipse]).T
    p = rotate_points(ellipse, xyabtheta[f, 4]).squeeze() + xyabtheta[f, :2][np.newaxis, :]

    # x_ellipse = xyabtheta_defect[f, 2] * np.cos(2*np.pi*node)
    # y_ellipse = xyabtheta_defect[f, 3] * np.sin(2*np.pi*node)
    # ellipse = np.vstack([x_ellipse, y_ellipse]).T
    # p_defect = rotate_points(ellipse, xyabtheta_defect[f, 4]).squeeze() + xyabtheta_defect[f, :2][np.newaxis, :]

    # abcdef = fitzgibbon_ellipse(markers[f], allow_nan=False)
    # ellipse_fitzgibbon = abcdef2xyabtheta(abcdef)
    x_ellipse_fitzgibbon = xyabtheta_fitzgibbon[f, 2] * np.cos(2*np.pi*node)
    y_ellipse_fitzgibbon = xyabtheta_fitzgibbon[f, 3] * np.sin(2*np.pi*node)
    markers_fitzgibbon = np.vstack([x_ellipse_fitzgibbon, y_ellipse_fitzgibbon]).T
    p_fitzgibbon = rotate_points(markers_fitzgibbon, xyabtheta_fitzgibbon[f, 4]).squeeze() + xyabtheta_fitzgibbon[f, :2][np.newaxis, :]

    print(f"circle center: {xyr[f, :2]}")
    print(f"ellipse center: {xyabtheta[f, :2]}")
    print(f"fitzgibbon ellipse center: {xyabtheta_fitzgibbon[f, :2]}")

    fig_trj, ax_trj = plt.subplots(figsize=(8, 8))
    ax_trj.set_aspect(1)
    ax_trj.grid()
    ax_trj.scatter(markers[f, 0, 0], markers[f, 0, 1], c='r', s=100, alpha=1)
    ax_trj.scatter(markers[f, 1, 0], markers[f, 1, 1], c='b', s=100, alpha=1)
    ax_trj.scatter(markers[f, 2:, 0], markers[f, 2:, 1], c='k', s=40, alpha=1)
    ax_trj.plot(x_circle, y_circle, lw=4, c='g', alpha=0.4)
    ax_trj.scatter(xyr[f, 0], xyr[f, 1], s=200, c='g', alpha=1, marker='x', zorder=100)
    ax_trj.plot(p[:, 0], p[:, 1], lw=1, c='b', alpha=1)
    ax_trj.scatter(xyabtheta[f, 0], xyabtheta[f, 1], s=200, c='b', alpha=0.2, marker='+', zorder=99)
    # ax_trj.plot(p_defect[:, 0], p_defect[:, 1], lw=8, c='m', alpha=0.2)
    # ax_trj.scatter(xyabtheta_defect[f, 0], xyabtheta_defect[f, 1], s=80, c='m', alpha=0.4)
    ax_trj.plot(p_fitzgibbon[f, 0], p_fitzgibbon[f, 1], lw=1, c='r', alpha=1)
    ax_trj.scatter(xyabtheta_fitzgibbon[f, 0], xyabtheta_fitzgibbon[f, 1], s=200, c='r', alpha=0.2, marker='o', zorder=99)

    xymax = max(np.abs(np.min(markers[f, :, :])), np.abs(np.max(markers[f, :, :]))) * 1.1
    ax_trj.set(xlim=(-xymax, xymax), ylim=(-xymax, xymax))

    plt.show()




