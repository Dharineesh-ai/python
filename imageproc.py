import logging
import cv2
import numpy as np
from typing import Optional,Tuple

logger = logging.getLogger(__name__)

def autofix_bgr_rgb(img_rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Auto-detect and fix probable BGR->RGB ordering."""
    if img_rgb is None:
        return None
    arr = np.ascontiguousarray(img_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return arr
    a = arr.astype(np.float32)
    r_mean = float(a[..., 0].mean())
    g_mean = float(a[..., 1].mean())
    b_mean = float(a[..., 2].mean())
    if b_mean > r_mean * 1.4 and b_mean - r_mean > 10:
        fixed = arr[..., [2, 1, 0]].copy()
        logger.warning("Auto-fix detected probable BGR ordering, converted to RGB")
        return fixed
    return arr

def ensure_uint8_rgb(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize to uint8 RGB, handling 2D/3D/4D inputs."""
    if img is None:
        return None
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        f = img.astype(np.float32)
        mn = float(np.min(f))
        mx = float(np.max(f))
        if mx > mn:
            f = (f - mn) / (mx - mn) * 255.0
        else:
            f = np.zeros_like(f, dtype=np.float32)
        img8 = np.clip(f, 0, 255).astype(np.uint8)
    else:
        img8 = img
    if img8.ndim == 2:
        rgb = np.stack([img8]*3, axis=2)
        return autofix_bgr_rgb(rgb)
    elif img8.ndim == 3:
        ch = img8.shape[2]
        if ch == 1:
            rgb = np.stack([img8[..., 0]]*3, axis=2)
            return autofix_bgr_rgb(rgb)
        if ch == 3:
            if (np.array_equal(img8[..., 0], img8[..., 1]) and 
                np.array_equal(img8[..., 1], img8[..., 2])):
                rgb = np.stack([img8[..., 0]]*3, axis=2)
                return autofix_bgr_rgb(rgb)
            return autofix_bgr_rgb(img8)
        if ch == 4:
            try:
                candidate = img8[..., 3].copy()
                if (np.array_equal(candidate[..., 0], candidate[..., 1]) and 
                    np.array_equal(candidate[..., 1], candidate[..., 2])):
                    return autofix_bgr_rgb(img8[..., 3])
                return autofix_bgr_rgb(candidate)
            except Exception:
                pass
            return autofix_bgr_rgb(img8[..., 3])
    return img8

def apply_window_level(
    frame: np.ndarray,
    window_min: float,
    window_max: float,
    data_min_max: Tuple[float, float] | None = None,
) -> np.ndarray:
    """
    ImageJ-like window/level.

    frame        : original image (float32 or uint16/uint8 etc).
    window_min   : lower display bound in data units.
    window_max   : upper display bound in data units.
    data_min_max : optional (min,max) of data; if None, computed from frame.
    """
    f = frame.astype(np.float32)
    if data_min_max is None:
        dmin = float(f.min())
        dmax = float(f.max())
    else:
        dmin, dmax = data_min_max

    # Clamp window to data range
    lo = max(window_min, dmin)
    hi = min(window_max, dmax)
    if hi <= lo:
        # avoid division by zero; show flat image
        return np.zeros_like(f, dtype=np.uint8)

    # Map [lo,hi] -> [0,255]
    f = (f - lo) * (255.0 / (hi - lo))
    f = np.clip(f, 0, 255)
    return f.astype(np.uint8)

def apply_color_selection(
    frame: np.ndarray, 
    color_norm: str, 
    ndim: int
) -> np.ndarray:
    """Apply Red/Green/Blue color selection to frame."""
    h_orig, w_orig = frame.shape[:2]
    if ndim == 2:
        if color_norm == "Red":
            rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            rgb_full[..., 0] = frame
        elif color_norm == "Green":
            rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            rgb_full[..., 1] = frame
        elif color_norm == "Blue":
            rgb_full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            rgb_full[..., 2] = frame
        else:
            rgb_full = np.stack([frame]*3, axis=2)
    else:  # 3D frame
        if frame.shape[2] == 3:
            if color_norm == "Red":
                rgb_full = np.zeros_like(frame)
                rgb_full[..., 0] = frame[..., 0]
            elif color_norm == "Green":
                rgb_full = np.zeros_like(frame)
                rgb_full[..., 1] = frame[..., 1]
            elif color_norm == "Blue":
                rgb_full = np.zeros_like(frame)
                rgb_full[..., 2] = frame[..., 2]
            else:
                rgb_full = frame[..., :3].copy()
        else:
            rgb_full = np.stack([frame[..., 0]]*3, axis=2) if ndim == 3 else np.stack([frame]*3, axis=2)
    return ensure_uint8_rgb(rgb_full)
