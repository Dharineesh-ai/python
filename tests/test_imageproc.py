import numpy as np
import pytest
from imageproc import ensure_uint8_rgb, autofix_bgr_rgb, apply_brightness_contrast

def test_ensure_uint8_rgb_2d():
    img = np.ones((10, 10), dtype=np.float32) * 128
    result = ensure_uint8_rgb(img)
    assert result.dtype == np.uint8
    assert result.shape == (10, 10, 3)
    assert np.all(result[..., 0] == 128)

def test_ensure_uint8_rgb_3d_grayscale():
    img = np.ones((10, 10, 1), dtype=np.uint8) * 100
    result = ensure_uint8_rgb(img)
    assert result.shape == (10, 10, 3)
    assert np.all(result == 100)

def test_autofix_bgr_rgb():
    # BGR image (blue heavy)
    bgr_img = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_img[..., 2] = 200  # Red channel high in BGR
    result = autofix_bgr_rgb(bgr_img)
    assert np.all(result[..., 0] == 200)  # Should swap to RGB

def test_brightness_contrast():
    frame = np.ones((10, 10), dtype=np.uint8) * 128
    result = apply_brightness_contrast(frame, brightness=50, contrast=50)
    assert np.all(result > 128)  # Should be brighter and more contrast
