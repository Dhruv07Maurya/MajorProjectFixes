# weed_detection/preprocessing.py
# ============================================================
# Copied from WeedIoTNew/src/preprocessing.py
# Self-contained preprocessing functions for weed detection.
# No external dependencies beyond OpenCV, NumPy, SciPy.
# ============================================================

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def compute_ndvi_from_bgr(img_bgr):
    """
    Compute a vegetation index from BGR image.
    Returns vegetation index in range [-1, 1].
    """
    img = img_bgr.astype(float)
    b, g, r = cv2.split(img)

    nir_proxy = g  # Green channel (vegetation)
    red = r        # Red channel (soil)

    denom = (nir_proxy + red + 1e-6)
    ndvi = (nir_proxy - red) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi


def threshold_mask_from_ndvi(ndvi, thresh=0.05):
    """
    Return binary mask (255 = vegetation, 0 = background).
    This detects ALL vegetation (crops + weeds).
    """
    mask = (ndvi > thresh).astype('uint8') * 255
    return mask


def detect_weeds_by_row_crops(img_bgr, row_spacing=50, row_tolerance=15):
    """
    Detect weeds in row-crop agriculture.
    Assumes crops are planted in regular rows.
    Anything between rows is considered a weed.
    """
    ndvi = compute_ndvi_from_bgr(img_bgr)
    veg_mask = (ndvi > 0.05).astype('uint8') * 255

    h, w = veg_mask.shape
    density = np.sum(veg_mask > 0, axis=0) / h

    smoothed = gaussian_filter1d(density, sigma=10)

    peaks, _ = find_peaks(smoothed, distance=row_spacing // 2, prominence=0.1)

    crop_row_mask = np.zeros_like(veg_mask)
    for peak in peaks:
        x_start = max(0, peak - row_tolerance)
        x_end = min(w, peak + row_tolerance)
        crop_row_mask[:, x_start:x_end] = 255

    weed_mask = cv2.bitwise_and(veg_mask, cv2.bitwise_not(crop_row_mask))

    return weed_mask


def detect_weeds_by_size_color(img_bgr, min_area=50, max_area=5000):
    """
    Detect weeds using size and color differences from crops.
    Assumes crops are larger, more uniform green than weeds.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    crop_lower = np.array([40, 80, 60])
    crop_upper = np.array([75, 255, 255])
    crop_mask = cv2.inRange(hsv, crop_lower, crop_upper)

    weed_lower = np.array([20, 30, 30])
    weed_upper = np.array([90, 255, 255])
    potential_weed = cv2.inRange(hsv, weed_lower, weed_upper)

    weed_mask = cv2.bitwise_and(potential_weed, cv2.bitwise_not(crop_mask))

    contours, _ = cv2.findContours(weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(weed_mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask


def detect_weeds_texture_based(img_bgr):
    """
    Detect weeds using texture analysis.
    Crops have uniform texture, weeds have irregular patterns.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    ndvi = compute_ndvi_from_bgr(img_bgr)
    veg_mask = (ndvi > 0.05).astype('uint8') * 255

    kernel_size = 15
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
    variance = sqr_mean - mean ** 2

    veg_pixels = variance[veg_mask > 0]
    if veg_pixels.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    texture_thresh = np.percentile(veg_pixels, 60)
    high_variance_mask = (variance > texture_thresh).astype('uint8') * 255

    weed_mask = cv2.bitwise_and(veg_mask, high_variance_mask)

    return weed_mask


def resize_keep_aspect(img, max_side=512):
    """Resize image keeping aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return new
