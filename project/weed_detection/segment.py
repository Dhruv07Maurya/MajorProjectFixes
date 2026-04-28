# weed_detection/segment.py
# ============================================================
# Copied from WeedIoTNew/src/models/segment_stub.py
# Self-contained segmentation logic for weed detection.
# Imports are adjusted to use local package paths.
# ============================================================

import cv2
import numpy as np
from weed_detection.preprocessing import (
    compute_ndvi_from_bgr,
    threshold_mask_from_ndvi,
    detect_weeds_by_size_color,
    detect_weeds_texture_based
)


def segment(img_bgr, method='ndvi', threshold=0.12):
    """
    Segment WEEDS specifically from image.

    Methods:
    - 'ndvi': Simple vegetation detection (detects ALL plants)
    - 'color': HSV-based weed detection (size + color filtering)
    - 'texture': Texture-based weed detection (irregular patterns)
    - 'size_filter': Color + size-based weed detection

    Args:
        img_bgr: Input image in BGR format (OpenCV default)
        method: Segmentation method
        threshold: Threshold value for segmentation

    Returns:
        mask: Binary mask (255 = WEED, 0 = crop/soil)
        aux: Dictionary with auxiliary information
    """
    if method == 'ndvi':
        ndvi = compute_ndvi_from_bgr(img_bgr)
        mask = threshold_mask_from_ndvi(ndvi, thresh=threshold)
        aux = {'ndvi_mean': float(np.mean(ndvi)), 'note': 'Detects all vegetation, not just weeds'}

    elif method == 'color':
        min_area = int(50 * (1.0 - threshold))
        max_area = 5000
        mask = detect_weeds_by_size_color(img_bgr, min_area=min_area, max_area=max_area)
        aux = {'method': 'size_color', 'min_area': min_area, 'max_area': max_area}

    elif method == 'texture':
        mask = detect_weeds_texture_based(img_bgr)
        aux = {'method': 'texture_based'}

    elif method == 'size_filter':
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([25, 30, 30])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)

        min_area = 30
        max_area = 3000
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

        mask = filtered_mask
        aux = {'method': 'size_filtered'}

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ndvi', 'color', 'texture', or 'size_filter'")

    return mask, aux
