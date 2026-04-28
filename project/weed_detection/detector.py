# weed_detection/detector.py
# ============================================================
# HIGH-LEVEL WEED DETECTION API
# ============================================================
# This is the main entry point for weed detection integration.
# It takes a BGR image, runs segmentation, and returns a
# structured result dict with detection stats and annotated image.
# ============================================================

import cv2
import numpy as np
import base64
from weed_detection.segment import segment
from weed_detection.preprocessing import resize_keep_aspect


def run_weed_detection(img_bgr, method='color', threshold=0.12):
    """
    Run weed detection on a BGR image and return structured results.

    Args:
        img_bgr: Input image as BGR numpy array
        method: Detection method ('ndvi', 'color', 'texture', 'size_filter')
        threshold: Sensitivity threshold (0.0 - 1.0, lower = more sensitive)

    Returns:
        dict with keys:
            - weed_detected (bool): Whether weeds were found
            - weed_coverage_percent (float): % of image area identified as weed
            - weed_pixel_count (int): Number of weed pixels
            - total_pixels (int): Total image pixels
            - num_weed_regions (int): Number of distinct weed regions
            - method (str): Method used
            - threshold (float): Threshold used
            - confidence (str): Confidence level description
            - annotated_image_b64 (str): Base64-encoded JPEG of annotated image
            - mask_image_b64 (str): Base64-encoded JPEG of weed mask
    """
    # Resize for consistent processing
    img = resize_keep_aspect(img_bgr, max_side=512)

    # Run segmentation
    mask, aux = segment(img, method=method, threshold=threshold)

    # Ensure mask is uint8 binary (some methods return float)
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask * 255).astype(np.uint8)

    # --- Compute statistics ---
    total_pixels = mask.shape[0] * mask.shape[1]
    weed_pixels = int(np.sum(mask > 0))
    weed_coverage = round((weed_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0.0

    # Count distinct weed regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = len(contours)

    # Determine confidence level
    if weed_coverage > 15:
        confidence = "High"
    elif weed_coverage > 5:
        confidence = "Medium"
    elif weed_coverage > 0.5:
        confidence = "Low"
    else:
        confidence = "None"

    weed_detected = weed_coverage > 0.5  # Minimum threshold to call it a detection

    # --- Create annotated image ---
    annotated = img.copy()
    # Draw red overlay where weeds are detected
    red_overlay = np.zeros_like(annotated)
    red_overlay[:, :, 2] = mask  # Red channel
    annotated = cv2.addWeighted(annotated, 0.7, red_overlay, 0.5, 0)
    # Draw contours
    cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

    # Add status text
    status_text = f"WEED DETECTED ({weed_coverage:.1f}%)" if weed_detected else "NO WEED DETECTED"
    color = (0, 0, 255) if weed_detected else (0, 255, 0)
    cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(annotated, f"Regions: {num_regions} | Method: {method}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Encode images to base64 for JSON response
    _, annotated_buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(annotated_buf).decode('utf-8')

    # Create a colorized mask for display
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    _, mask_buf = cv2.imencode('.jpg', mask_colored, [cv2.IMWRITE_JPEG_QUALITY, 85])
    mask_b64 = base64.b64encode(mask_buf).decode('utf-8')

    return {
        'weed_detected': weed_detected,
        'weed_coverage_percent': weed_coverage,
        'weed_pixel_count': weed_pixels,
        'total_pixels': total_pixels,
        'num_weed_regions': num_regions,
        'method': method,
        'threshold': threshold,
        'confidence': confidence,
        'annotated_image_b64': annotated_b64,
        'mask_image_b64': mask_b64,
    }
