# weed_detection/__init__.py
# ============================================================
# WEED DETECTION INTEGRATION MODULE
# ============================================================
# This package was integrated from the WeedIoTNew project.
# It provides OpenCV-based weed detection using multiple methods
# (NDVI, color filtering, texture analysis, size filtering).
#
# Usage:
#   from weed_detection import run_weed_detection
#   result = run_weed_detection(bgr_image, method='color', threshold=0.12)
# ============================================================

from weed_detection.detector import run_weed_detection

__all__ = ['run_weed_detection']
