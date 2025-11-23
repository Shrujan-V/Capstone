"""
Tumor detection and distance mapping.
"""
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from . import config
from . import geometry


def compute_distance_to_tumor(tumor_mask):
    """
    Compute distance transform: for each pixel, distance to nearest tumor boundary.
    """
    inverted = (tumor_mask == 0).astype(np.uint8)
    distance_map = distance_transform_edt(inverted)
    return distance_map


def get_tumor_centroid_and_boundary(tumor_mask):
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    if len(contours) > 1:
        all_points = [c.squeeze(axis=1) for c in contours if len(c.squeeze(axis=1)) > 0]
        if not all_points: return None, None
        full_boundary = np.concatenate(all_points, axis=0)
    elif len(contours) == 1:
        full_boundary = contours[0].squeeze(axis=1)
    else:
        return None, None
        
    M = cv2.moments(tumor_mask)
    if M["m00"] == 0:
        return None, None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])
    
    return centroid, full_boundary


def detect_tumor(flair_img):
    """
    Detect tumor from FLAIR image.
    Returns: tumor_mask
    """
    _, tumor_mask_binary = cv2.threshold(flair_img, config.Config.TUMOR_THRESH, 255, cv2.THRESH_BINARY)
    tumor_mask_morphed = cv2.morphologyEx(tumor_mask_binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tumor_mask = geometry.filter_small_regions(tumor_mask_morphed, min_size=config.Config.MIN_AREA)
    return tumor_mask

