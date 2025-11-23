"""
Tract mask extraction and centerline computation.
"""
import numpy as np
import cv2
from sklearn.decomposition import PCA
from . import config


def extract_tract_mask(dti_rgb, box):
    """Extract binary mask of tract fibers from DTI RGB."""
    x1, y1, x2, y2 = box
    h, w = dti_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    crop = dti_rgb[y1:y2, x1:x2, :]
    gray = np.mean(crop, axis=2).astype(np.uint8)
    _, tract_binary = cv2.threshold(gray, config.Config.TRACT_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    mask[y1:y2, x1:x2] = tract_binary
    return mask


def compute_tract_centerline(tract_mask):
    """
    Compute approximate centerline of tract using distance transform skeleton.
    Returns: coordinates of centerline pixels (y, x) format
    """
    if np.count_nonzero(tract_mask) == 0:
        return np.array([])
    
    dist_transform = cv2.distanceTransform(tract_mask, cv2.DIST_L2, 5)
    threshold = dist_transform.max() * 0.5
    centerline_mask = (dist_transform > threshold).astype(np.uint8)
    centerline_coords = np.column_stack(np.where(centerline_mask > 0))
    return centerline_coords


def get_tract_orientation_from_rgb(img_rgb, box):
    x1, y1, x2, y2 = box
    crop = img_rgb[y1:y2, x1:x2, :]
    pixels = crop.reshape(-1, 3).astype(np.float32) / 255.0
    mask = np.linalg.norm(pixels, axis=1) > 0.1
    pixels = pixels[mask]
    if len(pixels) < 10: return None
    directions = pixels[:, [0, 1, 2]]
    directions -= directions.mean(axis=0)
    pca = PCA(n_components=1)
    pca.fit(directions)
    direction_vector = pca.components_[0]
    direction_vector /= np.linalg.norm(direction_vector)
    return direction_vector

