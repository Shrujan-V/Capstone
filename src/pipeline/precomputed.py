"""
Precomputed data structure to avoid repeated heavy computations.
"""
import numpy as np
from . import config
from . import geometry
from . import tract_processing
from . import tumor_processing


class PrecomputedData:
    """
    Container for all precomputed data per image.
    All heavy operations are computed once and stored here.
    """
    def __init__(self):
        # Tract data (keyed by sub-tract name)
        self.tract_masks = {}           # dict[str, np.ndarray] - binary masks
        self.tract_centerlines = {}     # dict[str, np.ndarray] - centerline coords (y, x)
        self.tract_orientations = {}    # dict[str, np.ndarray] - PCA orientation vectors
        
        # Tumor data
        self.tumor_mask = None          # np.ndarray - binary mask
        self.tumor_centroid = None      # np.ndarray - [cx, cy]
        self.tumor_boundary = None      # np.ndarray - boundary points
        self.distance_map = None         # np.ndarray - distance transform
        
        # Tract segmentation
        self.sub_tract_boxes = {}       # dict[str, list] - bounding boxes
        self.parent_map = {}            # dict[str, str] - sub-tract -> parent mapping


def precompute_image_data(dti_rgb, flair_img):
    """
    Precompute all heavy operations once per image.
    
    Args:
        dti_rgb: RGB DTI image
        flair_img: Grayscale FLAIR image
        
    Returns:
        PrecomputedData object with all precomputed data, or None if tumor not found
    """
    data = PrecomputedData()
    
    # 1. Create sub-tract segmentation
    for name, box in config.Config.ORIGINAL_TRACTS.items():
        sub_tracts = geometry.split_tract_box(name, box, num_splits=6, axis='vertical')
        data.sub_tract_boxes.update(sub_tracts)
        for sub_name in sub_tracts:
            data.parent_map[sub_name] = name
    
    # 2. Detect tumor and compute distance map
    data.tumor_mask = tumor_processing.detect_tumor(flair_img)
    
    tumor_bbox = geometry.get_bbox(data.tumor_mask)
    if tumor_bbox is None:
        return None
    
    data.distance_map = tumor_processing.compute_distance_to_tumor(data.tumor_mask)
    data.tumor_centroid, data.tumor_boundary = tumor_processing.get_tumor_centroid_and_boundary(data.tumor_mask)
    
    if data.tumor_centroid is None:
        return None
    
    # 3. Precompute all tract masks and centerlines
    print("   Precomputing tract masks and centerlines...")
    for name, box in data.sub_tract_boxes.items():
        # Extract tract mask
        tract_mask = tract_processing.extract_tract_mask(dti_rgb, box)
        data.tract_masks[name] = tract_mask
        
        # Compute centerline
        tract_centerline = tract_processing.compute_tract_centerline(tract_mask)
        data.tract_centerlines[name] = tract_centerline
        
        # Compute PCA-based orientation (if tract has pixels)
        if np.count_nonzero(tract_mask) > 0:
            orientation = tract_processing.get_tract_orientation_from_rgb(dti_rgb, box)
            data.tract_orientations[name] = orientation
        else:
            data.tract_orientations[name] = None
    
    return data

