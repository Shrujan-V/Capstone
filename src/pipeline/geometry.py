"""
Geometry helper functions for bounding boxes and spatial calculations.
"""
import numpy as np
import cv2


def split_tract_box(name, box, num_splits=8, axis='vertical'):
    """
    Split a tract bounding box into multiple sub-tracts.
    """
    x1, y1, x2, y2 = box
    sub_tracts = {}
    
    if axis == 'vertical':
        height = y2 - y1
        if height < num_splits: num_splits = 1
        segment_h = height // num_splits
        if segment_h == 0: num_splits = 1
            
        for i in range(num_splits):
            sy1 = y1 + i * segment_h
            sy2 = y1 + (i + 1) * segment_h if i < num_splits - 1 else y2
            if sy2 > y2: sy2 = y2
            if i > 0 and sy1 >= sy2: continue
            sub_tracts[f"{name}_p{i+1}"] = [x1, sy1, x2, sy2]
    else: # horizontal
        width = x2 - x1
        if width < num_splits: num_splits = 1
        segment_w = width // num_splits
        if segment_w == 0: num_splits = 1

        for i in range(num_splits):
            sx1 = x1 + i * segment_w
            sx2 = x1 + (i + 1) * segment_w if i < num_splits - 1 else x2
            if sx2 > x2: sx2 = x2
            if i > 0 and sx1 >= sx2: continue
            sub_tracts[f"{name}_p{i+1}"] = [sx1, y1, sx2, y2]
            
    if not sub_tracts:
        sub_tracts[f"{name}_p1"] = [x1, y1, x2, y2]

    return sub_tracts


def get_bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_bbox(mask):
    y, x = np.where(mask)
    if len(x) == 0: return None
    return [min(x), min(y), max(x), max(y)]


def iou(boxA, boxB):
    if not boxA or not boxB: return 0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter) if (areaA + areaB - inter) else 0


def filter_small_regions(mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered[labels == i] = 255
    return filtered.astype(np.uint8)

