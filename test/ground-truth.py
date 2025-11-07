import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Define the bounding boxes for the tracts
tract_boxes = {
    "tract_1": [160, 110, 253, 213],
    "tract_2": [158, 234, 236, 425],
    "tract_3": [145, 429, 199, 592],
    "tract_4": [170, 415, 331, 487],
    "tract_5": [320, 424, 371, 569],
    "tract_6": [298, 249, 371, 421],
    "tract_7": [294, 108, 373, 213],
    "tract_8": [200, 198, 337, 266],
}

TUMOR_THRESH = 180
MIN_AREA = 300

def iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    if not boxA or not boxB:
        return 0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter) if (areaA + areaB - inter) else 0

def filter_small_regions(mask, min_size=300):
    """Filter out small connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered[labels == i] = 1
    return filtered

def get_bbox(mask):
    """Get bounding box from binary mask."""
    y, x = np.where(mask)
    if len(x) == 0:
        return None
    return [min(x), min(y), max(x), max(y)]

# Paths to the input folders
flair_folder = Path(r"D:\Capstone\dataset\main\flair-after")
output_csv_path = Path(r"D:\Capstone\results\recurrence.csv")

# Get all flair images
flair_files = list(flair_folder.glob("*.png"))

data_records = []
idx = 1

for flair_path in flair_files:
    image_id = flair_path.stem  # Use file name without extension as image_id
    print(f"\n🔹 Processing image {image_id}: {flair_path.name}")
    
    # Read the flair image
    flair_img = cv2.imread(str(flair_path), 0)
    
    # Threshold and process tumor mask
    _, tumor_mask = cv2.threshold(flair_img, TUMOR_THRESH, 255, cv2.THRESH_BINARY)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tumor_mask = filter_small_regions(tumor_mask, min_size=MIN_AREA)
    
    tumor_bbox = get_bbox(tumor_mask)
    
    # Calculate IoU for each tract and label only the tract with the max IoU
    tract_ious = {}
    for tract_id, tract_bbox in tract_boxes.items():
        tract_ious[tract_id] = iou(tumor_bbox, tract_bbox)
    
    # Determine which tract has the maximum IoU (if any)
    if tract_ious:
        # Get tract with max IoU (ties resolved by first occurrence)
        max_tract = max(tract_ious, key=lambda k: tract_ious[k])
        max_iou = tract_ious[max_tract]
    else:
        max_tract = None
        max_iou = 0

    for tract_id, tract_bbox in tract_boxes.items():
        iou_binary = 1 if (tract_id == max_tract and max_iou > 0) else 0
        data_records.append({
            "image_id": image_id,  # Use file name as image_id
            "tract_id": tract_id,
            "iou_binary": iou_binary,
            "iou": tract_ious[tract_id]  # Add IoU value for each tract
        })
    
    idx += 1

# Save results to CSV
df = pd.DataFrame(data_records)
df.to_csv(output_csv_path, index=False)
print(f"✅ Results saved to {output_csv_path}")