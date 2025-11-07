import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import networkx as nx

TUMOR_THRESH = 180
MIN_AREA = 300
DECAY = 0.5
TRACT_BRIGHTNESS_THRESHOLD = 40
SOFT_GRADIENT_DECAY = 0.03
TRACT_COLOR_BASE = np.array([255, 140, 0], dtype=np.float32)
OUTSIDE_COLOR = np.array([20, 20, 20], dtype=np.float32)

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

# Define tract groups
LEFT_TRACTS = ["tract_1", "tract_2", "tract_3"]
RIGHT_TRACTS = ["tract_5", "tract_6", "tract_7"]
MIDDLE_TRACTS = ["tract_4", "tract_8"]

def get_bbox_centroid(bbox):
    """Calculate centroid of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def build_tract_graph():
    """Build a graph with tracts as nodes and distances as edge weights."""
    G = nx.Graph()
    
    # Add all tracts as nodes with their centroids
    centroids = {}
    for name, bbox in tract_boxes.items():
        centroid = get_bbox_centroid(bbox)
        centroids[name] = centroid
        G.add_node(name, pos=centroid)
    
    # Define top and bottom tracts for each side
    # tract_8 is mid top, tract_4 is mid bottom
    # tract_1 is left top, tract_2 is left mid, tract_3 is left bottom
    # tract_7 is right top, tract_6 is right mid, tract_5 is right bottom
    
    # Add edges based on connectivity rules
    # 1. Connect left tracts to each other
    for i, tract_i in enumerate(LEFT_TRACTS):
        for tract_j in LEFT_TRACTS[i+1:]:
            dist = euclidean_distance(centroids[tract_i], centroids[tract_j])
            G.add_edge(tract_i, tract_j, weight=dist)
    
    # 2. Connect right tracts to each other
    for i, tract_i in enumerate(RIGHT_TRACTS):
        for tract_j in RIGHT_TRACTS[i+1:]:
            dist = euclidean_distance(centroids[tract_i], centroids[tract_j])
            G.add_edge(tract_i, tract_j, weight=dist)
    
    # 3. DO NOT connect middle tracts to each other (tract_4 and tract_8 are separate)
    
    # 4. Connect left/right TOP tracts to mid TOP (tract_8) only
    # tract_1 (left top) and tract_7 (right top) connect to tract_8 (mid top)
    top_tracts = ["tract_1", "tract_7"]
    for top_tract in top_tracts:
        dist = euclidean_distance(centroids[top_tract], centroids["tract_8"])
        G.add_edge(top_tract, "tract_8", weight=dist)
    
    # 5. Connect left/right BOTTOM tracts to mid BOTTOM (tract_4) only
    # tract_3 (left bottom) and tract_5 (right bottom) connect to tract_4 (mid bottom)
    bottom_tracts = ["tract_3", "tract_5"]
    for bottom_tract in bottom_tracts:
        dist = euclidean_distance(centroids[bottom_tract], centroids["tract_4"])
        G.add_edge(bottom_tract, "tract_4", weight=dist)
    
    return G, centroids

def compute_distance_to_bbox(x, y, bbox):
    """Compute distance from point to bounding box."""
    x1, y1, x2, y2 = bbox
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return np.sqrt(dx*dx + dy*dy)

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

def get_orientation(mask):
    """Get principal orientation vector from binary mask."""
    y, x = np.where(mask)
    if len(x) < 2:
        return None
    coords = np.column_stack((x, y))
    pca = PCA(n_components=1).fit(coords)
    vec = pca.components_[0]
    vec /= np.linalg.norm(vec)
    return vec

def get_tract_orientation_from_rgb(img_rgb, box):
    """Extract orientation from RGB tractography image."""
    x1, y1, x2, y2 = box
    crop = img_rgb[y1:y2, x1:x2, :]
    pixels = crop.reshape(-1, 3).astype(np.float32) / 255.0
    mask = np.linalg.norm(pixels, axis=1) > 0.1
    pixels = pixels[mask]
    if len(pixels) < 10:
        return None
    directions = pixels[:, [0, 1, 2]]
    directions -= directions.mean(axis=0)
    pca = PCA(n_components=1)
    pca.fit(directions)
    direction_vector = pca.components_[0]
    direction_vector /= np.linalg.norm(direction_vector)
    return direction_vector

def get_alignment(v1, v2):
    """Calculate alignment between two vectors."""
    if v1 is None or v2 is None:
        return 0
    return np.abs(np.dot(v1, v2[:2]))

def iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes."""
    if not boxA or not boxB:
        return 0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter) if (areaA + areaB - inter) else 0

def propagate_recurrence_scores(features, tumor_bbox, graph):
    """Propagate recurrence scores based on graph connectivity and tumor location."""
    # Check which tracts have tumor overlap
    tumor_in_middle = False
    for mid_tract in MIDDLE_TRACTS:
        if features[mid_tract]["iou"] > 0:
            tumor_in_middle = True
            break
    
    # Determine which side(s) the tumor is on
    tumor_on_left = any(features[tract]["iou"] > 0 for tract in LEFT_TRACTS)
    tumor_on_right = any(features[tract]["iou"] > 0 for tract in RIGHT_TRACTS)
    
    # If tumor is NOT in middle tracts, zero out scores for restricted tracts
    if not tumor_in_middle:
        # Zero out middle tracts
        for mid_tract in MIDDLE_TRACTS:
            features[mid_tract]["recurrence_score"] = 0
        
        # Zero out opposite side
        if tumor_on_left and not tumor_on_right:
            for right_tract in RIGHT_TRACTS:
                features[right_tract]["recurrence_score"] = 0
        elif tumor_on_right and not tumor_on_left:
            for left_tract in LEFT_TRACTS:
                features[left_tract]["recurrence_score"] = 0
    
    # Find tracts with non-zero scores to start propagation
    active_tracts = [name for name, feat in features.items() if feat["recurrence_score"] > 0]
    if not active_tracts:
        return features
    
    # Find the tract with maximum initial score among active tracts
    max_name = max(active_tracts, key=lambda x: features[x]["recurrence_score"])
    
    # Determine allowed propagation region
    if not tumor_in_middle:
        if tumor_on_left:
            allowed_tracts = set(LEFT_TRACTS)
        elif tumor_on_right:
            allowed_tracts = set(RIGHT_TRACTS)
        else:
            allowed_tracts = set(tract_boxes.keys())
    else:
        allowed_tracts = set(tract_boxes.keys())
    
    # Use Dijkstra-based propagation with decay
    visited = {max_name}
    priority_queue = [(0, max_name, features[max_name]["recurrence_score"])]
    
    while priority_queue:
        # Sort by cumulative distance
        priority_queue.sort(key=lambda x: x[0])
        cum_dist, current, current_score = priority_queue.pop(0)
        
        # Get neighbors in the graph
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
            
            # Skip if neighbor is not in allowed tracts
            if neighbor not in allowed_tracts:
                continue
            
            # Get edge distance
            edge_distance = graph[current][neighbor]['weight']
            new_cum_dist = cum_dist + edge_distance
            
            # Calculate decay based on distance
            decay_factor = np.exp(-DECAY * new_cum_dist / 100)  # Normalize distance
            increment = current_score * decay_factor
            
            features[neighbor]["recurrence_score"] += increment
            priority_queue.append((new_cum_dist, neighbor, features[neighbor]["recurrence_score"]))
            visited.add(neighbor)
    
    return features

# Build the tract graph once
tract_graph, tract_centroids = build_tract_graph()

data_records = []
tract_folder = Path(r"D:\Capstone\dataset\main\dti-before")
flair_folder = Path(r"D:\Capstone\dataset\main\flair-before")

tract_files = list(tract_folder.glob("*.png"))
flair_files = list(flair_folder.glob("*.png"))

idx = 1

for tract_path, flair_path in zip(tract_files, flair_files):
    print(f"\n🔹 Processing image {idx}")
    
    if not tract_path.exists() or not flair_path.exists():
        print(f"⚠️ Missing tract or flair image for {idx}, skipping.")
        idx += 1
        continue
    
    dti_bgr = cv2.imread(str(tract_path))
    dti_rgb = cv2.cvtColor(dti_bgr, cv2.COLOR_BGR2RGB)
    flair_img = cv2.imread(str(flair_path), 0)
    
    # Threshold and process tumor mask
    _, tumor_mask = cv2.threshold(flair_img, TUMOR_THRESH, 255, cv2.THRESH_BINARY)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tumor_mask = filter_small_regions(tumor_mask, min_size=MIN_AREA)
    
    tumor_bbox = get_bbox(tumor_mask)
    tumor_vec = get_orientation(tumor_mask)
    
    # Initialize features for all tracts
    features = {}
    h, w, _ = dti_rgb.shape
    
    for name, box in tract_boxes.items():
        tract_vec = get_tract_orientation_from_rgb(dti_rgb, box)
        alignment = get_alignment(tumor_vec, tract_vec)
        raw_iou = iou(tumor_bbox, box)
        weighted_iou = raw_iou * alignment
        
        features[name] = {
            "bbox": box,
            "alignment": round(alignment, 3),
            "iou": round(weighted_iou, 3),
            "recurrence_score": weighted_iou
        }
    
    # Propagate recurrence scores using graph-based approach
    features = propagate_recurrence_scores(features, tumor_bbox, tract_graph)
    
    # Normalize recurrence scores
    total_score = sum(feat["recurrence_score"] for feat in features.values())
    if total_score > 0:
        for feat in features.values():
            feat["recurrence_score"] /= total_score
    
    # Generate heatmap visualization with bounding boxes and recurrence scores
    gray_img = cv2.cvtColor(dti_rgb, cv2.COLOR_RGB2GRAY)
    bright_mask = gray_img > TRACT_BRIGHTNESS_THRESHOLD
    output_img = np.zeros((h, w, 3), dtype=np.float32)
    bright_indices = np.argwhere(bright_mask)

    for y, x in bright_indices:
        weights = []
        scores = []
        for name, feat in features.items():
            bbox = feat["bbox"]
            score = feat["recurrence_score"]
            dist = compute_distance_to_bbox(x, y, bbox)
            w = np.exp(-SOFT_GRADIENT_DECAY * dist)
            weights.append(w)
            scores.append(score)
        
        weights = np.array(weights)
        scores = np.array(scores)
        if weights.sum() > 0:
            weighted_score = (weights * scores).sum() / weights.sum()
        else:
            weighted_score = 0
        
        alpha = np.clip(weighted_score * 1.5, 0, 1)
        output_img[y, x] = alpha * TRACT_COLOR_BASE + (1 - alpha) * OUTSIDE_COLOR

    output_img_clipped = np.clip(output_img, 0, 255).astype(np.uint8)

    # Draw bounding boxes and recurrence scores on the heatmap
    for name, feat in features.items():
        bbox = feat["bbox"]
        score = feat["recurrence_score"]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(output_img_clipped, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        cv2.putText(
            output_img_clipped,
            f"{score * 100:.2f}%",  # Recurrence score as percentage
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # Save the updated heatmap
    output_filename = flair_path.stem + ".png"  # Ensure the file has a .png extension
    output_filepath = Path("D:\\Capstone\\results\\heatmaps") / output_filename
    cv2.imwrite(str(output_filepath), cv2.cvtColor(output_img_clipped, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved {output_filename} with bounding boxes and recurrence scores.")
    
    # Store data records
    for name, feat in features.items():
        data_records.append({
            "image_id": idx,
            "tract": name,
            "iou": feat["iou"],
            "alignment": feat["alignment"],
            "recurrence_score": round(feat["recurrence_score"] * 100, 2)
        })
    
    idx += 1

# Save results to CSV
df = pd.DataFrame(data_records)
df.to_csv("D:\\Capstone\\results\\tumor_tract_recurrence_graph_based.csv", index=False)
print("✅ All images and CSV saved with graph-based propagation.")