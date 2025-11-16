import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import networkx as nx
import sys
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

TUMOR_THRESH = 180
MIN_AREA = 300
DECAY = 0.5
TRACT_BRIGHTNESS_THRESHOLD = 40

# --- NEW: Bulge Analysis Weights ---
# (Tune these to adjust model sensitivity)
W_BULGE = 0.4      # How much to prioritize a "bulge"
W_ALIGNMENT = 0.3  # How much to prioritize alignment with the tract
W_PROXIMITY = 0.2  # How much to prioritize closeness to the tract
W_DENSITY = 0.1    # How much to prioritize local tumor "solidity"

# --- NEW: Blob Heatmap Constants ---
BLOB_HEATMAP_SIGMA = 40     # Controls the "spread" of the heatmap blob
BLOB_CONTOUR_THRESH = 0.25  # At what % to draw the white contour line
BLOB_HOTSPOT_OFFSET = 30    # How far from the tumor to project the hotspot

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

LEFT_TRACTS = ["tract_1", "tract_2", "tract_3"]
RIGHT_TRACTS = ["tract_5", "tract_6", "tract_7"]
MIDDLE_TRACTS = ["tract_4", "tract_8"]

# ==========================================
# 2. GEOMETRY & GRAPH HELPER FUNCTIONS
# ==========================================

def get_bbox_centroid(bbox):
    """Calculate centroid of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def build_tract_graph():
    """Build a graph with tracts as nodes and distances as edge weights."""
    G = nx.Graph()
    centroids = {}
    for name, bbox in tract_boxes.items():
        centroid = get_bbox_centroid(bbox)
        centroids[name] = centroid
        G.add_node(name, pos=centroid)
    
    for i, tract_i in enumerate(LEFT_TRACTS):
        for tract_j in LEFT_TRACTS[i+1:]:
            dist = euclidean_distance(centroids[tract_i], centroids[tract_j])
            G.add_edge(tract_i, tract_j, weight=dist)
    
    for i, tract_i in enumerate(RIGHT_TRACTS):
        for tract_j in RIGHT_TRACTS[i+1:]:
            dist = euclidean_distance(centroids[tract_i], centroids[tract_j])
            G.add_edge(tract_i, tract_j, weight=dist)
    
    top_tracts = ["tract_1", "tract_7"]
    for top_tract in top_tracts:
        dist = euclidean_distance(centroids[top_tract], centroids["tract_8"])
        G.add_edge(top_tract, "tract_8", weight=dist)
    
    bottom_tracts = ["tract_3", "tract_5"]
    for bottom_tract in bottom_tracts:
        dist = euclidean_distance(centroids[bottom_tract], centroids["tract_4"])
        G.add_edge(bottom_tract, "tract_4", weight=dist)
    
    return G, centroids

def get_bbox(mask):
    y, x = np.where(mask)
    if len(x) == 0: return None
    return [min(x), min(y), max(x), max(y)]

def get_tract_orientation_from_rgb(img_rgb, box):
    """Extract orientation from RGB tractography image."""
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

def filter_small_regions(mask, min_size=300):
    """Filter out small connected components from a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered[labels == i] = 1
    return filtered

def iou(boxA, boxB):
    if not boxA or not boxB: return 0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter) if (areaA + areaB - inter) else 0

# ==========================================
# 3. NEW: BULGE & DENSITY ANALYSIS FUNCTIONS
# ==========================================

def get_tumor_centroid_and_boundary(tumor_mask):
    """
    (Step 1 & 2) Extracts the full tumor contour and its center of mass.
    """
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    full_boundary = np.concatenate(contours, axis=0).squeeze(axis=1)
    
    M = cv2.moments(tumor_mask)
    if M["m00"] == 0:
        return None, None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])
    
    return centroid, full_boundary

def analyze_boundary_features(tumor_mask, centroid, boundary_points):
    """
    (Step 3 & 4) Analyzes each boundary point for bulge, density, and intensity.
    """
    if centroid is None or boundary_points is None:
        return []
    
    radial_vectors = boundary_points - centroid
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    
    # (Step 3) Normalize by average distance to get a "bulge score"
    avg_distance = np.mean(radial_distances)
    bulge_scores = radial_distances / avg_distance
    
    analyzed_points = []
    h, w = tumor_mask.shape
    
    for i, point in enumerate(boundary_points):
        x, y = point
        
        # (Step 4) Local Density
        # Sample 5px inwards along the *negative* radial vector
        if radial_distances[i] == 0: continue
        patch_center = point - (radial_vectors[i] / radial_distances[i]) * 5
        px, py = int(patch_center[0]), int(patch_center[1])
        
        patch = tumor_mask[max(0, py-5):min(h, py+5),
                           max(0, px-5):min(w, px+5)]
        
        local_density = np.mean(patch) / 255.0  # Score 0-1
        
        analyzed_points.append({
            "point": point,
            "bulge_score": bulge_scores[i],
            "density_score": local_density,
            "radial_vector": radial_vectors[i]
        })
        
    return analyzed_points

def get_dominant_infiltration_features(analyzed_boundary_points, tract_centroid, tract_vec):
    """
    (Step 5 & 6) Finds the *single best* infiltration vector for *one* tract.
    Scores boundary points based on bulge, alignment, proximity, and density.
    """
    if not analyzed_boundary_points:
        return None, 0
    
    # (Step 5) Filter by Proximity
    point_coords = np.array([p["point"] for p in analyzed_boundary_points])
    distances = np.linalg.norm(point_coords - tract_centroid, axis=1)
    
    # Get the indices of the 50 closest points
    k = min(50, len(analyzed_boundary_points))
    proximal_indices = np.argsort(distances)[:k]
    
    # (Step 6) Combine features
    best_score = -1
    best_vector = None
    
    for idx in proximal_indices:
        point_data = analyzed_boundary_points[idx]
        
        # 1. Proximity Score (inverse distance, normalized)
        prox_score = (1.0 / (distances[idx] + 1))
        
        # 2. Alignment Score (Bulge vector vs. Tract vector)
        bulge_vec = point_data["radial_vector"]
        if tract_vec is None or np.linalg.norm(bulge_vec) == 0 or np.linalg.norm(tract_vec[:2]) == 0:
            align_score = 0
        else:
            norm_bulge = bulge_vec / np.linalg.norm(bulge_vec)
            norm_tract = tract_vec[:2] / np.linalg.norm(tract_vec[:2])
            align_score = np.abs(np.dot(norm_bulge, norm_tract))
        
        # 3. Bulge Score
        bulge_score = point_data["bulge_score"]
        
        # 4. Density Score
        density_score = point_data["density_score"]

        # --- Final Weighted Score ---
        final_score = (
            (bulge_score * W_BULGE) +
            (align_score * W_ALIGNMENT) +
            (prox_score * W_PROXIMITY) +
            (density_score * W_DENSITY)
        )
        
        if final_score > best_score:
            best_score = final_score
            best_vector = {
                "source_point": point_data["point"],
                "direction": point_data["radial_vector"],
                "score": final_score
            }

    return best_vector, best_score

# ==========================================
# 4. NEW: BLOB VISUALIZATION FUNCTIONS
# ==========================================

def create_blob_heatmap(dti_rgb, hotspots_data, shape):
    """
    (Step 7) Generates a recurrence heatmap with "dark red" hotspots
    enclosed within a smooth, well-defined contour blob.
    
    hotspots_data = list of dicts: [{"source": (x,y), "direction": (dx,dy), "score": s}]
    """
    h, w = shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # 1. "Paint" the Gaussians for each hotspot
    for hotspot in hotspots_data:
        score = hotspot["score"]
        source_point = hotspot["source"]
        direction = hotspot["direction"]
        
        if score > 0 and np.linalg.norm(direction) > 0:
            # Project the hotspot *away* from the tumor
            norm_dir = direction / np.linalg.norm(direction)
            hotspot_x = int(source_point[0] + norm_dir[0] * BLOB_HOTSPOT_OFFSET)
            hotspot_y = int(source_point[1] + norm_dir[1] * BLOB_HOTSPOT_OFFSET)
            
            # Place a "dot" of intensity=score at the hotspot
            if 0 <= hotspot_y < h and 0 <= hotspot_x < w:
                heatmap[hotspot_y, hotspot_x] = float(score)

    # 2. Apply a massive Gaussian blur to create the "blob"
    heatmap = gaussian_filter(heatmap, sigma=BLOB_HEATMAP_SIGMA)
    
    # 3. Normalize the blurred map
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    
    # 4. Colorize (COLORMAP_JET makes high values red)
    heatmap_viz = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_JET)

    # 5. Blend with the original DTI
    alpha = 0.6 # How transparent the heatmap is
    output_img = cv2.addWeighted(dti_rgb, 1 - alpha, heatmap_color, alpha, 0)
    
    # 6. Draw the smooth, well-defined contour
    _, thresh_map = cv2.threshold(heatmap_viz, int(BLOB_CONTOUR_THRESH * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_img, contours, -1, (255, 255, 255), 2) # White contour

    return output_img

def add_simple_annotations(output_img, features):
    """Adds just the tract boxes and final scores."""
    for name, feat in features.items():
        if feat["recurrence_score"] > 0.01:
            x1, y1, x2, y2 = feat["bbox"]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            score_text = f"{feat['recurrence_score']*100:.1f}%"
            cv2.rectangle(output_img, (x1, y1-20), (x1+70, y1), (0, 0, 0), -1)
            cv2.putText(output_img, score_text, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output_img

# ==========================================
# 5. GRAPH PROPAGATION LOGIC (Unchanged)
# ==========================================

def propagate_recurrence_scores(features, graph):
    """Propagate recurrence scores based on graph connectivity and tumor location."""
    
    tumor_in_middle = False
    for mid_tract in MIDDLE_TRACTS:
        if features[mid_tract].get("iou", 0) > 0:
            tumor_in_middle = True
            break
            
    tumor_on_left = any(features[tract].get("iou", 0) > 0 for tract in LEFT_TRACTS)
    tumor_on_right = any(features[tract].get("iou", 0) > 0 for tract in RIGHT_TRACTS)
    
    if not tumor_in_middle:
        for mid_tract in MIDDLE_TRACTS:
            features[mid_tract]["recurrence_score"] = 0
            
        if tumor_on_left and not tumor_on_right:
            for right_tract in RIGHT_TRACTS:
                features[right_tract]["recurrence_score"] = 0
        elif tumor_on_right and not tumor_on_left:
            for left_tract in LEFT_TRACTS:
                features[left_tract]["recurrence_score"] = 0
    
    active_tracts = [name for name, feat in features.items() if feat["recurrence_score"] > 0]
    if not active_tracts:
        return features
    
    max_name = max(active_tracts, key=lambda x: features[x]["recurrence_score"])
    
    if not tumor_in_middle:
        if tumor_on_left: allowed_tracts = set(LEFT_TRACTS)
        elif tumor_on_right: allowed_tracts = set(RIGHT_TRACTS)
        else: allowed_tracts = set(tract_boxes.keys())
    else:
        allowed_tracts = set(tract_boxes.keys())
    
    visited = {max_name}
    priority_queue = [(0, max_name, features[max_name]["recurrence_score"])]
    
    while priority_queue:
        priority_queue.sort(key=lambda x: x[0])
        cum_dist, current, current_score = priority_queue.pop(0)
        
        for neighbor in graph.neighbors(current):
            if neighbor in visited: continue
            if neighbor not in allowed_tracts: continue
            
            edge_distance = graph[current][neighbor]['weight']
            new_cum_dist = cum_dist + edge_distance
            
            decay_factor = np.exp(-DECAY * new_cum_dist / 100)
            increment = current_score * decay_factor
            
            features[neighbor]["recurrence_score"] += increment
            priority_queue.append((new_cum_dist, neighbor, features[neighbor]["recurrence_score"]))
            visited.add(neighbor)
    
    return features

# ==========================================
# 6. MAIN PROCESSING LOOP (HEAVILY UPDATED)
# ==========================================

tract_graph, tract_centroids = build_tract_graph()

data_records = []
# Use relative paths for portability
workspace_root = Path(__file__).parent.parent
tract_folder = workspace_root / "dataset" / "main" / "dti-before"
flair_folder = workspace_root / "dataset" / "main" / "flair-before"
results_folder = workspace_root / "results"
heatmaps_folder = results_folder / "heatmaps"

results_folder.mkdir(parents=True, exist_ok=True)
heatmaps_folder.mkdir(parents=True, exist_ok=True)

tract_files = sorted(list(tract_folder.glob("*.png")))
flair_files = sorted(list(flair_folder.glob("*.png")))

print(f"📁 Found {len(tract_files)} DTI files and {len(flair_files)} FLAIR files")
if len(tract_files) == 0 or len(flair_files) == 0:
    print("⚠️ ERROR: No image files found!")
    print(f"   DTI folder: {tract_folder}")
    print(f"   FLAIR folder: {flair_folder}")
    sys.exit(1)

if len(tract_files) != len(flair_files):
    print(f"⚠️ WARNING: Mismatch in file counts ({len(tract_files)} vs {len(flair_files)})")

idx = 1

for tract_path, flair_path in zip(tract_files, flair_files):
    print(f"\n🔹 Processing image {idx}: {flair_path.name}")
    
    if not tract_path.exists() or not flair_path.exists():
        print(f"⚠️ Missing tract or flair image for {idx}, skipping.")
        idx += 1
        continue
    
    dti_bgr = cv2.imread(str(tract_path))
    dti_rgb = cv2.cvtColor(dti_bgr, cv2.COLOR_BGR2RGB)
    flair_img = cv2.imread(str(flair_path), 0)
    
    if dti_bgr is None or flair_img is None:
        print(f"⚠️ Failed to load images for {idx}, skipping.")
        idx += 1
        continue
    
    # --- A. Tumor Detection ---
    _, tumor_mask = cv2.threshold(flair_img, TUMOR_THRESH, 255, cv2.THRESH_BINARY)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tumor_mask = filter_small_regions(tumor_mask, min_size=MIN_AREA)
    print(f"   Found {np.count_nonzero(tumor_mask)} tumor pixels")
    
    tumor_bbox = get_bbox(tumor_mask)
    if tumor_bbox is None:
        print("   No tumor found, skipping.")
        idx+=1
        continue
        
    # --- NEW: Step 1-4: Analyze Tumor Boundary ---
    tumor_centroid, tumor_boundary_points = get_tumor_centroid_and_boundary(tumor_mask)
    if tumor_centroid is None:
        print("   No tumor centroid found, skipping.")
        idx+=1
        continue

    print("   Analyzing tumor boundary features...")
    analyzed_boundary_points = analyze_boundary_features(
        tumor_mask, tumor_centroid, tumor_boundary_points
    )
    
    # --- B. NEW: Initial Feature Extraction (Bulge Analysis) ---
    print("   Calculating dominant infiltration vectors...")
    features = {}
    dominant_vectors_dict = {} # To store data for visualization
    
    for name, box in tract_boxes.items():
        tract_centroid = get_bbox_centroid(box)
        tract_vec = get_tract_orientation_from_rgb(dti_rgb, box)
        
        # (Step 5 & 6) Find the single best "bulge" aimed at this tract
        dominant_vector_data, initial_score = get_dominant_infiltration_features(
            analyzed_boundary_points,
            tract_centroid,
            tract_vec
        )
        
        features[name] = {
            "bbox": box,
            "initial_score": initial_score,      # Save the raw bulge score
            "recurrence_score": initial_score,   # This one will be propagated
            "iou": iou(tumor_bbox, box)          # Needed for propagation logic
        }
        
        if dominant_vector_data:
            dominant_vectors_dict[name] = dominant_vector_data

    # --- C. Graph Propagation (Unchanged) ---
    print("   Propagating risk scores through graph...")
    features = propagate_recurrence_scores(features, tract_graph)
    
    # Normalize scores
    total_score = sum(feat["recurrence_score"] for feat in features.values())
    if total_score > 0:
        for feat in features.values():
            feat["recurrence_score"] /= total_score

    # --- D. Prep Data for Visualization ---
    # Use the FINAL propagated scores to build the heatmap data
    final_hotspots_for_viz = []
    for name, feat in features.items():
        final_score = feat["recurrence_score"]
        # Only create a hotspot if it's "at risk" AND had a dominant vector
        if final_score > 0.01 and name in dominant_vectors_dict:
            vector_data = dominant_vectors_dict[name]
            final_hotspots_for_viz.append({
                "source": vector_data["source_point"],
                "direction": vector_data["direction"],
                "score": final_score # Use the FINAL score for heatmap intensity
            })

    # --- E. Generate NEW "Blob" Visualization ---
    print("   Generating blob heatmap...")
    output_img = create_blob_heatmap(dti_rgb, final_hotspots_for_viz, dti_rgb.shape)
    
    # Add tract box annotations on top
    output_img = add_simple_annotations(output_img, features)

    # --- F. Save Results ---
    output_filename = flair_path.stem + "_bulge_prediction.png"
    output_filepath = heatmaps_folder / output_filename
    cv2.imwrite(str(output_filepath), cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved {output_filename}")
    
    # Store data records
    for name, feat in features.items():
        data_records.append({
            "image_id": idx,
            "tract": name,
            "iou": round(feat["iou"], 3),
            "initial_bulge_score": round(feat["initial_score"] * 100, 2),
            "final_recurrence_score": round(feat["recurrence_score"] * 100, 2)
        })
    
    idx += 1

# Save CSV
if data_records:
    df = pd.DataFrame(data_records)
    csv_path = results_folder / "bulge_analysis_recurrence.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n✅ CSV saved with {len(data_records)} records to {csv_path}")
else:
    print("\n⚠️ No data records to save. Check if images were processed successfully.")

print(f"\n🎉 Processing complete! Results saved to {heatmaps_folder}")