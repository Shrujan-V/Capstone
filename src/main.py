import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import sys

# --- NEW IMPORTS ---
from skimage.morphology import skeletonize
from skimage.feature import local_binary_pattern
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import minmax_scale
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA


# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

TUMOR_THRESH = 180
MIN_AREA = 300
DECAY = 0.5

# --- 1. Boundary Analysis & Feature Weights ---
LBP_RADIUS = 3
LBP_POINTS = LBP_RADIUS * 8
CURVATURE_SMOOTHING = 21  # Kernel size for smoothing contour
W_BULGE = 0.3        # Convexity defect & curvature
W_AGGRESSIVENESS = 0.4 # Texture & Gradient
W_ALIGNMENT = 0.2    # Alignment with tract
W_PROXIMITY = 0.1    # 1 / distance

# --- 2. Infiltration Vector Extraction ---
CLUSTER_EPS = 15.0  # DBSCAN: Max distance between points in a cluster
CLUSTER_MIN_SAMPLES = 3 # DBSCAN: Min points to form a cluster
TOP_N_VECTORS_PER_TRACT = 3 # Keep top 3 vectors per tract

# --- 3. Heatmap & Visualization (MODIFIED) ---
HEATMAP_INITIAL_OUTWARD_OFFSET = 30 # NEW: How far *outside* the tumor the heatmap begins
HEATMAP_SIGMA = 35           # Anisotropic blur "spread"
HEATMAP_STEP_SIZE = 10       # How far to project hotspot points *beyond the initial offset*
HEATMAP_STEPS = 5            # How many points to project
HEATMAP_DECAY = 0.7          # Decay for each hotspot step
CONTOUR_THRESHOLD = 0.3      # % brightness for drawing contour
CONTOUR_SMOOTHNESS = 0.001   # Douglas-Peucker simplification factor

# Tract definitions (unchanged)
tract_boxes = {
    "tract_1": [160, 110, 253, 213], "tract_2": [158, 234, 236, 425],
    "tract_3": [145, 429, 199, 592], "tract_4": [170, 415, 331, 487],
    "tract_5": [320, 424, 371, 569], "tract_6": [298, 249, 371, 421],
    "tract_7": [294, 108, 373, 213], "tract_8": [200, 198, 337, 266],
}
LEFT_TRACTS = ["tract_1", "tract_2", "tract_3"]
RIGHT_TRACTS = ["tract_5", "tract_6", "tract_7"]
MIDDLE_TRACTS = ["tract_4", "tract_8"]

# ==========================================
# 2. ADVANCED TUMOR & TRACT MODELING
# ==========================================

def get_tumor_geometry(tumor_mask):
    """
    Step 1: Extracts contour, skeleton, hull, and convexity defects.
    """
    if np.count_nonzero(tumor_mask) == 0:
        return None, None, None, None, None

    # Skeleton (Medial Axis)
    skeleton = skeletonize(tumor_mask > 0).astype(np.uint8)
    skeleton_points = np.argwhere(skeleton > 0)[:, ::-1] # (x, y) format
    if len(skeleton_points) == 0:
        # Fallback to centroid if skeletonization fails
        M = cv2.moments(tumor_mask)
        if M["m00"] == 0: return None, None, None, None, None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        skeleton_points = np.array([[cx, cy]])
        
    skeleton_tree = cKDTree(skeleton_points)
    
    # Contour
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None
    
    # Get the single largest contour
    contour = max(contours, key=cv2.contourArea).squeeze(1)
    
    # Convex Hull & Defects
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    hull_points = contour[hull_indices.squeeze()]
    
    defects = None
    if len(contour) > 3: # Need > 3 points for defects
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
        except Exception as e:
            print(f"   [Debug] Could not compute convexity defects: {e}")
            
    return contour, skeleton_tree, hull_points, defects, skeleton_points

def compute_boundary_curvature(contour, ksize):
    """
    Computes curvature using first and second derivatives of the contour.
    """
    # Smooth the contour
    contour_smooth = cv2.GaussianBlur(contour.astype(np.float32), (ksize, ksize), 0)
    
    # Get derivatives
    dx = np.gradient(contour_smooth[:, 0])
    dy = np.gradient(contour_smooth[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula: k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    numerator = (dx * ddy) - (dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2) + 1e-9
    
    curvature = numerator / denominator
    return curvature

def compute_feature_maps(flair_img):
    """
    Pre-computes full-image feature maps for efficient sampling.
    """
    # Gradient Magnitude
    grad_x = cv2.Sobel(flair_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(flair_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Laplacian of Gaussian (LoG)
    log_map = cv2.Laplacian(cv2.GaussianBlur(flair_img, (3, 3), 0), cv2.CV_64F)
    
    # Local Binary Pattern (Texture)
    lbp_map = local_binary_pattern(flair_img, LBP_POINTS, LBP_RADIUS, 'uniform')
    
    return {
        "grad_mag": minmax_scale(grad_mag.ravel(), feature_range=(0, 1)).reshape(grad_mag.shape),
        "log_map": minmax_scale(np.abs(log_map).ravel(), feature_range=(0, 1)).reshape(log_map.shape),
        "lbp_map": minmax_scale(lbp_map.ravel(), feature_range=(0, 1)).reshape(lbp_map.shape)
    }

def analyze_boundary_points(contour, skeleton_tree, defects, feature_maps):
    """
    Step 2: Computes a rich feature vector for every boundary point.
    """
    # Pre-calculate curvature for all points
    curvatures = compute_boundary_curvature(contour, CURVATURE_SMOOTHING)
    
    # Map convexity defects to a lookup table for fast access
    defect_map = {}
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0] # start, end, far, depth
            defect_map[f] = d / 256.0 # Store normalized depth at the 'far' index
    
    boundary_data = []
    
    for i, point in enumerate(contour):
        x, y = point
        
        # 1. Geometry Features
        dist, closest_skel_idx = skeleton_tree.query(point)
        skeleton_origin = skeleton_tree.data[closest_skel_idx]
        radial_vec = point - skeleton_origin
        radial_dist = np.linalg.norm(radial_vec)
        
        curvature = curvatures[i]
        bulge_score = max(0, curvature) # Only positive (outward) curvature
        
        # Add convexity defect score
        defect_depth = defect_map.get(i, 0)
        bulge_score = (bulge_score * 0.5) + (defect_depth * 0.5)

        # 2. Aggressiveness Features (Sample from pre-computed maps)
        grad = feature_maps["grad_mag"][y, x]
        log = feature_maps["log_map"][y, x]
        lbp = feature_maps["lbp_map"][y, x]
        
        aggressiveness_score = (grad * 0.4) + (log * 0.2) + (lbp * 0.4)
        
        boundary_data.append({
            "point": point,
            "radial_vec": radial_vec if radial_dist > 0 else np.array([0, 1]),
            "bulge_score": bulge_score,
            "aggressiveness_score": aggressiveness_score
        })
        
    return boundary_data

def create_tract_interaction_maps(tract_boxes, dti_rgb, shape):
    """
    Step 3: Creates tract proximity and direction field maps.
    """
    interaction_maps = {}
    h, w, _ = dti_rgb.shape
    
    for name, box in tract_boxes.items():
        # Proximity Map (Distance Transform)
        tract_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(tract_mask, (box[0], box[1]), (box[2], box[3]), 255, -1)
        # Invert mask: 0 inside tract, >0 outside
        dist_map = cv2.distanceTransform(cv2.bitwise_not(tract_mask), cv2.DIST_L2, 3)
        
        # Direction Field (PCA vector)
        tract_vec_3d = get_tract_orientation_from_rgb(dti_rgb, box)
        if tract_vec_3d is None:
            tract_vec_2d = np.array([0.0, 0.0])
        else:
            tract_vec_2d = tract_vec_3d[:2] # Use x, y components
            if np.linalg.norm(tract_vec_2d) > 0:
                tract_vec_2d /= np.linalg.norm(tract_vec_2d) # Normalize
        
        interaction_maps[name] = {
            "dist_map": dist_map,
            "vector": tract_vec_2d
        }
    return interaction_maps

# ==========================================
# 3. DOMINANT INFILTRATION VECTOR (DIV) EXTRACTION
# ==========================================

def extract_dominant_infiltration_vectors(boundary_data, interaction_maps):
    """
    Step 4 & 5: Fuses all features, clusters, and extracts top DIVs per tract.
    """
    all_divs = {}
    
    for tract_name, tract_maps in interaction_maps.items():
        tract_vec = tract_maps["vector"]
        dist_map = tract_maps["dist_map"]
        
        scored_points = []
        for i, data in enumerate(boundary_data):
            point = data["point"]
            x, y = point
            
            # 1. Proximity Score
            prox_score = 1.0 / (dist_map[y, x] + 1.0)
            
            # 2. Alignment Score
            radial_vec = data["radial_vec"]
            norm_radial = radial_vec / (np.linalg.norm(radial_vec) + 1e-9)
            align_score = max(0, np.dot(norm_radial, tract_vec)) # Cosine similarity
            
            # 3. Bulge Score
            bulge_score = data["bulge_score"]
            
            # 4. Aggressiveness Score
            agg_score = data["aggressiveness_score"]

            # --- Weighted Fusion Model ---
            final_score = (
                (bulge_score * W_BULGE) +
                (agg_score * W_AGGRESSIVENESS) +
                (align_score * W_ALIGNMENT) +
                (prox_score * W_PROXIMITY)
            )
            
            if final_score > 0:
                scored_points.append({
                    "point": point,
                    "score": final_score,
                    "direction": norm_radial, # Store normalized vector
                })
        
        if not scored_points:
            continue
            
        # --- Cluster High-Scoring Vectors ---
        scores = np.array([p["score"] for p in scored_points])
        score_thresh = np.percentile(scores, 75)
        
        high_score_data = [p for p in scored_points if p["score"] >= score_thresh]
        if not high_score_data:
            continue
            
        high_score_points = np.array([p["point"] for p in high_score_data])
        
        clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(high_score_points)
        labels = clustering.labels_
        unique_labels = set(labels)
        
        tract_divs = []
        for k in unique_labels:
            if k == -1: continue # Skip noise points
            
            cluster_indices = [i for i, label in enumerate(labels) if label == k]
            cluster_data = [high_score_data[i] for i in cluster_indices]
            
            best_in_cluster = max(cluster_data, key=lambda x: x["score"])
            tract_divs.append(best_in_cluster)
            
        tract_divs.sort(key=lambda x: x["score"], reverse=True)
        all_divs[tract_name] = tract_divs[:TOP_N_VECTORS_PER_TRACT]
        
    return all_divs

# ==========================================
# 4. ANISOTROPIC HEATMAP VISUALIZATION (MODIFIED)
# ==========================================

def generate_anisotropic_heatmap(shape, div_map_with_final_scores, tumor_mask):
    """
    MODIFIED: Generates a recurrence heatmap that projects *outward* from the tumor
    and explicitly excludes the tumor body.
    """
    h, w = shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    for tract_name, divs in div_map_with_final_scores.items():
        for div in divs:
            final_score = div["score"]
            source_point = div["point"]
            direction = div["direction"]
            
            if final_score <= 0.01: # Skip very low scores
                continue

            # NEW: Project the *initial* hotspot origin outward from the tumor boundary
            initial_projected_point = (source_point + direction * HEATMAP_INITIAL_OUTWARD_OFFSET).astype(int)
            
            # Project multiple points along the direction from the initial projected point
            for i in range(HEATMAP_STEPS):
                dist = (i + 1) * HEATMAP_STEP_SIZE
                px = int(initial_projected_point[0] + direction[0] * dist)
                py = int(initial_projected_point[1] + direction[1] * dist)
                
                step_score = final_score * (HEATMAP_DECAY ** i)
                
                if 0 <= py < h and 0 <= px < w:
                    heatmap[py, px] = max(heatmap[py, px], step_score)

    # Apply Gaussian blur
    if np.max(heatmap) > 0:
        heatmap = gaussian_filter(heatmap, sigma=HEATMAP_SIGMA)
        heatmap = (heatmap / np.max(heatmap)) # Normalize
    
    # NEW: Explicitly mask out the tumor region from the heatmap
    heatmap[tumor_mask > 0] = 0 # Set heatmap values to 0 inside the tumor

    return heatmap

def create_final_visualization(dti_rgb, heatmap, div_map, tract_features):
    """
    MODIFIED: Blends heatmap, draws smooth contours, and adds annotations.
    """
    h, w, _ = dti_rgb.shape
    
    # 1. Create Red-Orange-Yellow Colormap
    heatmap_viz = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_HOT)
    
    # 2. Blend - Blend only where heatmap has non-zero values
    alpha_map = np.expand_dims(heatmap * 0.7, axis=2) # 0.7 for heatmap opacity
    output_img = (1 - alpha_map) * dti_rgb + alpha_map * heatmap_color
    output_img = np.clip(output_img, 0, 255).astype(np.uint8) # Ensure correct type and range
    
    # 3. Extract and Smooth Contours
    _, thresh_map = cv2.threshold(heatmap_viz, int(CONTOUR_THRESHOLD * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        smooth_contours = [cv2.approxPolyDP(c, CONTOUR_SMOOTHNESS * cv2.arcLength(c, True), True) for c in contours]
        cv2.drawContours(output_img, smooth_contours, -1, (255, 255, 255), 2) # White contour
    
    # 4. Add Annotations (Tract Scores)
    for name, feat in tract_features.items():
        score = feat["recurrence_score"]
        if score > 0.01:
            x1, y1, x2, y2 = feat["bbox"]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            score_text = f"{score*100:.1f}%"
            cv2.rectangle(output_img, (x1, y1-20), (x1+60, y1), (0, 0, 0), -1)
            cv2.putText(output_img, score_text, (x1+3, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    # 5. Add Annotations (DIVs)
    # Draw the top 1 DIV for each at-risk tract
    for name, divs in div_map.items():
        if tract_features[name]["recurrence_score"] > 0.01 and divs:
            top_div = divs[0] # Get the best one
            pt = top_div["point"]
            vec = top_div["direction"]
            
            # NEW: Draw the arrow from *outside* the tumor, matching heatmap origin
            start_pt = (pt + vec * HEATMAP_INITIAL_OUTWARD_OFFSET).astype(int)
            end_pt = (start_pt + vec * 40).astype(int) # Draw a 40px arrow from start_pt
            
            cv2.arrowedLine(output_img, tuple(start_pt), tuple(end_pt), (255, 0, 255), 2) # Magenta arrow
    
    return output_img

# ==========================================
# 5. ENHANCED GRAPH PROPAGATION (MODIFIED)
# ==========================================

def get_tract_orientation_from_rgb(img_rgb, box): # (Duplicating for use in propagation)
    x1, y1, x2, y2 = box
    crop = img_rgb[y1:y2, x1:x2, :]
    pixels = crop.reshape(-1, 3).astype(np.float32) / 255.0
    mask = np.linalg.norm(pixels, axis=1) > 0.1
    pixels = pixels[mask]
    if len(pixels) < 10: return None
    pca = PCA(n_components=1).fit(pixels[:, [0, 1, 2]] - pixels[:, [0, 1, 2]].mean(axis=0))
    vec = pca.components_[0] / np.linalg.norm(pca.components_[0])
    return vec

def propagate_recurrence_scores(features, graph, dti_rgb):
    """
    Step 5: Enhanced propagation with anisotropic decay based on tract alignment.
    """
    # Pre-calculate all tract vectors for alignment check
    tract_vectors = {}
    for name, feat in features.items():
        vec_3d = get_tract_orientation_from_rgb(dti_rgb, feat["bbox"])
        tract_vectors[name] = vec_3d[:2] if vec_3d is not None else np.array([0, 0])

    # --- Standard propagation logic (iou checks, etc) ---
    tumor_in_middle = any(features[tract]["iou"] > 0 for tract in MIDDLE_TRACTS)
    tumor_on_left = any(features[tract]["iou"] > 0 for tract in LEFT_TRACTS)
    tumor_on_right = any(features[tract]["iou"] > 0 for tract in RIGHT_TRACTS)
    
    if not tumor_in_middle:
        for mid_tract in MIDDLE_TRACTS:
            features[mid_tract]["recurrence_score"] = 0
        if tumor_on_left and not tumor_on_right:
            for right_tract in RIGHT_TRACTS: features[right_tract]["recurrence_score"] = 0
        elif tumor_on_right and not tumor_on_left:
            for left_tract in LEFT_TRACTS: features[left_tract]["recurrence_score"] = 0
    
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
            
            # --- ANISOTROPIC DECAY ---
            vec_current = tract_vectors[current]
            vec_neighbor = tract_vectors[neighbor]
            
            if np.linalg.norm(vec_current) == 0 or np.linalg.norm(vec_neighbor) == 0:
                alignment = 0.5 # Neutral alignment if vector is missing
            else:
                alignment = (np.abs(np.dot(vec_current, vec_neighbor)) + 1) / 2 # Range [0.5, 1]
            
            decay_factor = np.exp(-DECAY * new_cum_dist / 100)
            anisotropic_decay = decay_factor * alignment # Modulate by alignment
            
            increment = current_score * anisotropic_decay
            
            features[neighbor]["recurrence_score"] += increment
            priority_queue.append((new_cum_dist, neighbor, features[neighbor]["recurrence_score"]))
            visited.add(neighbor)
    
    return features

# ==========================================
# 6. MAIN PROCESSING LOOP (NEW ORCHESTRATION)
# ==========================================

# --- Helper functions that were at the top ---
def get_bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def build_tract_graph():
    G = nx.Graph()
    centroids = {}
    for name, bbox in tract_boxes.items():
        centroid = get_bbox_centroid(bbox)
        centroids[name] = centroid
        G.add_node(name, pos=centroid)
    for i, tract_i in enumerate(LEFT_TRACTS):
        for tract_j in LEFT_TRACTS[i+1:]: G.add_edge(tract_i, tract_j, weight=euclidean_distance(centroids[tract_i], centroids[tract_j]))
    for i, tract_i in enumerate(RIGHT_TRACTS):
        for tract_j in RIGHT_TRACTS[i+1:]: G.add_edge(tract_i, tract_j, weight=euclidean_distance(centroids[tract_i], centroids[tract_j]))
    for top_tract in ["tract_1", "tract_7"]: G.add_edge(top_tract, "tract_8", weight=euclidean_distance(centroids[top_tract], centroids["tract_8"]))
    for bottom_tract in ["tract_3", "tract_5"]: G.add_edge(bottom_tract, "tract_4", weight=euclidean_distance(centroids[bottom_tract], centroids["tract_4"]))
    return G, centroids
def get_bbox(mask):
    y, x = np.where(mask)
    if len(x) == 0: return None
    return [min(x), min(y), max(x), max(y)]
def filter_small_regions(mask, min_size=300):
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

# --- Start Main ---
print("🚀 Starting Advanced Recurrence Pipeline...")
tract_graph, tract_centroids = build_tract_graph()

data_records = []
# Use robust relative path to find project root
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
    print(f"⚠️ ERROR: No image files found! Searched folders:")
    print(f"   DTI: {tract_folder}")
    print(f"   FLAIR: {flair_folder}")
    sys.exit(1)

idx = 1
for tract_path, flair_path in zip(tract_files, flair_files):
    print(f"\n🔹 Processing image {idx}: {flair_path.name}")
    
    dti_bgr = cv2.imread(str(tract_path))
    dti_rgb = cv2.cvtColor(dti_bgr, cv2.COLOR_BGR2RGB)
    flair_img = cv2.imread(str(flair_path), 0)
    
    if dti_bgr is None or flair_img is None:
        print(f"⚠️ Failed to load images, skipping.")
        idx += 1
        continue
    
    # --- Step A: Tumor Detection ---
    _, tumor_mask = cv2.threshold(flair_img, TUMOR_THRESH, 255, cv2.THRESH_BINARY)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tumor_mask = filter_small_regions(tumor_mask, min_size=MIN_AREA)
    
    if np.count_nonzero(tumor_mask) < MIN_AREA:
        print("   No tumor found after filtering, skipping.")
        idx+=1
        continue
    
    # --- Step B: Tumor Geometry Modeling ---
    print("   1. Modeling tumor geometry...")
    contour, skel_tree, hull, defects, skel_points = get_tumor_geometry(tumor_mask)
    if contour is None:
        print("   Failed to model tumor geometry, skipping.")
        idx+=1
        continue

    # --- Step C: Boundary Feature Analysis ---
    print("   2. Analyzing boundary features...")
    feature_maps = compute_feature_maps(flair_img)
    boundary_data = analyze_boundary_points(contour, skel_tree, defects, feature_maps)
    
    # --- Step D: Tract Interaction Modeling ---
    print("   3. Modeling tract interactions...")
    interaction_maps = create_tract_interaction_maps(tract_boxes, dti_rgb, dti_rgb.shape)
    
    # --- Step E: Dominant Infiltration Vector (DIV) Extraction ---
    print("   4. Extracting Dominant Infiltration Vectors...")
    all_divs = extract_dominant_infiltration_vectors(boundary_data, interaction_maps)
    
    # --- Step F: Graph Propagation ---
    print("   5. Propagating risk scores...")
    features = {}
    for name, box in tract_boxes.items():
        initial_score = sum(div["score"] for div in all_divs.get(name, []))
        features[name] = {
            "bbox": box,
            "recurrence_score": initial_score,
            "iou": iou(get_bbox(tumor_mask), box)
        }
        
    features = propagate_recurrence_scores(features, tract_graph, dti_rgb)
    
    # Normalize final scores
    total_score = sum(feat["recurrence_score"] for feat in features.values())
    if total_score > 0:
        for feat in features.values():
            feat["recurrence_score"] /= total_score
    
    # --- Step G: Heatmap & Visualization ---
    print("   6. Generating final heatmap and visualization...")
    
    divs_with_final_scores = {}
    for name, feat in features.items():
        if feat["recurrence_score"] > 0.01 and name in all_divs:
            for div in all_divs[name]:
                div["score"] *= feat["recurrence_score"] # Scale by final tract risk
            divs_with_final_scores[name] = all_divs[name]

    # MODIFIED: Pass the tumor_mask to the heatmap generation
    heatmap = generate_anisotropic_heatmap(dti_rgb.shape[:2], divs_with_final_scores, tumor_mask)
    
    output_img = create_final_visualization(dti_rgb, heatmap, all_divs, features)
    
    # --- Step H: Save Output ---
    output_filename = flair_path.stem + "_advanced_prediction.png"
    output_filepath = heatmaps_folder / output_filename
    cv2.imwrite(str(output_filepath), cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved {output_filename}")
    
    # Store data records
    for name, feat in features.items():
        data_records.append({
            "image_id": idx,
            "tract": name,
            "final_recurrence_score": round(feat["recurrence_score"] * 100, 2)
        })
    
    idx += 1

# Save CSV
if data_records:
    df = pd.DataFrame(data_records)
    csv_path = results_folder / "advanced_recurrence_analysis.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n✅ CSV saved with {len(data_records)} records to {csv_path}")
else:
    print("\n⚠️ No data records to save. Check if images were processed successfully.")

print(f"\n🎉 Processing complete! Results saved to {heatmaps_folder}")