import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import networkx as nx
import sys
from scipy.ndimage import gaussian_filter, distance_transform_edt

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

TUMOR_THRESH = 180
MIN_AREA = 300
DECAY = 0.5
TRACT_BRIGHTNESS_THRESHOLD = 40

# SOFT CONTACT: Distance-based risk stratification
DIST_VERY_HIGH_RISK = 2   # pixels
DIST_HIGH_RISK = 5        # pixels
DIST_MODERATE_RISK = 10   # pixels
DIST_NO_RISK = 15         # beyond this = zero contribution

# Risk weights by distance tier
RISK_WEIGHT_VERY_HIGH = 1.0
RISK_WEIGHT_HIGH = 0.7
RISK_WEIGHT_MODERATE = 0.4
RISK_WEIGHT_LOW = 0.1

# Directional filtering
ANGULAR_THRESHOLD = 45  # degrees - max angle between radial vector and tract direction

# Bulge Analysis Weights (distance-aware)
W_PROXIMITY = 0.4     # Dominates - how close to tract
W_DIRECTION = 0.3     # Is radial vector pointing toward tract?
W_BULGE = 0.2         # Bulge magnitude
W_DENSITY = 0.1       # Local tumor density

# Blob Heatmap Constants
BLOB_HEATMAP_SIGMA = 55
BLOB_CORE_SIGMA = 18
BLOB_CONTOUR_THRESH = 0.12
BLOB_HOTSPOT_OFFSET = 25

original_tracts = {
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
# 2. GEOMETRY & GRAPH HELPERS
# ==========================================

def split_tract_box(name, box, num_splits=4, axis='vertical'):
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

def build_tract_graph():
    G = nx.Graph()
    centroids = {}
    for name, bbox in original_tracts.items():
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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered[labels == i] = 255
    return filtered.astype(np.uint8)

def iou(boxA, boxB):
    if not boxA or not boxB: return 0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter) if (areaA + areaB - inter) else 0

# ==========================================
# 3. TRACT MASK EXTRACTION
# ==========================================

def extract_tract_mask(dti_rgb, box):
    """Extract binary mask of tract fibers from DTI RGB."""
    x1, y1, x2, y2 = box
    h, w = dti_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    crop = dti_rgb[y1:y2, x1:x2, :]
    gray = np.mean(crop, axis=2).astype(np.uint8)
    _, tract_binary = cv2.threshold(gray, TRACT_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    
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

# ==========================================
# 4. SOFT CONTACT: DISTANCE-BASED RISK
# ==========================================

def compute_distance_to_tumor(tumor_mask):
    """
    Compute distance transform: for each pixel, distance to nearest tumor boundary.
    """
    inverted = (tumor_mask == 0).astype(np.uint8)
    distance_map = distance_transform_edt(inverted)
    return distance_map

def compute_tract_proximity_score(tract_mask, distance_map):
    """
    Compute soft-contact score based on proximity to tumor.
    """
    if np.count_nonzero(tract_mask) == 0:
        return float('inf'), 0.0, "none"
    
    tract_distances = distance_map[tract_mask > 0]
    min_distance = np.min(tract_distances)
    
    weights = np.zeros_like(tract_distances)
    weights[tract_distances <= DIST_VERY_HIGH_RISK] = RISK_WEIGHT_VERY_HIGH
    weights[(tract_distances > DIST_VERY_HIGH_RISK) & (tract_distances <= DIST_HIGH_RISK)] = RISK_WEIGHT_HIGH
    weights[(tract_distances > DIST_HIGH_RISK) & (tract_distances <= DIST_MODERATE_RISK)] = RISK_WEIGHT_MODERATE
    weights[(tract_distances > DIST_MODERATE_RISK) & (tract_distances <= DIST_NO_RISK)] = RISK_WEIGHT_LOW
    
    proximity_score = np.mean(weights) if len(weights) > 0 else 0.0
    
    if min_distance <= DIST_VERY_HIGH_RISK: risk_tier = "very_high"
    elif min_distance <= DIST_HIGH_RISK: risk_tier = "high"
    elif min_distance <= DIST_MODERATE_RISK: risk_tier = "moderate"
    elif min_distance <= DIST_NO_RISK: risk_tier = "low"
    else: risk_tier = "none"
    
    return min_distance, proximity_score, risk_tier

def find_nearest_tract_point(boundary_point, tract_centerline):
    """
    Find the nearest point on tract centerline to given boundary point.
    """
    if len(tract_centerline) == 0:
        return None, float('inf')
    
    bp_yx = np.array([boundary_point[1], boundary_point[0]])
    distances = np.linalg.norm(tract_centerline - bp_yx, axis=1)
    min_idx = np.argmin(distances)
    nearest_yx = tract_centerline[min_idx]
    nearest_xy = np.array([nearest_yx[1], nearest_yx[0]])
    
    return nearest_xy, distances[min_idx]

# ==========================================
# 5. DIRECTIONAL FILTERING
# ==========================================

def check_radial_alignment(radial_vector, boundary_point, tract_centerline):
    """
    Check if radial vector points TOWARD the tract.
    """
    if len(tract_centerline) == 0:
        return 0.0, False
    
    nearest_tract_point, dist = find_nearest_tract_point(boundary_point, tract_centerline)
    
    if nearest_tract_point is None or dist > DIST_NO_RISK:
        return 0.0, False
    
    to_tract_vector = nearest_tract_point - boundary_point
    
    if np.linalg.norm(to_tract_vector) == 0 or np.linalg.norm(radial_vector) == 0:
        return 0.0, False
    
    norm_radial = radial_vector / np.linalg.norm(radial_vector)
    norm_to_tract = to_tract_vector / np.linalg.norm(to_tract_vector)
    
    dot_product = np.clip(np.dot(norm_radial, norm_to_tract), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_product))
    
    alignment_score = max(0, 1.0 - (angle_deg / 90.0))
    is_aligned = angle_deg <= ANGULAR_THRESHOLD
    
    return alignment_score, is_aligned

# ==========================================
# 6. TUMOR BOUNDARY ANALYSIS
# ==========================================

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

def analyze_boundary_features_for_tract(tumor_mask, tumor_centroid, boundary_points, 
                                       tract_centerline, distance_map):
    """
    Analyze tumor boundary points specifically in relation to ONE tract.
    """
    if tumor_centroid is None or boundary_points is None or len(boundary_points) == 0:
        return []
    
    radial_vectors = boundary_points - tumor_centroid
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    
    avg_distance = np.mean(radial_distances[radial_distances > 0])
    if np.isnan(avg_distance) or avg_distance == 0: avg_distance = 1
    
    bulge_scores = radial_distances / avg_distance
    
    analyzed_points = []
    h, w = tumor_mask.shape
    
    for i, point in enumerate(boundary_points):
        x, y = point
        
        if y >= distance_map.shape[0] or x >= distance_map.shape[1]:
            continue
        
        _, dist_to_tract = find_nearest_tract_point(point, tract_centerline)
        
        if dist_to_tract > DIST_NO_RISK:
            continue
        
        alignment_score, is_aligned = check_radial_alignment(
            radial_vectors[i], point, tract_centerline
        )
        
        if not is_aligned:
            continue
        
        if radial_distances[i] == 0:
            continue
        
        patch_center = point - (radial_vectors[i] / radial_distances[i]) * 5
        px, py = int(patch_center[0]), int(patch_center[1])
        
        if px < 0 or py < 0 or px >= w or py >= h:
            continue
        
        patch = tumor_mask[max(0, py-5):min(h, py+5),
                           max(0, px-5):min(w, px+5)]
        
        local_density = np.mean(patch) / 255.0
        proximity_score = 1.0 / (dist_to_tract + 1.0)
        
        analyzed_points.append({
            "point": point,
            "bulge_score": bulge_scores[i],
            "density_score": local_density,
            "radial_vector": radial_vectors[i],
            "dist_to_tract": dist_to_tract,
            "proximity_score": proximity_score,
            "alignment_score": alignment_score
        })
    
    return analyzed_points

# ==========================================
# 7. DOMINANT INFILTRATION VECTOR
# ==========================================

def get_dominant_infiltration_vector(analyzed_points, tract_centerline):
    """
    Find the single best infiltration vector for a tract.
    """
    if not analyzed_points:
        return None, 0.0
    
    best_score = -1
    best_vector = None
    
    for point_data in analyzed_points:
        prox_score = point_data["proximity_score"]
        dir_score = point_data["alignment_score"]
        bulge_score = point_data["bulge_score"]
        density_score = point_data["density_score"]
        
        final_score = (
            (prox_score * W_PROXIMITY) +
            (dir_score * W_DIRECTION) +
            (bulge_score * W_BULGE) +
            (density_score * W_DENSITY)
        )
        
        if final_score > best_score:
            best_score = final_score
            
            nearest_tract_point, _ = find_nearest_tract_point(
                point_data["point"], tract_centerline
            )
            
            best_vector = {
                "source_point": nearest_tract_point if nearest_tract_point is not None else point_data["point"],
                "direction": point_data["radial_vector"],
                "score": final_score,
                "dist_to_tract": point_data["dist_to_tract"]
            }
    
    return best_vector, best_score

# ==========================================
# 8. BLOB VISUALIZATION (MODIFIED)
# ==========================================

def create_progressive_blob_heatmap(dti_rgb, hotspots_data, shape):
    """
    Create anatomically-grounded heatmap with progressive intensity.
    """
    h, w = shape[:2]
    
    blob_layer = np.zeros((h, w), dtype=np.float32)
    core_layer = np.zeros((h, w), dtype=np.float32)
    
    for hotspot in hotspots_data:
        score = hotspot["score"]
        source_point = hotspot["source"]
        direction = hotspot["direction"]
        
        if score <= 0 or np.linalg.norm(direction) == 0:
            continue
        
        norm_dir = direction / np.linalg.norm(direction)
        hotspot_x = int(source_point[0] + norm_dir[0] * BLOB_HOTSPOT_OFFSET)
        hotspot_y = int(source_point[1] + norm_dir[1] * BLOB_HOTSPOT_OFFSET)
        
        if not (0 <= hotspot_y < h and 0 <= hotspot_x < w):
            continue
        
        blob_layer[hotspot_y, hotspot_x] = float(score) * 0.6
        core_layer[hotspot_y, hotspot_x] = float(score)
    
    blob_layer = gaussian_filter(blob_layer, sigma=BLOB_HEATMAP_SIGMA)
    core_layer = gaussian_filter(core_layer, sigma=BLOB_CORE_SIGMA)
    combined = blob_layer + core_layer * 0.6
    
    max_val = np.max(combined)
    if max_val > 0:
        combined /= max_val
    
    heatmap_viz = (combined * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_JET)
    
    alpha = 0.55
    output_img = cv2.addWeighted(dti_rgb, 1 - alpha, heatmap_color, alpha, 0)
    
    _, thresh_map = cv2.threshold(heatmap_viz, int(BLOB_CONTOUR_THRESH * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_img, contours, -1, (255, 255, 255), 2)
    
    return output_img

def add_recurrence_label(img, total_score):
    """Add top-right recurrence percentage."""
    # total_score is now the MAX parent score (0-1)
    label = f"Recurrence Risk: {total_score*100:.1f}%" 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x = img.shape[1] - text_w - 15
    y = 30
    cv2.rectangle(img, (x-5, y-text_h-5), (x+text_w+5, y+5), (0, 0, 0), -1)
    cv2.putText(img, label, (x, y), font, font_scale, (0, 255, 0), thickness)
    return img

# --- NEW: Function to draw the vector debug map ---
def create_vector_debug_visualization(dti_rgb, all_divs, sub_features, tumor_mask):
    """
    Creates Image B (DIV / Vector Debug Map):
    Shows DTI, ALL SUB-TRACT boxes, and DIV arrows.
    """
    output_img = dti_rgb.copy()
    
    # 1. Draw tumor outline for context (Gray outline)
    if tumor_mask is not None:
        tumor_contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_img, tumor_contours, -1, (128, 128, 128), 2)
    
    # 2. Draw ALL SUB-TRACT Bounding Boxes
    for name, feat in sub_features.items():
        x1, y1, x2, y2 = feat["bbox"]
        risk_tier = feat["risk_tier"]
        score = feat["recurrence_score"]
        
        # Color by risk tier
        if risk_tier == "very_high": color = (255, 0, 0)      # Red
        elif risk_tier == "high": color = (255, 128, 0)     # Orange
        elif risk_tier == "moderate": color = (255, 255, 0) # Yellow
        elif risk_tier == "low": color = (128, 255, 0)      # Light green
        else: color = (128, 128, 128)                       # Gray
        
        thickness = 2 if score > 0.01 else 1
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)
        
        # Add labels
        label = f"{name}\n{feat['min_distance']:.1f}px\n{score*100:.1f}%"
        y_offset = y1 - 10
        for line in label.split('\n'):
            cv2.putText(output_img, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 15
            
    # 3. Draw Enhanced DIV Visualization
    for seg_name, vector_data in all_divs.items():
        score = sub_features[seg_name]["recurrence_score"]
        # Use permissive threshold for drawing
        if score > 0.005: 
            source = vector_data["source_point"]
            direction = vector_data["direction"]
            
            if np.linalg.norm(direction) > 0:
                arrow_end = source + (direction / np.linalg.norm(direction)) * 40
                
                if score > 0.3: arrow_color = (255, 0, 255) # Magenta
                elif score > 0.15: arrow_color = (255, 165, 0) # Orange
                else: arrow_color = (0, 255, 255) # Cyan
                    
                thickness = max(2, int(1 + score * 30))
                
                cv2.arrowedLine(output_img, tuple(source.astype(int)), tuple(arrow_end.astype(int)), arrow_color, thickness,
                                tipLength=0.25, line_type=cv2.LINE_AA)
                cv2.circle(output_img, tuple(source.astype(int)), 3, arrow_color, -1)
    
    # 4. Add legend
    legend_x = 10; legend_y = output_img.shape[0] - 150
    cv2.rectangle(output_img, (legend_x-5, legend_y-25), (legend_x+160, legend_y+100), (0, 0, 0), -1)
    cv2.putText(output_img, "Risk Tiers:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(output_img, (legend_x, legend_y+5), (legend_x+20, legend_y+15), (255, 0, 0), -1)
    cv2.putText(output_img, "Very High (<2px)", (legend_x+25, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+20), (legend_x+20, legend_y+30), (255, 128, 0), -1)
    cv2.putText(debug_img, "High (<5px)", (legend_x+25, legend_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+35), (legend_x+20, legend_y+45), (255, 255, 0), -1)
    cv2.putText(debug_img, "Moderate (<10px)", (legend_x+25, legend_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+50), (legend_x+20, legend_y+60), (128, 255, 0), -1)
    cv2.putText(debug_img, "Low (<15px)", (legend_x+25, legend_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+65), (legend_x+20, legend_y+75), (128, 128, 128), -1)
    cv2.putText(debug_img, "None (>15px)", (legend_x+25, legend_y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return output_img

# ==========================================
# 9. PROGRESSIVE PROPAGATION (FIXED)
# ==========================================

def propagate_recurrence_progressive(parent_features, sub_features, graph):
    """
    Propagate recurrence scores using the 8-node parent graph.
    """
    # 1. Aggregate scores from sub-tracts to parent tracts
    for name, feat in parent_features.items():
        sub_tract_names = [sub_name for sub_name in sub_features if sub_name.startswith(name)]
        
        total_initial_score = sum(sub_features[sub_name].get("initial_score", 0) for sub_name in sub_tract_names)
        
        # --- FIX #2: Use max() for proximity, not sum() ---
        total_prox_score = max(
            (sub_features[sub_name].get("proximity_score", 0) for sub_name in sub_tract_names),
            default=0.0
        )
        
        feat["initial_score"] = total_initial_score
        feat["recurrence_score"] = total_initial_score
        feat["proximity_score"] = total_prox_score

    # 2. Run propagation on the 8 parent tracts
    at_risk_tracts = [name for name, feat in parent_features.items() 
                     if feat.get("proximity_score", 0) > 0]
    
    if not at_risk_tracts:
        for feat in parent_features.values():
            feat["recurrence_score"] = 0
        return parent_features, sub_features

    tumor_in_middle = any(name in MIDDLE_TRACTS for name in at_risk_tracts)
    tumor_on_left = any(name in LEFT_TRACTS for name in at_risk_tracts)
    tumor_on_right = any(name in RIGHT_TRACTS for name in at_risk_tracts)
    
    if not tumor_in_middle:
        for mid_tract in MIDDLE_TRACTS:
            if parent_features[mid_tract].get("proximity_score", 0) == 0:
                parent_features[mid_tract]["recurrence_score"] = 0
        
        if tumor_on_left and not tumor_on_right:
            for right_tract in RIGHT_TRACTS:
                parent_features[right_tract]["recurrence_score"] = 0
        elif tumor_on_right and not tumor_on_left:
            for left_tract in LEFT_TRACTS:
                parent_features[left_tract]["recurrence_score"] = 0
    
    active_tracts = [name for name, feat in parent_features.items() if feat["recurrence_score"] > 0]
    if not active_tracts:
        return parent_features, sub_features

    max_name = max(active_tracts, key=lambda x: parent_features[x]["recurrence_score"])
    
    # --- FIX #1: Use parent_features.keys() ---
    if not tumor_in_middle:
        if tumor_on_left: allowed_tracts = set(LEFT_TRACTS)
        elif tumor_on_right: allowed_tracts = set(RIGHT_TRACTS)
        else: allowed_tracts = set(parent_features.keys()) # <-- FIX 1a
    else:
        allowed_tracts = set(parent_features.keys()) # <-- FIX 1b
    
    visited = {max_name}
    priority_queue = [(0, max_name, parent_features[max_name]["recurrence_score"])]
    
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
            
            parent_features[neighbor]["recurrence_score"] += increment
            priority_queue.append((new_cum_dist, neighbor, parent_features[neighbor]["recurrence_score"]))
            visited.add(neighbor)
            
    # 3. Broadcast final scores from parent tracts back to sub-tracts
    for sub_name, sub_feat in sub_features.items():
        parent_name = sub_name.split('_p')[0]
        parent_final_score = parent_features[parent_name]["recurrence_score"]
        parent_initial_total = parent_features[parent_name]["initial_score"]
        
        if parent_initial_total > 0:
            sub_initial_score = sub_feat.get("initial_score", 0)
            sub_feat["recurrence_score"] = parent_final_score * (sub_initial_score / parent_initial_total)
        else:
            num_sub_tracts = len([s for s in sub_features if s.startswith(parent_name)])
            if num_sub_tracts > 0:
                sub_feat["recurrence_score"] = parent_final_score / num_sub_tracts
            else:
                sub_feat["recurrence_score"] = 0
            
    return parent_features, sub_features

# ==========================================
# 10. MAIN PROCESSING LOOP
# ==========================================

# Build the 8-node graph ONCE
tract_graph, tract_centroids = build_tract_graph()

data_records = []
workspace_root = Path(__file__).parent.parent
tract_folder = workspace_root / "dataset" / "main" / "dti-before"
flair_folder = workspace_root / "dataset" / "main" / "flair-before"
results_folder = workspace_root / "results"
heatmaps_folder = results_folder / "heatmaps"
debug_folder = results_folder / "debug_vectors"

results_folder.mkdir(parents=True, exist_ok=True)
heatmaps_folder.mkdir(parents=True, exist_ok=True)
debug_folder.mkdir(parents=True, exist_ok=True)

tract_files = sorted(list(tract_folder.glob("*.png")))
flair_files = sorted(list(flair_folder.glob("*.png")))

print(f"🔍 Found {len(tract_files)} DTI files and {len(flair_files)} FLAIR files")
if len(tract_files) == 0 or len(flair_files) == 0:
    print(f"⚠️ ERROR: No image files found in {tract_folder} or {flair_folder}")
    sys.exit(1)

idx = 1

for tract_path, flair_path in zip(tract_files, flair_files):
    print(f"\n🔹 Processing image {idx}: {flair_path.name}")
    
    # --- FIX #1: Check for None *before* cvtColor ---
    dti_bgr = cv2.imread(str(tract_path))
    flair_img = cv2.imread(str(flair_path), 0)
    
    if dti_bgr is None or flair_img is None:
        print(f"⚠️ Failed to load images")
        idx += 1
        continue
        
    dti_rgb = cv2.cvtColor(dti_bgr, cv2.COLOR_BGR2RGB) # Moved here
    
    # --- Create this image's tract segments ---
    tract_boxes = {}
    parent_tract_map = {} # Maps sub-tract name to parent name
    for name, box in original_tracts.items():
        sub_tracts = split_tract_box(name, box, num_splits=4, axis='vertical') 
        tract_boxes.update(sub_tracts)
        for sub_name in sub_tracts:
            parent_tract_map[sub_name] = name
            
    # Tumor detection
    _, tumor_mask_binary = cv2.threshold(flair_img, TUMOR_THRESH, 255, cv2.THRESH_BINARY)
    tumor_mask_morphed = cv2.morphologyEx(tumor_mask_binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tumor_mask = filter_small_regions(tumor_mask_morphed, min_size=MIN_AREA)
    
    tumor_bbox = get_bbox(tumor_mask)
    if tumor_bbox is None:
        print("   No tumor found")
        idx += 1
        continue
    
    print(f"   Tumor: {np.count_nonzero(tumor_mask)} pixels")
    
    distance_map = compute_distance_to_tumor(tumor_mask)
    tumor_centroid, tumor_boundary = get_tumor_centroid_and_boundary(tumor_mask)
    
    if tumor_centroid is None:
        print("   No tumor centroid found")
        idx += 1
        continue
    
    parent_features = {} # For the 8 parent tracts
    sub_features = {}    # For all sub-tracts
    dominant_vectors_dict = {}
    
    print("   Computing soft-contact proximity scores...")
    
    for name, box in original_tracts.items():
        parent_features[name] = {
            "bbox": box,
            "iou": iou(tumor_bbox, box),
            "initial_score": 0,
            "recurrence_score": 0,
            "proximity_score": 0
        }
    
    for name, box in tract_boxes.items():
        tract_mask = extract_tract_mask(dti_rgb, box)
        tract_centerline = compute_tract_centerline(tract_mask)
        
        min_dist, proximity_score, risk_tier = compute_tract_proximity_score(
            tract_mask, distance_map
        )
        
        print(f"     {name}: {risk_tier} risk (min_dist={min_dist:.1f}px, prox={proximity_score:.3f})")
        
        sub_features[name] = {
            "bbox": box,
            "min_distance": min_dist,
            "proximity_score": proximity_score,
            "risk_tier": risk_tier,
            "initial_score": 0,
            "recurrence_score": 0
        }
        
        if proximity_score == 0 or min_dist > DIST_NO_RISK:
            continue
        
        analyzed_points = analyze_boundary_features_for_tract(
            tumor_mask, tumor_centroid, tumor_boundary,
            tract_centerline, distance_map
        )
        
        if len(analyzed_points) == 0:
            print(f"       No directionally-aligned boundary points found")
            continue
        
        dominant_vector, initial_score = get_dominant_infiltration_vector(
            analyzed_points, tract_centerline
        )
        
        if dominant_vector:
            weighted_score = initial_score * proximity_score
            sub_features[name]["initial_score"] = weighted_score
            dominant_vectors_dict[name] = dominant_vector
            print(f"       Initial score: {weighted_score:.3f}")

    print("   Propagating along fiber pathways...")
    parent_features, sub_features = propagate_recurrence_progressive(
        parent_features, sub_features, tract_graph
    )
    
    # Normalize scores
    total_score_sum = sum(feat["recurrence_score"] for feat in parent_features.values())
    if total_score_sum > 0:
        for feat in parent_features.values():
            feat["recurrence_score"] /= total_score_sum
        
    display_score = 0.0
    if parent_features:
        # Get the max score *after* normalization
        display_score = max(feat["recurrence_score"] for feat in parent_features.values())
            
    final_hotspots = []
    for name, feat in sub_features.items():
        parent_name = parent_tract_map[name]
        parent_final_score = parent_features[parent_name]["recurrence_score"]
        parent_initial_total = parent_features[parent_name]["initial_score"]
        
        final_score = 0
        if parent_initial_total > 0:
            sub_initial_score = feat.get("initial_score", 0)
            final_score = parent_final_score * (sub_initial_score / parent_initial_total)
        else:
            num_sub_tracts = len([s for s in sub_features if s.startswith(parent_name)])
            if num_sub_tracts > 0:
                final_score = parent_final_score / num_sub_tracts

        feat["recurrence_score"] = final_score
        
        # Use permissive threshold for visualization
        if final_score > 0.005 and name in dominant_vectors_dict:
            vector_data = dominant_vectors_dict[name]
            final_hotspots.append({
                "source": vector_data["source_point"],
                "direction": vector_data["direction"],
                "score": final_score
            })
    
    print(f"   Generated {len(final_hotspots)} hotspots")
    
    print("   Rendering heatmap...")
    hotspot_img = create_progressive_blob_heatmap(dti_rgb, final_hotspots, dti_rgb.shape)
    
    tumor_contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hotspot_img, tumor_contours, -1, (255, 255, 0), 2)
    
    # Use total_score_sum (pre-normalization) for the label if you want to see values > 100%
    # Or use display_score (1.0) to show that risk is present
    hotspot_img = add_recurrence_label(hotspot_img, display_score) # Changed to display_score
    
    # --- MODIFIED: Save Image A ---
    output_filename = flair_path.stem + "_refined_confidence_prediction.png"
    output_path = heatmaps_folder / output_filename
    cv2.imwrite(str(output_path), cv2.cvtColor(hotspot_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved {output_filename}")
    
    # --- MODIFIED: Generate debug visualization (Image B) ---
    debug_img = dti_rgb.copy()
    
    # Draw SUB-TRACT boxes
    for name, feat in sub_features.items():
        x1, y1, x2, y2 = feat["bbox"]
        risk_tier = feat["risk_tier"]
        score = feat["recurrence_score"]

        if risk_tier == "very_high": color = (255, 0, 0)
        elif risk_tier == "high": color = (255, 128, 0)
        elif risk_tier == "moderate": color = (255, 255, 0)
        elif risk_tier == "low": color = (128, 255, 0)
        else: color = (128, 128, 128)
        
        thickness = 2 if score > 0.01 else 1
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)
        
        # Add labels for sub-tracts
        label = f"{name.split('_')[-1]}\n{feat['min_distance']:.1f}px\n{score*100:.1f}%"
        y_offset = y1 + 10
        for line in label.split('\n'):
            cv2.putText(debug_img, line, (x1 + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
            
    # Draw infiltration vectors
    for name, vector_data in dominant_vectors_dict.items():
        score = sub_features[name]["recurrence_score"]
        if score <= 0.005:
            print(f"[DEBUG] Dropped vector {name} from viz with score {score:.4f}")
            continue

        source = vector_data["source_point"]
        direction = vector_data["direction"]
        if np.linalg.norm(direction) > 0:
            arrow_end = source + (direction / np.linalg.norm(direction)) * 40
            
            # Scale color and thickness by normalized score
            if score > 0.3: arrow_color = (255, 0, 255)
            elif score > 0.15: arrow_color = (255, 165, 0)
            else: arrow_color = (0, 255, 255)
            thickness = max(2, int(1 + score * 30)) # Scale thickness more aggressively
            
            cv2.arrowedLine(debug_img, tuple(source.astype(int)), tuple(arrow_end.astype(int)), arrow_color, thickness, tipLength=0.3)
            cv2.circle(debug_img, tuple(source.astype(int)), 4, (0, 255, 255), -1)
    
    cv2.drawContours(debug_img, tumor_contours, -1, (255, 255, 0), 2)
    
    # Add legend
    legend_x = 10; legend_y = debug_img.shape[0] - 150
    cv2.rectangle(debug_img, (legend_x-5, legend_y-25), (legend_x+160, legend_y+100), (0, 0, 0), -1)
    cv2.putText(debug_img, "Risk Tiers:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+5), (legend_x+20, legend_y+15), (255, 0, 0), -1)
    cv2.putText(debug_img, "Very High (<2px)", (legend_x+25, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+20), (legend_x+20, legend_y+30), (255, 128, 0), -1)
    cv2.putText(debug_img, "High (<5px)", (legend_x+25, legend_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+35), (legend_x+20, legend_y+45), (255, 255, 0), -1)
    cv2.putText(debug_img, "Moderate (<10px)", (legend_x+25, legend_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+50), (legend_x+20, legend_y+60), (128, 255, 0), -1)
    cv2.putText(debug_img, "Low (<15px)", (legend_x+25, legend_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(debug_img, (legend_x, legend_y+65), (legend_x+20, legend_y+75), (128, 128, 128), -1)
    cv2.putText(debug_img, "None (>15px)", (legend_x+25, legend_y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # --- MODIFIED: Save Image B ---
    debug_filename = flair_path.stem + "_vectors.png" # New filename
    debug_path = debug_folder / debug_filename
    cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved {debug_filename}")
    
    # Store data records
    for name, feat in sub_features.items(): # Save sub-tract data
        parent_name = parent_tract_map[name]
        data_records.append({
            "image_id": idx,
            "tract": name,
            "parent_tract": parent_name,
            "risk_tier": feat["risk_tier"],
            "min_distance_px": round(feat["min_distance"], 2),
            "proximity_score": round(feat["proximity_score"], 3),
            "initial_score": round(feat["initial_score"] * 100, 2),
            "final_recurrence_score": round(feat["recurrence_score"] * 100, 2)
        })
    
    idx += 1

# Save CSV
if data_records:
    df = pd.DataFrame(data_records)
    csv_path = results_folder / "soft_contact_recurrence_analysis.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n✅ CSV saved: {csv_path}")
    print(f"   Total records: {len(data_records)}")
    print(f"   Images processed: {idx - 1}")
    
    print("\n📊 Summary Statistics:")
    if (idx - 1) > 0:
        print(f"   Average sub-tracts at risk per image: {len(df[df['final_recurrence_score'] > 0]) / (idx - 1):.1f}")
    risk_counts = df['risk_tier'].value_counts()
    print(f"   Risk tier distribution (all sub-tracts):")
    for tier, count in risk_counts.items():
        print(f"     - {tier}: {count} ({count/len(data_records)*100:.1f}%)")
else:
    print("\n⚠️ No data records to save")

print(f"\n🎉 Processing complete!")
print(f"   Heatmaps: {heatmaps_folder}")
print(f"   Debug images: {debug_folder}")
print(f"   Analysis CSV: {results_folder / 'soft_contact_recurrence_analysis.csv'}")
print(f"\n💡 Key improvements:")
print(f"   ✓ Sub-tract segmentation for localized risk")
print(f"   ✓ Soft-contact proximity scoring (0-15px risk stratification)")
print(f"   ✓ Directional filtering (radial vectors must point toward tracts)")
print(f"   ✓ Anatomical snapping (hotspots placed on tract centerlines)")
print(f"   ✓ Progressive propagation (risk spreads along fiber pathways)")
print(f"   ✓ Two-stage blob rendering (smooth blobs + highlighted cores)")