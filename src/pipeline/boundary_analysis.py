"""
Boundary analysis and proximity scoring.
"""
import numpy as np
from . import config


def compute_tract_proximity_score(tract_mask, distance_map):
    """
    Compute soft-contact score based on proximity to tumor.
    """
    if np.count_nonzero(tract_mask) == 0:
        return float('inf'), 0.0, "none"
    
    tract_distances = distance_map[tract_mask > 0]
    min_distance = np.min(tract_distances)
    
    weights = np.zeros_like(tract_distances)
    weights[tract_distances <= config.Config.DIST_VERY_HIGH_RISK] = config.Config.RISK_WEIGHT_VERY_HIGH
    weights[(tract_distances > config.Config.DIST_VERY_HIGH_RISK) & (tract_distances <= config.Config.DIST_HIGH_RISK)] = config.Config.RISK_WEIGHT_HIGH
    weights[(tract_distances > config.Config.DIST_HIGH_RISK) & (tract_distances <= config.Config.DIST_MODERATE_RISK)] = config.Config.RISK_WEIGHT_MODERATE
    weights[(tract_distances > config.Config.DIST_MODERATE_RISK) & (tract_distances <= config.Config.DIST_NO_RISK)] = config.Config.RISK_WEIGHT_LOW
    
    proximity_score = np.mean(weights) if len(weights) > 0 else 0.0
    
    if min_distance <= config.Config.DIST_VERY_HIGH_RISK: risk_tier = "very_high"
    elif min_distance <= config.Config.DIST_HIGH_RISK: risk_tier = "high"
    elif min_distance <= config.Config.DIST_MODERATE_RISK: risk_tier = "moderate"
    elif min_distance <= config.Config.DIST_NO_RISK: risk_tier = "low"
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


def check_radial_alignment(radial_vector, boundary_point, tract_centerline):
    """
    Check if radial vector points TOWARD the tract.
    """
    if len(tract_centerline) == 0:
        return 0.0, False
    
    nearest_tract_point, dist = find_nearest_tract_point(boundary_point, tract_centerline)
    
    if nearest_tract_point is None or dist > config.Config.DIST_NO_RISK:
        return 0.0, False
    
    to_tract_vector = nearest_tract_point - boundary_point
    
    if np.linalg.norm(to_tract_vector) == 0 or np.linalg.norm(radial_vector) == 0:
        return 0.0, False
    
    norm_radial = radial_vector / np.linalg.norm(radial_vector)
    norm_to_tract = to_tract_vector / np.linalg.norm(to_tract_vector)
    
    dot_product = np.clip(np.dot(norm_radial, norm_to_tract), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_product))
    
    alignment_score = max(0, 1.0 - (angle_deg / 90.0))
    is_aligned = angle_deg <= config.Config.ANGULAR_THRESHOLD
    
    return alignment_score, is_aligned


def analyze_boundary_features_for_tract(tumor_mask, tumor_centroid, boundary_points, 
                                       tract_centerline, distance_map):
    """
    Analyze tumor boundary points specifically in relation to ONE tract.
    Vectorized implementation using NumPy broadcasting.
    """
    if tumor_centroid is None or boundary_points is None or len(boundary_points) == 0:
        return []
    
    if len(tract_centerline) == 0:
        return []
    
    # Convert boundary points to numpy array if not already
    boundary_points = np.asarray(boundary_points)
    if boundary_points.ndim == 1:
        boundary_points = boundary_points.reshape(1, -1)
    
    num_points = len(boundary_points)
    h, w = tumor_mask.shape
    
    # 1. Compute radial vectors and distances (already vectorized)
    # Ensure float type for proper division operations
    radial_vectors = (boundary_points - tumor_centroid).astype(np.float64)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    
    avg_distance = np.mean(radial_distances[radial_distances > 0])
    if np.isnan(avg_distance) or avg_distance == 0:
        avg_distance = 1
    
    bulge_scores = radial_distances / avg_distance
    
    # 2. Filter out points with zero radial distance
    valid_radial_mask = radial_distances > 0
    
    # 3. Filter out points outside image bounds
    x_coords = boundary_points[:, 0].astype(int)
    y_coords = boundary_points[:, 1].astype(int)
    valid_bounds_mask = (x_coords >= 0) & (x_coords < distance_map.shape[1]) & \
                        (y_coords >= 0) & (y_coords < distance_map.shape[0])
    
    # 4. Vectorized nearest tract point computation
    # Convert centerline from (y, x) to (x, y) format for consistency
    if len(tract_centerline) > 0:
        centerline_xy = np.column_stack([tract_centerline[:, 1], tract_centerline[:, 0]])
    else:
        centerline_xy = np.empty((0, 2))
    
    # Compute distances from all boundary points to all centerline points
    # Shape: (num_boundary_points, num_centerline_points)
    if len(centerline_xy) > 0:
        # Broadcasting: (num_points, 1, 2) - (1, num_centerline, 2) = (num_points, num_centerline, 2)
        diff = boundary_points[:, np.newaxis, :] - centerline_xy[np.newaxis, :, :]
        distances_to_centerline = np.linalg.norm(diff, axis=2)  # (num_points, num_centerline)
        
        # Find nearest centerline point for each boundary point
        min_indices = np.argmin(distances_to_centerline, axis=1)  # (num_points,)
        dists_to_tract = distances_to_centerline[np.arange(num_points), min_indices]  # (num_points,)
        nearest_tract_points = centerline_xy[min_indices]  # (num_points, 2)
    else:
        dists_to_tract = np.full(num_points, np.inf)
        nearest_tract_points = np.full((num_points, 2), np.nan)
    
    # 5. Filter by distance threshold
    valid_distance_mask = dists_to_tract <= config.Config.DIST_NO_RISK
    
    # 6. Vectorized radial alignment computation
    # Compute vectors from boundary points to nearest tract points
    # Ensure float type for proper division operations
    to_tract_vectors = (nearest_tract_points - boundary_points).astype(np.float64)  # (num_points, 2)
    
    # Normalize radial vectors and to_tract vectors
    radial_norms = np.linalg.norm(radial_vectors, axis=1, keepdims=True)  # (num_points, 1)
    to_tract_norms = np.linalg.norm(to_tract_vectors, axis=1, keepdims=True)  # (num_points, 1)
    
    # Avoid division by zero
    valid_radial_norm_mask = radial_norms.squeeze() > 0
    valid_to_tract_norm_mask = to_tract_norms.squeeze() > 0
    
    # Normalize vectors (ensure float output)
    norm_radial = np.divide(radial_vectors.astype(np.float64), radial_norms.astype(np.float64), 
                           out=np.zeros_like(radial_vectors, dtype=np.float64), 
                           where=radial_norms > 0)
    norm_to_tract = np.divide(to_tract_vectors.astype(np.float64), to_tract_norms.astype(np.float64),
                              out=np.zeros_like(to_tract_vectors, dtype=np.float64),
                              where=to_tract_norms > 0)
    
    # Compute dot products (cosine of angle)
    dot_products = np.sum(norm_radial * norm_to_tract, axis=1)  # (num_points,)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Compute angles in degrees
    angles_deg = np.degrees(np.arccos(dot_products))
    
    # Compute alignment scores and mask
    alignment_scores = np.maximum(0, 1.0 - (angles_deg / 90.0))
    is_aligned_mask = angles_deg <= config.Config.ANGULAR_THRESHOLD
    
    # Combine all validity masks
    valid_mask = (valid_radial_mask & valid_bounds_mask & valid_distance_mask & 
                  valid_radial_norm_mask & valid_to_tract_norm_mask & is_aligned_mask)
    
    if not np.any(valid_mask):
        return []
    
    # 7. Compute patch centers for density computation
    # patch_center = point - (radial_vector / radial_distance) * 5
    normalized_radial = np.divide(radial_vectors.astype(np.float64), radial_norms.astype(np.float64),
                                  out=np.zeros_like(radial_vectors, dtype=np.float64),
                                  where=radial_norms > 0)
    patch_centers = boundary_points - normalized_radial * 5  # (num_points, 2)
    patch_centers_int = patch_centers.astype(int)
    px_coords = patch_centers_int[:, 0]
    py_coords = patch_centers_int[:, 1]
    
    # Filter patch centers that are within bounds
    valid_patch_mask = (px_coords >= 0) & (px_coords < w) & (py_coords >= 0) & (py_coords < h)
    valid_mask = valid_mask & valid_patch_mask
    
    if not np.any(valid_mask):
        return []
    
    # 8. Vectorized density computation using uniform filter approach
    # Use scipy's uniform filter for efficient patch-based density computation
    from scipy.ndimage import uniform_filter
    
    # Create a normalized tumor mask for density computation
    tumor_mask_normalized = tumor_mask.astype(np.float32) / 255.0
    
    # Apply uniform filter (equivalent to mean over 11x11 patches, accounting for edges)
    # This computes local density efficiently for all points
    patch_size = 11  # 5 pixels on each side + center = 11x11
    density_map = uniform_filter(tumor_mask_normalized, size=patch_size, mode='constant', cval=0.0)
    
    # Extract densities at patch center locations (clamped to valid bounds)
    px_clamped = np.clip(px_coords, 0, w - 1)
    py_clamped = np.clip(py_coords, 0, h - 1)
    local_densities = density_map[py_clamped, px_clamped]
    
    # 9. Compute proximity scores (vectorized)
    proximity_scores = 1.0 / (dists_to_tract + 1.0)
    
    # 10. Build result list from valid points using vectorized indexing
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return []
    
    # Extract all data for valid points using vectorized indexing
    valid_points = boundary_points[valid_indices]
    valid_bulge_scores = bulge_scores[valid_indices]
    valid_densities = local_densities[valid_indices]
    valid_radial_vectors = radial_vectors[valid_indices]
    valid_dists_to_tract = dists_to_tract[valid_indices]
    valid_proximity_scores = proximity_scores[valid_indices]
    valid_alignment_scores = alignment_scores[valid_indices]
    
    # Build result list using list comprehension (most efficient for dict creation)
    analyzed_points = [
        {
            "point": valid_points[i],
            "bulge_score": valid_bulge_scores[i],
            "density_score": valid_densities[i],
            "radial_vector": valid_radial_vectors[i],
            "dist_to_tract": valid_dists_to_tract[i],
            "proximity_score": valid_proximity_scores[i],
            "alignment_score": valid_alignment_scores[i]
        }
        for i in range(len(valid_indices))
    ]
    
    return analyzed_points

