"""
Visualization functions for heatmaps and debug images.
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from . import config


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
        hotspot_x = int(source_point[0] + norm_dir[0] * config.Config.BLOB_HOTSPOT_OFFSET)
        hotspot_y = int(source_point[1] + norm_dir[1] * config.Config.BLOB_HOTSPOT_OFFSET)
        
        if not (0 <= hotspot_y < h and 0 <= hotspot_x < w):
            continue
        
        blob_layer[hotspot_y, hotspot_x] = float(score) * 0.6
        core_layer[hotspot_y, hotspot_x] = float(score)
    
    blob_layer = gaussian_filter(blob_layer, sigma=config.Config.BLOB_HEATMAP_SIGMA)
    core_layer = gaussian_filter(core_layer, sigma=config.Config.BLOB_CORE_SIGMA)
    combined = blob_layer + core_layer * 0.6
    
    max_val = np.max(combined)
    if max_val > 0:
        combined /= max_val
    
    heatmap_viz = (combined * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_JET)
    
    alpha = 0.55
    output_img = cv2.addWeighted(dti_rgb, 1 - alpha, heatmap_color, alpha, 0)
    
    _, thresh_map = cv2.threshold(heatmap_viz, int(config.Config.BLOB_CONTOUR_THRESH * 255), 255, cv2.THRESH_BINARY)
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
    cv2.rectangle(output_img, (legend_x, legend_y+20), (legend_x+20, legend_y+30), (255, 128, 0), -1)
    cv2.putText(output_img, "High (<5px)", (legend_x+25, legend_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(output_img, (legend_x, legend_y+35), (legend_x+20, legend_y+45), (255, 255, 0), -1)
    cv2.putText(output_img, "Moderate (<10px)", (legend_x+25, legend_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(output_img, (legend_x, legend_y+50), (legend_x+20, legend_y+60), (128, 255, 0), -1)
    cv2.putText(output_img, "Low (<15px)", (legend_x+25, legend_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(output_img, (legend_x, legend_y+65), (legend_x+20, legend_y+75), (128, 128, 128), -1)
    cv2.putText(output_img, "None (>15px)", (legend_x+25, legend_y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return output_img


def create_debug_visualization(dti_rgb, sub_features, dominant_vectors_dict, tumor_mask):
    """
    Create debug visualization with sub-tract boxes and infiltration vectors.
    """
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
    
    tumor_contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    
    return debug_img


def visualize_tract_masks(dti_rgb, precomputed_data):
    """
    Visualize all tract masks overlaid on DTI image.
    Stage 1: Tract masks
    """
    output_img = dti_rgb.copy()
    
    # Combine all tract masks
    combined_mask = np.zeros(dti_rgb.shape[:2], dtype=np.uint8)
    
    # Draw each tract mask with different colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    color_idx = 0
    for name, tract_mask in precomputed_data.tract_masks.items():
        if np.count_nonzero(tract_mask) > 0:
            color = colors[color_idx % len(colors)]
            # Create colored mask
            mask_colored = np.zeros_like(output_img)
            mask_colored[tract_mask > 0] = color
            # Blend with original image
            output_img = cv2.addWeighted(output_img, 0.7, mask_colored, 0.3, 0)
            color_idx += 1
    
    # Draw bounding boxes
    for name, box in precomputed_data.sub_tract_boxes.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(output_img, name.split('_')[-1], (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return output_img


def visualize_tumor_boundary(dti_rgb, precomputed_data):
    """
    Visualize tumor boundary and centroid.
    Stage 2: Tumor boundary
    """
    output_img = dti_rgb.copy()
    
    # Draw tumor boundary
    if precomputed_data.tumor_boundary is not None:
        boundary_points = precomputed_data.tumor_boundary.astype(int)
        # Draw boundary as connected points
        for i in range(len(boundary_points) - 1):
            pt1 = tuple(boundary_points[i])
            pt2 = tuple(boundary_points[i + 1])
            cv2.line(output_img, pt1, pt2, (255, 0, 0), 2)
        # Close the loop
        if len(boundary_points) > 2:
            cv2.line(output_img, tuple(boundary_points[-1]), tuple(boundary_points[0]), (255, 0, 0), 2)
    
    # Draw tumor centroid
    if precomputed_data.tumor_centroid is not None:
        centroid = tuple(precomputed_data.tumor_centroid.astype(int))
        cv2.circle(output_img, centroid, 5, (0, 255, 0), -1)
        cv2.circle(output_img, centroid, 10, (0, 255, 0), 2)
    
    # Draw tumor mask as semi-transparent overlay
    tumor_overlay = output_img.copy()
    tumor_overlay[precomputed_data.tumor_mask > 0] = [255, 0, 0]
    output_img = cv2.addWeighted(output_img, 0.7, tumor_overlay, 0.3, 0)
    
    return output_img


def visualize_radial_and_boundary_vectors(dti_rgb, precomputed_data, analyzed_points_dict):
    """
    Visualize radial vectors from centroid to boundary and vectors from boundary to tract.
    Stage 4: Radial vectors and boundary-to-tract vectors
    """
    from . import boundary_analysis
    
    output_img = dti_rgb.copy()
    
    # Count total radial vectors
    total_radial_vectors = sum(len(points) for points in analyzed_points_dict.values())
    
    # Add text at the top showing the count
    count_text = f"Total Radial Vectors: {total_radial_vectors}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(count_text, font, font_scale, thickness)
    
    # Draw background rectangle for text
    text_x = 10
    text_y = 30
    cv2.rectangle(output_img, (text_x - 5, text_y - text_h - 5), 
                  (text_x + text_w + 5, text_y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(output_img, count_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Draw tumor boundary and centroid
    if precomputed_data.tumor_boundary is not None:
        boundary_points = precomputed_data.tumor_boundary.astype(int)
        for i in range(len(boundary_points) - 1):
            cv2.line(output_img, tuple(boundary_points[i]), tuple(boundary_points[i + 1]), (128, 128, 128), 1)
    
    if precomputed_data.tumor_centroid is not None:
        centroid = tuple(precomputed_data.tumor_centroid.astype(int))
        cv2.circle(output_img, centroid, 5, (0, 255, 0), -1)
        cv2.circle(output_img, centroid, 10, (0, 255, 0), 2)
    
    # Draw tract centerlines
    for name, centerline in precomputed_data.tract_centerlines.items():
        if len(centerline) > 0:
            # Convert from (y, x) to (x, y) for drawing
            centerline_xy = np.column_stack([centerline[:, 1], centerline[:, 0]]).astype(int)
            # Draw centerline as connected line
            for i in range(len(centerline_xy) - 1):
                cv2.line(output_img, tuple(centerline_xy[i]), tuple(centerline_xy[i + 1]), (255, 255, 0), 2)
            # Draw points
            for pt in centerline_xy:
                cv2.circle(output_img, tuple(pt), 2, (255, 255, 0), -1)
    
    # Draw vectors for each tract
    vector_colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 165, 0),    # Orange
        (128, 0, 128),    # Purple
        (255, 192, 203),  # Pink
        (0, 255, 127),    # Spring Green
        (255, 20, 147),   # Deep Pink
        (0, 191, 255),    # Deep Sky Blue
    ]
    
    # Track color index per tract
    tract_color_map = {}
    color_idx = 0
    
    # First pass: assign colors to tracts
    for tract_name in analyzed_points_dict.keys():
        if len(analyzed_points_dict[tract_name]) > 0:
            tract_color_map[tract_name] = vector_colors[color_idx % len(vector_colors)]
            color_idx += 1
    
    # Second pass: draw vectors with assigned colors
    for tract_name, analyzed_points in analyzed_points_dict.items():
        if len(analyzed_points) == 0:
            continue
        
        tract_color = tract_color_map.get(tract_name, (255, 255, 255))
        
        # Use different shades for each boundary point within the same tract
        num_points = len(analyzed_points)
        for point_idx, point_data in enumerate(analyzed_points):
            boundary_point = point_data["point"].astype(int)
            radial_vector = point_data["radial_vector"]
            
            # Create a slightly different shade for each point in the tract
            # This helps distinguish individual vectors
            shade_factor = 0.7 + (point_idx % 3) * 0.1  # Vary between 0.7 and 0.9
            radial_color = tuple(int(c * shade_factor) for c in tract_color)
            
            # Draw radial vector (from centroid to boundary point)
            if precomputed_data.tumor_centroid is not None:
                centroid = precomputed_data.tumor_centroid.astype(int)
                cv2.arrowedLine(output_img, tuple(centroid), tuple(boundary_point), 
                              radial_color, 2, tipLength=0.15)
            
            # Draw vector from boundary to nearest tract point
            # Find nearest tract point using the centerline for this tract
            tract_centerline = precomputed_data.tract_centerlines.get(tract_name)
            if tract_centerline is not None and len(tract_centerline) > 0:
                nearest_tract_point, dist_to_tract = boundary_analysis.find_nearest_tract_point(
                    point_data["point"], tract_centerline
                )
                
                if nearest_tract_point is not None and dist_to_tract < config.Config.DIST_NO_RISK:
                    # Draw vector from boundary point to nearest tract point
                    # Use a brighter version of the tract color for boundary-to-tract vectors
                    tract_vector_color = tuple(min(255, int(c * 1.2)) for c in tract_color)
                    tract_point_int = nearest_tract_point.astype(int)
                    cv2.arrowedLine(output_img, tuple(boundary_point), tuple(tract_point_int),
                                  tract_vector_color, 2, tipLength=0.15)
                    cv2.circle(output_img, tuple(tract_point_int), 3, tract_vector_color, -1)
            
            # Mark boundary point with the radial vector color
            cv2.circle(output_img, tuple(boundary_point), 3, radial_color, -1)
    
    return output_img


def visualize_tract_centerlines(dti_rgb, precomputed_data):
    """
    Visualize tract centerlines.
    Stage 3: Tract centerlines (alternative view of tract masks)
    """
    output_img = dti_rgb.copy()
    
    # Draw tumor boundary for context
    if precomputed_data.tumor_boundary is not None:
        boundary_points = precomputed_data.tumor_boundary.astype(int)
        for i in range(len(boundary_points) - 1):
            cv2.line(output_img, tuple(boundary_points[i]), tuple(boundary_points[i + 1]), (128, 128, 128), 1)
    
    # Draw each tract centerline with different colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    color_idx = 0
    for name, centerline in precomputed_data.tract_centerlines.items():
        if len(centerline) > 0:
            color = colors[color_idx % len(colors)]
            # Convert from (y, x) to (x, y) for drawing
            centerline_xy = np.column_stack([centerline[:, 1], centerline[:, 0]]).astype(int)
            
            # Draw centerline as connected points
            for i in range(len(centerline_xy) - 1):
                cv2.line(output_img, tuple(centerline_xy[i]), tuple(centerline_xy[i + 1]), color, 2)
            
            # Draw points
            for pt in centerline_xy:
                cv2.circle(output_img, tuple(pt), 2, color, -1)
            
            # Draw bounding box
            box = precomputed_data.sub_tract_boxes[name]
            x1, y1, x2, y2 = box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(output_img, name.split('_')[-1], (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            color_idx += 1
    
    return output_img

