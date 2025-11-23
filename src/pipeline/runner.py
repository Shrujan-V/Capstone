"""
Main processing entry point for the tumor recurrence analysis pipeline.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from . import config
from . import geometry
from . import tract_graph
from . import tumor_processing
from . import boundary_analysis
from . import infiltration
from . import propagation
from . import visualization
from . import precomputed


def main():
    """Main processing loop."""
    # Build the 8-node graph ONCE
    tract_graph_obj, tract_centroids = tract_graph.build_tract_graph()
    
    data_records = []
    workspace_root = Path(__file__).parent.parent.parent
    tract_folder = workspace_root / "dataset" / "main" / "dti-before"
    flair_folder = workspace_root / "dataset" / "main" / "flair-before"
    results_folder = workspace_root / "results"
    heatmaps_folder = results_folder / "heatmaps"
    debug_folder = results_folder / "debug_vectors"
    stages_folder = results_folder / "stages"
    
    # Create separate folders for each stage
    stage1_folder = stages_folder / "stage1_tract_masks"
    stage2_folder = stages_folder / "stage2_tumor_boundary"
    stage3_folder = stages_folder / "stage3_tract_centerlines"
    stage4_folder = stages_folder / "stage4_radial_vectors"
    
    results_folder.mkdir(parents=True, exist_ok=True)
    heatmaps_folder.mkdir(parents=True, exist_ok=True)
    debug_folder.mkdir(parents=True, exist_ok=True)
    stages_folder.mkdir(parents=True, exist_ok=True)
    stage1_folder.mkdir(parents=True, exist_ok=True)
    stage2_folder.mkdir(parents=True, exist_ok=True)
    stage3_folder.mkdir(parents=True, exist_ok=True)
    stage4_folder.mkdir(parents=True, exist_ok=True)
    
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
        
        # Precompute all heavy operations once per image
        precomputed_data = precomputed.precompute_image_data(dti_rgb, flair_img)
        if precomputed_data is None:
            print("   No tumor found or failed to precompute data")
            idx += 1
            continue
        
        print(f"   Tumor: {np.count_nonzero(precomputed_data.tumor_mask)} pixels")
        
        tumor_bbox = geometry.get_bbox(precomputed_data.tumor_mask)
        
        parent_features = {} # For the 8 parent tracts
        sub_features = {}    # For all sub-tracts
        dominant_vectors_dict = {}
        analyzed_points_dict = {}  # Store analyzed points for each tract for visualization
        
        print("   Computing soft-contact proximity scores...")
        
        for name, box in config.Config.ORIGINAL_TRACTS.items():
            parent_features[name] = {
                "bbox": box,
                "iou": geometry.iou(tumor_bbox, box),
                "initial_score": 0,
                "recurrence_score": 0,
                "proximity_score": 0
            }
        
        # Use precomputed data instead of recomputing
        for name in precomputed_data.sub_tract_boxes.keys():
            tract_mask = precomputed_data.tract_masks[name]
            tract_centerline = precomputed_data.tract_centerlines[name]
            box = precomputed_data.sub_tract_boxes[name]
            
            min_dist, proximity_score, risk_tier = boundary_analysis.compute_tract_proximity_score(
                tract_mask, precomputed_data.distance_map
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
            
            if proximity_score == 0 or min_dist > config.Config.DIST_NO_RISK:
                continue
            
            analyzed_points = boundary_analysis.analyze_boundary_features_for_tract(
                precomputed_data.tumor_mask, precomputed_data.tumor_centroid, precomputed_data.tumor_boundary,
                tract_centerline, precomputed_data.distance_map
            )
            
            # Store analyzed points for visualization
            analyzed_points_dict[name] = analyzed_points
            
            if len(analyzed_points) == 0:
                print(f"       No directionally-aligned boundary points found")
                continue
            
            dominant_vector, initial_score = infiltration.get_dominant_infiltration_vector(
                analyzed_points, tract_centerline
            )
            
            if dominant_vector:
                weighted_score = initial_score * proximity_score
                sub_features[name]["initial_score"] = weighted_score
                dominant_vectors_dict[name] = dominant_vector
                print(f"       Initial score: {weighted_score:.3f}")

        print("   Propagating along fiber pathways...")
        parent_features, sub_features = propagation.propagate_recurrence_progressive(
            parent_features, sub_features, tract_graph_obj
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
            parent_name = precomputed_data.parent_map[name]
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
        
        # Save stage visualizations
        print("   Saving stage visualizations...")
        
        # Stage 1: Tract masks
        tract_masks_img = visualization.visualize_tract_masks(dti_rgb, precomputed_data)
        stage1_path = stage1_folder / f"{flair_path.stem}.png"
        cv2.imwrite(str(stage1_path), cv2.cvtColor(tract_masks_img, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved Stage 1: Tract masks")
        
        # Stage 2: Tumor boundary
        tumor_boundary_img = visualization.visualize_tumor_boundary(dti_rgb, precomputed_data)
        stage2_path = stage2_folder / f"{flair_path.stem}.png"
        cv2.imwrite(str(stage2_path), cv2.cvtColor(tumor_boundary_img, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved Stage 2: Tumor boundary")
        
        # Stage 3: Tract centerlines
        tract_centerlines_img = visualization.visualize_tract_centerlines(dti_rgb, precomputed_data)
        stage3_path = stage3_folder / f"{flair_path.stem}.png"
        cv2.imwrite(str(stage3_path), cv2.cvtColor(tract_centerlines_img, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved Stage 3: Tract centerlines")
        
        # Stage 4: Radial vectors and boundary-to-tract vectors
        vectors_img = visualization.visualize_radial_and_boundary_vectors(
            dti_rgb, precomputed_data, analyzed_points_dict
        )
        stage4_path = stage4_folder / f"{flair_path.stem}.png"
        cv2.imwrite(str(stage4_path), cv2.cvtColor(vectors_img, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved Stage 4: Radial and boundary-to-tract vectors")
        
        print("   Rendering heatmap...")
        hotspot_img = visualization.create_progressive_blob_heatmap(dti_rgb, final_hotspots, dti_rgb.shape)
        
        tumor_contours, _ = cv2.findContours(precomputed_data.tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(hotspot_img, tumor_contours, -1, (255, 255, 0), 2)
        
        # Use total_score_sum (pre-normalization) for the label if you want to see values > 100%
        # Or use display_score (1.0) to show that risk is present
        hotspot_img = visualization.add_recurrence_label(hotspot_img, display_score) # Changed to display_score
        
        # --- MODIFIED: Save Image A ---
        output_filename = flair_path.stem + "_refined_confidence_prediction.png"
        output_path = heatmaps_folder / output_filename
        cv2.imwrite(str(output_path), cv2.cvtColor(hotspot_img, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved {output_filename}")
        
        # --- MODIFIED: Generate debug visualization (Image B) ---
        debug_img = visualization.create_debug_visualization(
            dti_rgb, sub_features, dominant_vectors_dict, precomputed_data.tumor_mask
        )
        
        # --- MODIFIED: Save Image B ---
        debug_filename = flair_path.stem + "_vectors.png" # New filename
        debug_path = debug_folder / debug_filename
        cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved {debug_filename}")
        
        # Store data records
        for name, feat in sub_features.items(): # Save sub-tract data
            parent_name = precomputed_data.parent_map[name]
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
    print(f"   Stage visualizations:")
    print(f"     - Stage 1 (Tract masks): {stage1_folder}")
    print(f"     - Stage 2 (Tumor boundary): {stage2_folder}")
    print(f"     - Stage 3 (Tract centerlines): {stage3_folder}")
    print(f"     - Stage 4 (Radial vectors): {stage4_folder}")
    print(f"   Analysis CSV: {results_folder / 'soft_contact_recurrence_analysis.csv'}")
    print(f"\n💡 Key improvements:")
    print(f"   ✓ Sub-tract segmentation for localized risk")
    print(f"   ✓ Soft-contact proximity scoring (0-15px risk stratification)")
    print(f"   ✓ Directional filtering (radial vectors must point toward tracts)")
    print(f"   ✓ Anatomical snapping (hotspots placed on tract centerlines)")
    print(f"   ✓ Progressive propagation (risk spreads along fiber pathways)")
    print(f"   ✓ Two-stage blob rendering (smooth blobs + highlighted cores)")


if __name__ == "__main__":
    main()

