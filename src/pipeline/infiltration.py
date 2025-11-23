"""
Dominant infiltration vector computation.
"""
from . import config
from . import boundary_analysis


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
            (prox_score * config.Config.W_PROXIMITY) +
            (dir_score * config.Config.W_DIRECTION) +
            (bulge_score * config.Config.W_BULGE) +
            (density_score * config.Config.W_DENSITY)
        )
        
        if final_score > best_score:
            best_score = final_score
            
            nearest_tract_point, _ = boundary_analysis.find_nearest_tract_point(
                point_data["point"], tract_centerline
            )
            
            best_vector = {
                "source_point": nearest_tract_point if nearest_tract_point is not None else point_data["point"],
                "direction": point_data["radial_vector"],
                "score": final_score,
                "dist_to_tract": point_data["dist_to_tract"]
            }
    
    return best_vector, best_score

