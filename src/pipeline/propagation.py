"""
Progressive recurrence propagation along fiber pathways.
"""
import numpy as np
from . import config


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

    tumor_in_middle = any(name in config.Config.MIDDLE_TRACTS for name in at_risk_tracts)
    tumor_on_left = any(name in config.Config.LEFT_TRACTS for name in at_risk_tracts)
    tumor_on_right = any(name in config.Config.RIGHT_TRACTS for name in at_risk_tracts)
    
    if not tumor_in_middle:
        for mid_tract in config.Config.MIDDLE_TRACTS:
            if parent_features[mid_tract].get("proximity_score", 0) == 0:
                parent_features[mid_tract]["recurrence_score"] = 0
        
        if tumor_on_left and not tumor_on_right:
            for right_tract in config.Config.RIGHT_TRACTS:
                parent_features[right_tract]["recurrence_score"] = 0
        elif tumor_on_right and not tumor_on_left:
            for left_tract in config.Config.LEFT_TRACTS:
                parent_features[left_tract]["recurrence_score"] = 0
    
    active_tracts = [name for name, feat in parent_features.items() if feat["recurrence_score"] > 0]
    if not active_tracts:
        return parent_features, sub_features

    max_name = max(active_tracts, key=lambda x: parent_features[x]["recurrence_score"])
    
    # --- FIX #1: Use parent_features.keys() ---
    if not tumor_in_middle:
        if tumor_on_left: allowed_tracts = set(config.Config.LEFT_TRACTS)
        elif tumor_on_right: allowed_tracts = set(config.Config.RIGHT_TRACTS)
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
            decay_factor = np.exp(-config.Config.DECAY * new_cum_dist / 100)
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

