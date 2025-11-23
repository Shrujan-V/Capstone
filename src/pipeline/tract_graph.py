"""
Tract graph construction for propagation.
"""
import networkx as nx
from . import geometry
from . import config


def build_tract_graph():
    G = nx.Graph()
    centroids = {}
    for name, bbox in config.Config.ORIGINAL_TRACTS.items():
        centroid = geometry.get_bbox_centroid(bbox)
        centroids[name] = centroid
        G.add_node(name, pos=centroid)
    
    for i, tract_i in enumerate(config.Config.LEFT_TRACTS):
        for tract_j in config.Config.LEFT_TRACTS[i+1:]:
            dist = geometry.euclidean_distance(centroids[tract_i], centroids[tract_j])
            G.add_edge(tract_i, tract_j, weight=dist)
    
    for i, tract_i in enumerate(config.Config.RIGHT_TRACTS):
        for tract_j in config.Config.RIGHT_TRACTS[i+1:]:
            dist = geometry.euclidean_distance(centroids[tract_i], centroids[tract_j])
            G.add_edge(tract_i, tract_j, weight=dist)
    
    top_tracts = ["tract_1", "tract_7"]
    for top_tract in top_tracts:
        dist = geometry.euclidean_distance(centroids[top_tract], centroids["tract_8"])
        G.add_edge(top_tract, "tract_8", weight=dist)
    
    bottom_tracts = ["tract_3", "tract_5"]
    for bottom_tract in bottom_tracts:
        dist = geometry.euclidean_distance(centroids[bottom_tract], centroids["tract_4"])
        G.add_edge(bottom_tract, "tract_4", weight=dist)
    
    return G, centroids

