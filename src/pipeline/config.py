"""
Configuration constants and tract definitions.
"""


class Config:
    """Configuration class containing all constants and tract definitions."""
    
    # Tumor detection
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
    
    # Original tract definitions
    ORIGINAL_TRACTS = {
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

