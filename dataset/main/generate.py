import cv2
import numpy as np
import math
import random
import os

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb[::-1]  # Convert RGB to BGR

def create_irregular_tumor_mask(center_x, center_y, radius_px, shape):
    num_points = 100
    angle_step = 2 * np.pi / num_points
    max_perturbation = int(radius_px * 0.3)
    contour = []
    for i in range(num_points):
        angle = i * angle_step
        perturb = random.randint(-max_perturbation, max_perturbation)
        r = radius_px + perturb
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
        contour.append([x, y])
    contour = np.array([contour], dtype=np.int32)

    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], color=255)
    feathered = cv2.GaussianBlur(mask, (15, 15), sigmaX=5)
    alpha_mask = feathered.astype(np.float32) / 255.0
    return alpha_mask

def apply_tumor(image, alpha_mask, tumor_color, alpha_max=0.9):
    tumor_color_array = np.array(tumor_color, dtype=np.float32)
    image_with_tumor = image.copy().astype(np.float32)
    for c in range(3):
        image_with_tumor[:, :, c] = (
            (1 - alpha_mask * alpha_max) * image_with_tumor[:, :, c] +
            (alpha_mask * alpha_max) * tumor_color_array[c]
        )
    return np.clip(image_with_tumor, 0, 255).astype(np.uint8)

def generate_paired_tumor_images(
    tractography_path, flair_path, output_dir_tract, output_dir_flair,
    num_images=5, bbox=(174, 131, 420, 572), dpi=96, radius_cm=1.6
):
    os.makedirs(output_dir_tract, exist_ok=True)
    os.makedirs(output_dir_flair, exist_ok=True)

    tract_img = cv2.imread(tractography_path)
    flair_img = cv2.imread(flair_path)
    if tract_img is None or flair_img is None:
        raise FileNotFoundError("Could not load one or both input images.")

    for i in range(num_images):
        min_radius_px = int(0.5 * radius_cm * 0.393701 * dpi)
        max_radius_px = int(radius_cm * 0.393701 * dpi)
        radius_px = random.randint(min_radius_px, max_radius_px)
        center_x = random.randint(bbox[0] + radius_px, bbox[2] - radius_px)
        center_y = random.randint(bbox[1] + radius_px, bbox[3] - radius_px)

        alpha_mask = create_irregular_tumor_mask(center_x, center_y, radius_px, tract_img.shape)

        # Apply to dti (black tumor)
        tract_tumor = apply_tumor(tract_img, alpha_mask, tumor_color=(0, 0, 0), alpha_max=0.9)
        cv2.imwrite(os.path.join(output_dir_tract, f"{radius_px}_{center_x}_{center_y}.png"), tract_tumor)

        # Apply to flair (white tumor)
        flair_tumor = apply_tumor(flair_img, alpha_mask, tumor_color=(255, 255, 255), alpha_max=0.9)
        cv2.imwrite(os.path.join(output_dir_flair, f"{radius_px}_{center_x}_{center_y}.png"), flair_tumor)

# Step 3: Generate and download tumor images
tract_dir = "/dataset/main/dti-before"
flair_dir = "/dataset/main/flair-before"

# Generate paired tumor images
generate_paired_tumor_images(
    tractography_path="./tractography.jpg",
    flair_path="./flair.png",
    output_dir_tract=tract_dir,
    output_dir_flair=flair_dir,
    num_images=1000
)
