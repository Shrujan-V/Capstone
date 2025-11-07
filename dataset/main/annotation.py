import cv2
import numpy as np
import math
import random
import os
import pandas as pd

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
    tractography_path, flair_path, output_dir_tract, output_dir_flair, bbox=(174, 131, 420, 572), dpi=96, radius_cm=1.6
):
    os.makedirs(output_dir_tract, exist_ok=True)
    os.makedirs(output_dir_flair, exist_ok=True)

    tract_img = cv2.imread(tractography_path)
    flair_img = cv2.imread(flair_path)
    if tract_img is None or flair_img is None:
        raise FileNotFoundError("Could not load one or both input images.")

    for _, row in data.iterrows():
        image_name = row["image_name"] + ".png"
        _, x_center, y_center, width, height = map(float, row["coordinates"].split())
        img_h, img_w, _ = flair_img.shape
        print(img_h, img_w)
        x_center = int(x_center * img_w)
        y_center = int(y_center * img_h)
        width = int(width * img_w)
        height = int(height * img_h)

        radius_px = min(width, height) // 2
        center_x = x_center
        center_y = y_center

        alpha_mask = create_irregular_tumor_mask(center_x, center_y, radius_px, tract_img.shape)

        # Apply to dti (black tumor)
        tract_tumor = apply_tumor(tract_img, alpha_mask, tumor_color=(0, 0, 0), alpha_max=0.9)
        cv2.imwrite(os.path.join(output_dir_tract, image_name), tract_tumor)

        # Apply to flair (white tumor)
        flair_tumor = apply_tumor(flair_img, alpha_mask, tumor_color=(255, 255, 255), alpha_max=0.9)
        cv2.imwrite(os.path.join(output_dir_flair, image_name), flair_tumor)

# Step 3: Generate and download tumor images

# Output folders
tract_dir = "/dataset/main/dti-after"
flair_dir = "/dataset/main/flair-after"

csv_path = r"reannotation_labels_250.csv"
data = pd.read_csv(csv_path)

# Generate paired tumor images
generate_paired_tumor_images(
    tractography_path="/images/dti.jpg",
    flair_path="/images/flair.png",
    output_dir_tract=tract_dir,
    output_dir_flair=flair_dir
)
