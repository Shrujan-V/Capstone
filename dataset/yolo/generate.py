import cv2
import numpy as np
import math
import random
import os

# Constants
IMG_HEIGHT = 700  # Image height
IMG_WIDTH = 540   # Image width
NUM_TRAIN = 3200
NUM_VAL = 800
dpi = 96
radius_cm = 1.6
MIN_RADIUS = int(0.5 * radius_cm * 0.393701 * dpi)
MAX_RADIUS = int(radius_cm * 0.393701 * dpi)

# Directories
OUTPUT_DIR = r"D:\\codebase\\dataset"
TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(OUTPUT_DIR, "train", "labels")
TRAIN_BBOX_IMG_DIR = os.path.join(OUTPUT_DIR, "train", "bbox_images")
VAL_IMG_DIR = os.path.join(OUTPUT_DIR, "val", "images")
VAL_LABEL_DIR = os.path.join(OUTPUT_DIR, "val", "labels")
VAL_BBOX_IMG_DIR = os.path.join(OUTPUT_DIR, "val", "bbox_images")

# Ensure directories exist
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(TRAIN_BBOX_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_BBOX_IMG_DIR, exist_ok=True)

def create_irregular_tumor_mask(center_x, center_y, radius_px, shape):
    """
    Creates an irregular tumor mask with random perturbations.
    """
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
    """
    Applies the tumor mask to the image with transparency.
    """
    tumor_color_array = np.array(tumor_color, dtype=np.float32)
    image_with_tumor = image.copy().astype(np.float32)
    for c in range(3):
        image_with_tumor[:, :, c] = (
            (1 - alpha_mask * alpha_max) * image_with_tumor[:, :, c] +
            (alpha_mask * alpha_max) * tumor_color_array[c]
        )
    return np.clip(image_with_tumor, 0, 255).astype(np.uint8)

def save_image_and_label(image, mask, bbox, img_path, label_path):
    """
    Saves the image and its corresponding YOLO label.
    """
    # Combine the tumor mask with the black background
    tumor_image = apply_tumor(image, mask, tumor_color=(255, 255, 255))

    # Save the image
    cv2.imwrite(img_path, tumor_image)

    # Save the label in YOLO format
    with open(label_path, "w") as f:
        f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")  # Class ID is 0 for tumor

def draw_bounding_box(image_path, label_path, output_path):
    """
    Draws bounding boxes on the image using YOLO format labels and saves the result.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Read the YOLO label
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Parse YOLO format: class_id x_center y_center width height
        _, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert YOLO format to pixel coordinates
        x_center_px = int(x_center * IMG_WIDTH)
        y_center_px = int(y_center * IMG_HEIGHT)
        width_px = int(width * IMG_WIDTH)
        height_px = int(height * IMG_HEIGHT)

        x_min = max(0, int(x_center_px - width_px / 2))
        y_min = max(0, int(y_center_px - height_px / 2))
        x_max = min(IMG_WIDTH, int(x_center_px + width_px / 2))
        y_max = min(IMG_HEIGHT, int(y_center_px + height_px / 2))

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Save the image with bounding box
    cv2.imwrite(output_path, image)

def generate_dataset(num_images, img_dir, label_dir, bbox_img_dir):
    """
    Generates a dataset of images with random irregular tumors and YOLO labels.
    """
    for i in range(num_images):
        # Create a black image
        image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        # Generate a random tumor
        radius_px = random.randint(MIN_RADIUS, MAX_RADIUS)
        center_x = random.randint(radius_px, IMG_WIDTH - radius_px)
        center_y = random.randint(radius_px, IMG_HEIGHT - radius_px)
        mask = create_irregular_tumor_mask(center_x, center_y, radius_px, image.shape)

        # Calculate bounding box
        y, x = np.where(mask > 0)
        x_min, y_min, x_max, y_max = min(x), min(y), max(x), max(y)
        x_center = (x_min + x_max) / 2 / IMG_WIDTH
        y_center = (y_min + y_max) / 2 / IMG_HEIGHT
        width = (x_max - x_min) / IMG_WIDTH
        height = (y_max - y_min) / IMG_HEIGHT
        bbox = (x_center, y_center, width, height)

        # File paths
        img_path = os.path.join(img_dir, f"image_{i:04d}.jpg")
        label_path = os.path.join(label_dir, f"image_{i:04d}.txt")
        bbox_img_path = os.path.join(bbox_img_dir, f"image_{i:04d}.jpg")

        # Save the image and label
        save_image_and_label(image, mask, bbox, img_path, label_path)

        # Draw bounding box and save the image
        draw_bounding_box(img_path, label_path, bbox_img_path)

# Generate training and validation datasets
generate_dataset(NUM_TRAIN, TRAIN_IMG_DIR, TRAIN_LABEL_DIR, TRAIN_BBOX_IMG_DIR)
generate_dataset(NUM_VAL, VAL_IMG_DIR, VAL_LABEL_DIR, VAL_BBOX_IMG_DIR)

print("✅ Dataset generation with bounding boxes complete!")