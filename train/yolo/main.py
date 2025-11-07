# Install YOLOv8 and dependencies
!pip install ultralytics

# Define paths for the dataset
import os

# Replace 'your-dataset-name' with the name of your dataset directory
dataset_path = r"/dataset/yolo"
train_images = os.path.join(dataset_path, 'train/images')
val_images = os.path.join(dataset_path, 'val/images')
train_labels = os.path.join(dataset_path, 'train/labels')
val_labels = os.path.join(dataset_path, 'val/labels')

# Create a dataset.yaml file
dataset_yaml = f"""
path: {dataset_path}  # Base dataset path
train: train/images  # Train images (relative to 'path')
val: val/images      # Validation images (relative to 'path')

nc: 1                # Number of classes
names: ['tumor']     # Class names
"""

# Save the dataset.yaml file
yaml_path = os.path.join(dataset_path, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(dataset_yaml)

# Train YOLOv8
from ultralytics import YOLO

# Load a YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
model = YOLO('yolov8s.pt')  # You can replace 'yolov8s.pt' with other model variants

# Train the model
model.train(data=yaml_path, epochs=50, batch=16, imgsz=704)

# Evaluate the model
metrics = model.val()
print(metrics)