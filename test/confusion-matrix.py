import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load the ground truth file
ground_truth_file = "D:\\Capstone\\test\\ground-truth.csv"
ground_truth_df = pd.read_csv(ground_truth_file)
test_df = pd.read_csv("D:\\Capstone\\test_label_spreading_results.csv")

# Filter ground truth to include only images in the test set
filtered_ground_truth = ground_truth_df[ground_truth_df['image_id'].isin(test_df['image_id'].unique())]

# Merge the test predictions with the filtered ground truth
merged_df = pd.merge(test_df, filtered_ground_truth, on=['image_id', 'tract_id'])

# Extract ground truth labels and predictions
y_true = merged_df['iou_binary']  # Ground truth labels
y_pred = merged_df['label_spread_prediction']  # Model predictions

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Optionally, print a classification report for additional metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred))