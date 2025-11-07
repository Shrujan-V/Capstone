import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("D:\\Capstone\\results\\tumor_tract_recurrence_graph_based.csv")

# Set image_group to be the same as the image name (image_id)
df['image_group'] = df['image_id']

# Drop the image_id column as it's no longer needed
df.drop(columns=['image_id'], inplace=True)

# Split into train and test sets (70% train, 30% test)
unique_images = df['image_group'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.3, random_state=42)

train_df = df[df['image_group'].isin(train_images)].copy()
test_df = df[df['image_group'].isin(test_images)].copy()

# Label 35% of the training set
np.random.seed(42)
train_images_labeled = np.random.choice(train_images, size=int(0.35 * len(train_images)), replace=False)

train_df['initial_label'] = -1  # Default to -1
for image_group in train_images_labeled:
    image_tracts = train_df[train_df['image_group'] == image_group]
    
    # Select top 2 recurrence scores
    top2_indices = image_tracts.nlargest(2, 'recurrence_score').index
    
    # Assign labels
    train_df.loc[top2_indices, 'initial_label'] = 1
    train_df.loc[image_tracts.index.difference(top2_indices), 'initial_label'] = 0

# Prepare feature matrix and labels for training
X_train = train_df[['iou', 'alignment', 'recurrence_score']].values
y_train = train_df['initial_label'].values

# Fit the Label Spreading model
label_spread = LabelSpreading(kernel='rbf', alpha=0.2)
label_spread.fit(X_train, y_train)

# Predict labels for all tracts in the training set
train_df['label_spread_prediction'] = label_spread.transduction_

# Save the model
joblib.dump(label_spread, "label_spreading_model.pkl")

# Load the saved model
loaded_model = joblib.load("label_spreading_model.pkl")

# Prepare the test set feature matrix
X_test = test_df[['iou', 'alignment', 'recurrence_score']].values

# Predict labels for the test set
test_df['label_spread_prediction'] = loaded_model.predict(X_test)

# Save the results without shuffling
train_df.to_csv("train_label_spreading_results.csv", index=False)
test_df.to_csv("test_label_spreading_results.csv", index=False)

print("✅ Label spreading predictions saved to train_label_spreading_results.csv and test_label_spreading_results.csv")