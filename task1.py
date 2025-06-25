

# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------
# Step 1: Load the Iris Dataset
# -------------------------
# The Iris dataset is a built-in dataset in scikit-learn. It includes flower features and species (target).
iris = load_iris()

# Convert the dataset to a pandas DataFrame for easy manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column (species) to the DataFrame
df['target'] = iris.target

# Display first few rows to understand structure
print("Initial Data Sample:\n", df.head())

# -------------------------
# Step 2: Introduce Missing Values (for demonstration purposes)
# -------------------------
# To simulate real-world messy data, we'll insert some missing (NaN) values
df.iloc[1, 2] = np.nan  # Introduce NaN in 2nd row, 3rd column
df.iloc[4, 0] = np.nan  # Introduce NaN in 5th row, 1st column

# -------------------------
# Step 3: Handle Missing Values
# -------------------------
# Fill missing numeric values using the column mean
df.fillna(df.mean(numeric_only=True), inplace=True)
# -------------------------
# Step 4: Encode Target Labels
# -------------------------
# In real datasets, labels may be strings, so we use LabelEncoder for consistency
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# -------------------------
# Step 5: Feature Scaling
# -------------------------
# Extract only the feature columns (exclude target)
features = df.drop('target', axis=1)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features (zero mean, unit variance)
scaled_features = scaler.fit_transform(features)

# -------------------------
# Step 6: Split into Training and Testing Sets
# -------------------------
# Use 80% of data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, df['target'], test_size=0.2, random_state=42
)

# -------------------------
# Step 7: Save Processed Data to CSV
# -------------------------
# Convert training and testing arrays back to DataFrames
train_df = pd.DataFrame(X_train, columns=features.columns)
train_df['target'] = y_train.values

test_df = pd.DataFrame(X_test, columns=features.columns)
test_df['target'] = y_test.values

# Save the processed DataFrames as CSV files
train_df.to_csv('processed_train_data.csv', index=False)
test_df.to_csv('processed_test_data.csv', index=False)

# -------------------------
# Completion Message
# -------------------------
print("\nETL Pipeline Completed. Files saved as 'processed_train_data.csv' and 'processed_test_data.csv'")