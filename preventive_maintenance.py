import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define the directory and file path
drive_path = '/content/drive/My Drive/Colab Notebooks/'
os.makedirs(drive_path, exist_ok=True)  # Ensure the directory exists
dataset_path = os.path.join(drive_path, 'predictive_maintenance_data.csv')

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
sensor1 = np.random.normal(loc=50, scale=10, size=n_samples)
sensor2 = np.random.normal(loc=60, scale=15, size=n_samples)
sensor3 = np.random.normal(loc=70, scale=20, size=n_samples)
failure = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'sensor1': sensor1,
    'sensor2': sensor2,
    'sensor3': sensor3,
    'failure': failure
})

# Save dataset to Google Drive
df.to_csv(dataset_path, index=False)
print(f"Dataset saved to {dataset_path}")

# Load dataset from Google Drive
df = pd.read_csv(dataset_path)

# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Check for missing values
df.fillna(method='ffill', inplace=True)

# Feature selection (assuming last column is the target 'failure')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForest Classifier with class balancing
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix Visualization with Explanation
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Adjust font color based on background intensity
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        text = ""
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        if i == 0 and j == 0:
            text = "True Negative (TN)\nCorrect No Failure"
        elif i == 0 and j == 1:
            text = "False Positive (FP)\nWrongly Predicted Failure"
        elif i == 1 and j == 0:
            text = "False Negative (FN)\nMissed Failure"
        elif i == 1 and j == 1:
            text = "True Positive (TP)\nCorrectly Predicted Failure"
        ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=9, color=color)

plt.show()

# Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,5))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
