from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Mount Google Drive
drive.mount('/content/drive')

# Generate Synthetic Dataset
data = {
    'temperature': np.random.normal(50, 10, 1000),
    'pressure': np.random.normal(100, 15, 1000),
    'vibration': np.random.normal(5, 2, 1000),
    'failure': np.random.choice([0, 1], size=1000, p=[0.8, 0.2])  # 80% no failure, 20% failure
}
df = pd.DataFrame(data)

# Save dataset to Google Drive
file_path = '/content/drive/My Drive/Colab Notebooks/machine_sensor_data.csv'
df.to_csv(file_path, index=False)

# Display first few rows
display(df.head())

# Exploratory Data Analysis (EDA)
print("Dataset Summary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Visualize distributions
plt.figure(figsize=(12, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Target Distribution
sns.countplot(x='failure', data=df)
plt.title('Failure Distribution')
plt.show()

# Feature Distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['temperature'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Temperature Distribution')
sns.histplot(df['pressure'], bins=30, kde=True, ax=axes[1])
axes[1].set_title('Pressure Distribution')
sns.histplot(df['vibration'], bins=30, kde=True, ax=axes[2])
axes[2].set_title('Vibration Distribution')
plt.show()

# Data Preprocessing
X = df.drop(columns=['failure'])  # Features
y = df['failure']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Machine Learning Model (Support Vector Machine)
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
