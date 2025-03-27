import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define file path in Google Drive
file_path = '/content/drive/My Drive/Colab Notebooks/process_control_data.csv'

# Create a synthetic dataset
timestamps = pd.date_range(start='2024-01-01', periods=100, freq='D')
measurements = np.random.normal(loc=50, scale=5, size=100)  # Normal distribution

df = pd.DataFrame({'timestamp': timestamps, 'measurement': measurements})

# Save dataset to CSV
df.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}")

# Load dataset
df = pd.read_csv(file_path)
display(df.head())

# Assuming dataset has columns 'timestamp', 'measurement'
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Compute control limits (3-sigma limits)
mean_val = df['measurement'].mean()
std_dev = df['measurement'].std()
UCL = mean_val + 3 * std_dev  # Upper Control Limit
LCL = mean_val - 3 * std_dev  # Lower Control Limit

# Enhanced Visualization
plt.figure(figsize=(14, 7))
sns.set_style("darkgrid")
sns.lineplot(x=df.index, y=df['measurement'], label='Measurements', color='blue', linewidth=2)
plt.axhline(mean_val, color='green', linestyle='dashed', linewidth=2, label='Mean')
plt.axhline(UCL, color='red', linestyle='dashed', linewidth=2, label='Upper Control Limit (UCL)')
plt.axhline(LCL, color='red', linestyle='dashed', linewidth=2, label='Lower Control Limit (LCL)')
sns.scatterplot(x=df.index, y=df['measurement'], color='blue', s=50, alpha=0.7)
plt.fill_between(df.index, LCL, UCL, color='red', alpha=0.1)
plt.legend()
plt.title('Statistical Process Control (SPC) Chart', fontsize=14, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Measurement', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Process Capability Analysis (Six Sigma Metrics)
usl, lsl = mean_val + 6 * std_dev, mean_val - 6 * std_dev  # 6-sigma limits
cp = (usl - lsl) / (6 * std_dev)  # Process Capability Index (Cp)
cpk = min((usl - mean_val) / (3 * std_dev), (mean_val - lsl) / (3 * std_dev))  # Adjusted Cpk

print(f"Process Capability Index (Cp): {cp:.2f}")
print(f"Process Performance Index (Cpk): {cpk:.2f}")

# Normality test
k2, p_value = stats.normaltest(df['measurement'])
print(f"Normality Test p-value: {p_value:.4f}")
if p_value > 0.05:
    print("Data follows a normal distribution.")
else:
    print("Data does not follow a normal distribution.")
