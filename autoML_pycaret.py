# 🚀 Install & Upgrade Required Libraries
!pip install --upgrade scipy numpy pandas matplotlib seaborn pycaret

# 🚀 Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import *

# 🚀 Generate Sample Demand Data
dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
demand = np.random.randint(50, 200, size=(100,))

df = pd.DataFrame({"Date": dates, "Demand": demand})

# 📊 Visualizing Demand Trends
plt.figure(figsize=(12,6))
sns.lineplot(x=df['Date'], y=df['Demand'], label='Demand Trend')
plt.title('Demand Over Time')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.show()

# 📌 Feature Engineering - Creating Lag Features
df['Lag_1'] = df['Demand'].shift(1)
df['Lag_2'] = df['Demand'].shift(2)
df['Lag_3'] = df['Demand'].shift(3)

# Drop missing values
df.dropna(inplace=True)

# 📌 Initialize AutoML with PyCaret (Fixed)
exp1 = setup(data=df, target='Demand', session_id=42, verbose=False)

# 🚀 Train Multiple Models & Select Best One
best_model = compare_models()

# 🚀 Predict Future Demand Using Best Model
predictions = predict_model(best_model, data=df)

# ✅ Fixing KeyError by Using 'prediction_label'
plt.figure(figsize=(12,6))
plt.plot(df['Demand'].values, label="Actual Demand", linestyle='dashed')
plt.plot(predictions['prediction_label'].values, label="Predicted Demand", alpha=0.8)
plt.title('Actual vs. Predicted Demand using AutoML (PyCaret)')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.show()
