# Data Preprocessing & ML Pipeline (Decode Labs - Project 2)

A robust Python-based pipeline for data preprocessing, exploratory data analysis (EDA), and machine learning predictions.

## 🚀 Project Overview
This project is part of **Decode Labs - Project 2**, focusing on foundational Data Science and Machine Learning workflows using Python. The primary objective is to take raw, unstructured dataset samples containing missing values and outliers, process them through a rigorous data cleaning pipeline, apply feature standardization, and build a predictive Machine Learning model using Linear Regression.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## ⚙️ Key Features & Workflow
* **Data Preprocessing:** Automated identification and imputation of missing values using median statistics.
* **Outlier Removal:** Filtered out extreme and unrealistic values to ensure high data quality.
* **Feature Scaling:** Standardized features using `StandardScaler` from `scikit-learn`.
* **Machine Learning Model:** Implemented a `LinearRegression` model for predictive tasks.

## 💻 Code Implementation
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset Creation & Loading
data = {
    'age': [25.0, 30.0, np.nan, 24.5, 28.0, 45.0, 120.0, 29.0, 31.0, np.nan],
    'salary': [50000.0, 60000.0, 55000.0, 57500.0, 52000.0, 80000.0, 900000.0, 53000.0, np.nan, 61000.0],
    'experience': [1, 5, 2, 0, 3, 15, 4, 2, 3, 1]
}

df = pd.DataFrame(data)

# Handling Missing Values and Outliers
df['age'] = df['age'].fillna(df['age'].median())
df['salary'] = df['salary'].fillna(df['salary'].median())
df = df[df['age'] <= 60]

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['age', 'salary', 'experience']])
df_scaled = pd.DataFrame(scaled_features, columns=['age', 'salary', 'experience'])

# Model Training
X = df_scaled[['age', 'experience']]
y = df_scaled['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted Salaries:", y_pred)
