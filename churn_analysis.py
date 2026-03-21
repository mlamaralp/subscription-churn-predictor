import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

os.makedirs('images', exist_ok=True)

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Shape: {df.shape}")
print(f"\nChurn distribution:")
print(df['Churn'].value_counts())
print(f"\nOverall churn rate: {(df['Churn'] == 'Yes').mean():.1%}")

print("Column types:")
print(df.dtypes)
print(f"\nNull values per column:")
print(df.isnull().sum())
print(f"\nTotalCharges blank strings: {(df['TotalCharges'].str.strip() == '').sum()}")

# TotalCharges contains blank strings for 11 customers with tenure = 0
# These are new accounts with no completed billing cycle.
# The correct value is 0, not NaN : these are not missing observations.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# customerID is a row identifier with no predictive signal
df = df.drop(columns=['customerID'])

# Binary target column
df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)

print(f"Dataset after cleaning: {df.shape}")
print(f"Null values remaining: {df.isnull().sum().sum()}")
print(f"Churn rate: {df['Churn_binary'].mean():.1%}")
