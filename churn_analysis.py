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

# ── EDA : churn rates by key categorical features ──────────────────────────

contract_churn = (
    df.groupby('Contract')['Churn_binary']
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
    .rename(columns={'Churn_binary': 'Churn Rate (%)'})
)
print("Churn rate by contract type:")
print(contract_churn.to_string(index=False))

payment_churn = (
    df.groupby('PaymentMethod')['Churn_binary']
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
    .rename(columns={'Churn_binary': 'Churn Rate (%)'})
    .sort_values('Churn Rate (%)', ascending=False)
)
print("\nChurn rate by payment method:")
print(payment_churn.to_string(index=False))

internet_churn = (
    df.groupby('InternetService')['Churn_binary']
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
    .rename(columns={'Churn_binary': 'Churn Rate (%)'})
    .sort_values('Churn Rate (%)', ascending=False)
)
print("\nChurn rate by internet service:")
print(internet_churn.to_string(index=False))

# ── EDA : distribution plots ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

contract_order = ['Month-to-month', 'One year', 'Two year']
contract_rates = (
    df.groupby('Contract')['Churn_binary']
    .mean()
    .mul(100)
    .reindex(contract_order)
)
bar_colors = ['#e05c5c', '#f0a500', '#4caf7d']
axes[0].bar(contract_rates.index, contract_rates.values, color=bar_colors, width=0.55)
axes[0].set_title('Churn Rate by Contract Type', fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].set_ylim(0, 55)
axes[0].set_xticklabels(contract_order, rotation=10, ha='right')
for i, v in enumerate(contract_rates.values):
    axes[0].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

axes[1].hist(df[df['Churn'] == 'No']['tenure'], bins=30,
             alpha=0.65, label='Retained', color='#4caf7d', edgecolor='white')
axes[1].hist(df[df['Churn'] == 'Yes']['tenure'], bins=30,
             alpha=0.65, label='Churned', color='#e05c5c', edgecolor='white')
axes[1].set_title('Tenure Distribution by Churn', fontweight='bold')
axes[1].set_xlabel('Tenure (months)')
axes[1].set_ylabel('Count')
axes[1].legend()

axes[2].hist(df[df['Churn'] == 'No']['MonthlyCharges'], bins=30,
             alpha=0.65, label='Retained', color='#4caf7d', edgecolor='white')
axes[2].hist(df[df['Churn'] == 'Yes']['MonthlyCharges'], bins=30,
             alpha=0.65, label='Churned', color='#e05c5c', edgecolor='white')
axes[2].set_title('Monthly Charges by Churn', fontweight='bold')
axes[2].set_xlabel('Monthly Charges ($)')
axes[2].set_ylabel('Count')
axes[2].legend()

plt.tight_layout()
plt.savefig('images/eda_distributions.png', bbox_inches='tight')
plt.show()
print("Saved: images/eda_distributions.png")
