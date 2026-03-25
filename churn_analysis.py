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

service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
df['_num_services_preview'] = df[service_cols].apply(
    lambda row: sum(1 for v in row if v == 'Yes'), axis=1
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

svc_churn = df.groupby('_num_services_preview')['Churn_binary'].mean().mul(100)
axes[0].bar(svc_churn.index, svc_churn.values, color='#4c72b0', width=0.6)
axes[0].set_title('Churn Rate by Number of Services', fontweight='bold')
axes[0].set_xlabel('Number of Add-on Services')
axes[0].set_ylabel('Churn Rate (%)')
for i, v in enumerate(svc_churn.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_binary']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=axes[1], linewidths=0.5,
            annot_kws={'size': 11})
axes[1].set_title('Correlation Matrix : Numeric Features', fontweight='bold')

plt.tight_layout()
plt.savefig('images/eda_services_corr.png', bbox_inches='tight')
plt.show()
print("Saved: images/eda_services_corr.png")

df = df.drop(columns=['_num_services_preview'])

# ── Feature engineering ────────────────────────────────────────────────────

# tenure_group : bin raw tenure into lifecycle stages
tenure_bins = [0, 12, 24, 36, 48, 72]
tenure_labels = ['New (0-12m)', 'Developing (13-24m)', 'Established (25-36m)',
                 'Loyal (37-48m)', 'Long-term (49-72m)']
df['tenure_group'] = pd.cut(df['tenure'], bins=tenure_bins,
                             labels=tenure_labels, include_lowest=True)

tenure_churn = (
    df.groupby('tenure_group', observed=True)['Churn_binary']
    .mean()
    .mul(100)
    .round(1)
)
print("Churn rate by tenure group:")
print(tenure_churn.to_string())

# num_services : count of optional add-on services per customer
# Hypothesis: more services = higher switching cost = lower churn propensity
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
df['num_services'] = df[service_cols].apply(
    lambda row: sum(1 for v in row if v == 'Yes'), axis=1
)

services_churn = (
    df.groupby('num_services')['Churn_binary']
    .mean()
    .mul(100)
    .round(1)
)
print("Churn rate by number of services:")
print(services_churn.to_string())

# monthly_to_total_ratio : share of lifetime spend represented by current monthly bill
# High ratio = new customer with little billing history = lower inertia = higher churn risk
df['monthly_to_total_ratio'] = df['MonthlyCharges'] / df['TotalCharges'].replace(0, np.nan)
df['monthly_to_total_ratio'] = df['monthly_to_total_ratio'].fillna(1.0)

print(f"monthly_to_total_ratio summary:")
print(df['monthly_to_total_ratio'].describe().round(3))

# ── Encoding pipeline and train/test split ─────────────────────────────────

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df_model = df.copy()

binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'MultipleLines']

le = LabelEncoder()
for col in binary_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

df_model = pd.get_dummies(
    df_model,
    columns=['InternetService', 'Contract', 'PaymentMethod'],
    drop_first=False
)

df_model = df_model.drop(columns=['Churn', 'tenure_group'])

feature_cols = [c for c in df_model.columns if c != 'Churn_binary']
X = df_model[feature_cols]
y = df_model['Churn_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set : {X_train.shape[0]} records")
print(f"Test set     : {X_test.shape[0]} records")
print(f"Churn rate in training set : {y_train.mean():.1%}")
print(f"Churn rate in test set     : {y_test.mean():.1%}")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"XGBoost scale_pos_weight : {scale_pos_weight:.2f}")

# ── Model 1 : Logistic Regression (Baseline) ──────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score,
                              precision_score, recall_score, accuracy_score)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression : Classification Report")
print(classification_report(y_test, y_pred_lr, target_names=['Retained', 'Churned']))
print(f"AUC-ROC : {roc_auc_score(y_test, y_prob_lr):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'],
            annot_kws={'size': 13})
axes[0].set_title('Logistic Regression : Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
axes[1].plot(fpr_lr, tpr_lr, color='#4c72b0', lw=2,
             label=f'Logistic Regression (AUC = {auc_lr:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve : Logistic Regression Baseline', fontweight='bold')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('images/lr_results.png', bbox_inches='tight')
plt.show()
print("Saved: images/lr_results.png")
