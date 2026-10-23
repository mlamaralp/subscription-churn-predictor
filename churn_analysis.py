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


# ── Model 2 : XGBoost with Grid Search ────────────────────────────────────

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Hyperparameter grid : 3 x 3 x 2 x 2 = 36 combinations, each tested with 5-fold CV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 300],
    'subsample': [0.8, 1.0]
}

xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,   # handles the 73/27 class imbalance
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='roc_auc',     # optimise AUC-ROC, as in the README
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print(f"\nBest parameters   : {grid.best_params_}")
print(f"Best CV AUC-ROC   : {grid.best_score_:.4f}")

xgb = grid.best_estimator_
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("\nXGBoost (tuned) : Classification Report")
print(classification_report(y_test, y_pred_xgb, target_names=['Retained', 'Churned']))
print(f"AUC-ROC : {roc_auc_score(y_test, y_prob_xgb):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'],
            annot_kws={'size': 13})
axes[0].set_title('XGBoost : Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
axes[1].plot(fpr_lr, tpr_lr, color='#4c72b0', lw=2,
             label=f'Logistic Regression (AUC = {auc_lr:.3f})')
axes[1].plot(fpr_xgb, tpr_xgb, color='#4caf7d', lw=2,
             label=f"XGBoost (AUC = {roc_auc_score(y_test, y_prob_xgb):.3f})")
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve : Model Comparison', fontweight='bold')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('images/xgb_results.png', bbox_inches='tight')
plt.show()
print("Saved: images/xgb_results.png")

# ── Model comparison : Logistic Regression vs XGBoost ─────────────────────

# Compare evaluation metrics
def evaluate(name, y_true, y_pred, y_prob):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_prob)
    }

comparison = pd.DataFrame([
    evaluate('Logistic Regression', y_test, y_pred_lr, y_prob_lr),
    evaluate('XGBoost (tuned)', y_test, y_pred_xgb, y_prob_xgb)
]).set_index('Model').round(3)

print("\nModel comparison on the test set:")
print(comparison.to_string())

# ── Calibration check : are the predicted probabilities realistic? ────────

# scale_pos_weight helps the model detect churners, but it inflates the predicted probabilities. This table compares what the model predicted with what actually happened, by probability band.

def calibration_table(y_true, y_prob):
    bands = pd.cut(y_prob, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return (pd.DataFrame({'prob': y_prob, 'churned': np.asarray(y_true), 'band': bands})
            .groupby('band', observed=True)
            .agg(customers=('churned', 'size'),
                 predicted=('prob', 'mean'),
                 actual=('churned', 'mean'))
            .round(2))

print(f"\nWeighted XGBoost : mean predicted churn = {y_prob_xgb.mean():.1%} "
      f"vs actual churn rate = {y_test.mean():.1%}")
print(calibration_table(y_test, y_prob_xgb).to_string())

# Same tuned hyperparameters, but without class weights, for comparison
xgb_unweighted = XGBClassifier(**{**grid.best_params_, 'eval_metric': 'logloss',
                                  'random_state': 42, 'n_jobs': -1})
xgb_unweighted.fit(X_train, y_train)
y_prob_unw = xgb_unweighted.predict_proba(X_test)[:, 1]

print(f"\nUnweighted XGBoost : AUC-ROC = {roc_auc_score(y_test, y_prob_unw):.4f}, "
      f"mean predicted churn = {y_prob_unw.mean():.1%}")
print(calibration_table(y_test, y_prob_unw).to_string())

# ── Revenue at risk : prioritising customers by financial impact ──────────

# MonthlyCharges must contain the original monthly charge in dollars

if 'MonthlyCharges' not in X_test.columns:
    raise KeyError(
        "'MonthlyCharges' is missing from X_test. "
        "Use the original, unscaled monthly charges."
    )

if len(X_test) != len(y_prob_xgb) or len(X_test) != len(y_test):    
    raise ValueError("X_test, y_prob_xgb and y_test must have the same length.")

# revenue_at_risk = churn probability x monthly bill
# A customer with 80% churn risk paying $100/month matters more than one with 90% churn risk paying $20/month.
risk = X_test.copy()
risk['churn_probability'] = y_prob_xgb
risk['actual_churn'] = y_test.values
risk['revenue_at_risk'] = (
        risk['churn_probability'] * risk['MonthlyCharges']
)
risk = risk.sort_values('revenue_at_risk', ascending=False).reset_index(drop=True)

# Intervention tier : top 10% of customers by revenue at risk
top_n = int(len(risk) * 0.10)
top_tier = risk.head(top_n)

total_rar = risk['revenue_at_risk'].sum()
top_rar = top_tier['revenue_at_risk'].sum()

print(f"\nTotal monthly revenue at risk (test set) : ${total_rar:,.0f}")

print(f"Top 10% tier ({top_n} customers)           : ${top_rar:,.0f} "
      f"({top_rar / total_rar:.1%} of total)")

print(f"Churn rate inside the top 10% tier         : {top_tier['actual_churn'].mean():.1%} "
      f"(vs {risk['actual_churn'].mean():.1%} overall)")

print(f"Monthly charges associated with actual churn in top 10% tier:  : "
      f"${top_tier.loc[top_tier['actual_churn'] == 1, 'MonthlyCharges'].sum():,.0f}/month")

print("\nTop 10 customers by revenue at risk:")

print(risk[['churn_probability', 'MonthlyCharges', 'revenue_at_risk',
            'actual_churn']].head(10).round(2).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
cum_share = risk['revenue_at_risk'].cumsum() / total_rar * 100
pct_customers = (np.arange(1, len(risk) + 1) / len(risk)) * 100
ax.plot(pct_customers, cum_share, color='#e05c5c', lw=2)
ax.axvline(10, color='k', ls='--', lw=1)
ax.set_xlabel('% of customers (ranked by revenue at risk)')
ax.set_ylabel('% of total revenue at risk captured')
ax.set_title('Revenue at Risk : Cumulative Capture Curve', fontweight='bold')
ax.text(11, cum_share.iloc[top_n] - 8, f'Top 10% = {top_rar / total_rar:.0%} of revenue at risk',
        fontsize=10)
plt.tight_layout()
plt.savefig('images/revenue_at_risk.png', bbox_inches='tight')
plt.show()
print("Saved: images/revenue_at_risk.png")

# ── SHAP : explaining the XGBoost predictions ─────────────────────────────

import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Global view : which features matter most, and in which direction
plt.figure()
shap.summary_plot(shap_values, X_test, show=False, max_display=15)
plt.title('SHAP Summary : Drivers of Churn Prediction', fontweight='bold')
plt.tight_layout()
plt.savefig('images/shap_summary.png', bbox_inches='tight')
plt.show()
print("Saved: images/shap_summary.png")

# Ranking by mean absolute SHAP value
shap_importance = (
    pd.DataFrame({'feature': X_test.columns,
                  'mean_abs_shap': np.abs(shap_values).mean(axis=0)})
    .sort_values('mean_abs_shap', ascending=False)
)
print("\nTop 10 features by mean |SHAP|:")
print(shap_importance.head(10).round(4).to_string(index=False))
