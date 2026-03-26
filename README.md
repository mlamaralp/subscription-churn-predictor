# Subscription Churn Prediction : Revenue-Weighted Retention Prioritisation

Predicting customer churn and quantifying the expected revenue at risk for each account, using the IBM Telco Customer Churn dataset (7,043 customers, 20 features). The goal is not just to classify who will leave, but to rank customers by the financial cost of losing them : turning a binary classifier into a prioritisation tool for retention teams.

---

## Why This Problem

Every major technology company that operates a subscription product runs some version of this analysis. Microsoft 365, Xbox Game Pass, and Azure all depend on recurring revenue that evaporates the moment a customer cancels. Google One, YouTube Premium, and Google Workspace face the same problem. Churn prediction is not an academic exercise at these companies: it is a live system that feeds retention campaigns, pricing decisions, and product roadmaps.

The standard approach : build a classifier, report accuracy, stop : does not answer the question a product or revenue team actually needs to act on: which customers should we contact first, and what is the financial cost of getting it wrong? This project adds that layer. Each customer's churn probability is multiplied by their monthly charges to produce an expected monthly revenue at risk score, which re-ranks the output of the model into a prioritised intervention list. A customer with a 60% churn probability and $100/month in charges is worth more to retain than one with an 80% probability and $20/month.

The framing is intentional: this is the kind of thinking that distinguishes a data scientist who understands business impact from one who optimises metrics in isolation.

---

## Research Questions

1. **Which customer and contract characteristics are the strongest predictors of churn?** Across demographics, service subscriptions, billing type, and contract length, which features drive the classification most and in which direction?
2. **How does a gradient-boosted model compare to a logistic regression baseline on an imbalanced binary target?** With a 26.5% churn rate, what does a meaningful gain in recall look like in practice, and what does it cost in false positives?
3. **Can churn probability be combined with revenue data to produce an actionable prioritisation score?** Does re-ranking customers by expected revenue at risk change which accounts a retention team would contact first?

---

## Dataset

**Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) on Kaggle (IBM sample dataset).

The dataset contains 7,043 customer records across 20 features covering demographics, subscribed services, contract and billing details, and a binary churn label. It was chosen because its feature structure : contract type, payment method, tenure, monthly charges : maps directly onto the subscription dynamics of any recurring-revenue product, including the Microsoft 365 and Google One ecosystems. The known data quality issue (11 blank values in TotalCharges for customers with zero tenure) is documented and handled explicitly in preprocessing.

| Feature group | Features |
|---|---|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Billing | MonthlyCharges, TotalCharges |
| Target | Churn (Yes / No) |

Overall churn rate: 26.5% (1,869 out of 7,043 customers). Moderately imbalanced : addressed through class weighting rather than resampling.

---

## Repository Structure

```
subscription-churn/
├── churn_analysis.ipynb         # Full pipeline: EDA, feature engineering, modelling, SHAP, revenue scoring
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── images/                      # Generated automatically on first run
└── requirements.txt
```

---

## Tools

| Area | Stack |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Modelling | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Environment | Jupyter Notebook |

---

## Methodology

### Data Cleaning

TotalCharges contains 11 blank string values corresponding to customers with zero tenure (new accounts with no billing history). These were converted to 0 rather than dropped or imputed with the column mean, since zero is the correct value: a customer who has never completed a billing cycle has zero total charges. Dropping these rows would silently remove new customers from the dataset and bias the tenure distribution.

CustomerID was dropped before modelling as a row identifier with no predictive signal. SeniorCitizen was already encoded as a binary integer (0/1) and required no transformation. All remaining binary Yes/No columns were label-encoded. InternetService, Contract, and PaymentMethod : the three multi-class categorical features : were one-hot encoded. Class imbalance (26.5% churn) was addressed through class_weight='balanced' in Logistic Regression and scale_pos_weight in XGBoost, rather than SMOTE or other resampling. With a 26.5% minority class, synthetic oversampling introduces less value than proper class weighting and avoids the risk of generating implausible combinations of contract type, tenure, and billing values.

### Feature Engineering

**tenure_group:** Tenure in months (range: 0–72) was binned into five groups of roughly equal width: New (0–12), Developing (13–24), Established (25–36), Loyal (37–48), and Long-term (49–72). The raw numeric tenure was retained for the model alongside the grouped version. The grouping serves two purposes: it enables the EDA to show churn rates by customer lifecycle stage cleanly, and it gives the model a coarser representation that is less sensitive to month-to-month noise in the lower tenure range where churn is most volatile.

**num_services:** A count of how many of the seven optional services each customer has subscribed to (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, MultipleLines). The hypothesis is that customers embedded in more services have a higher switching cost and lower churn propensity : a pattern common in bundled subscription products. This feature captures product stickiness as a single number, which the model can use directly rather than having to learn the relationship across seven binary columns independently.

**monthly_to_total_ratio:** MonthlyCharges divided by TotalCharges, representing the share of the customer's total spend that their most recent bill represents. For long-tenure customers this ratio approaches zero; for new customers it approaches one. It is a proxy for tenure without being collinear with raw tenure, since it also captures billing history and is influenced by plan changes over time. Customers with a high ratio : whose recent bill is a large fraction of what they have ever paid : have had little time to build inertia with the service and are more likely to churn.

### Model 1: Logistic Regression (Baseline)

Logistic Regression provides a transparent, interpretable baseline. Its coefficients directly represent log-odds, making it possible to verify that the model has learned directionally sensible relationships before moving to a more complex approach (for example, confirming that month-to-month contracts increase churn log-odds relative to two-year contracts). Parameters: max_iter=1000, class_weight='balanced', random_state=42.

The structural limitation here is the same as in any linear model applied to subscription data: the interaction between low tenure and month-to-month contract type is likely stronger than either variable independently, and a linear boundary cannot capture that interaction without an explicit interaction term.

### Model 2: XGBoost with Grid Search

XGBoost was chosen over Random Forest as the second model because gradient-boosted trees are the industry standard for tabular churn prediction and are explicitly optimised for the kind of moderate imbalance present here. They also generalise better than Random Forest on datasets of this size (roughly 5,600 training records) because they correct residuals iteratively rather than averaging independent trees, which tends to reduce variance more efficiently in lower-data regimes.

A grid search was run over learning_rate (0.01, 0.1, 0.3), max_depth (3, 5, 7), n_estimators (100, 300), and subsample (0.8, 1.0), scored using 5-fold cross-validation on AUC-ROC. AUC-ROC was chosen as the optimisation target rather than F1 because it is threshold-independent: it measures the model's ability to rank churners above non-churners across all possible classification thresholds, which is exactly what the downstream revenue-weighted scoring step requires.

**Why AUC-ROC and not accuracy or F1?** With 26.5% churn, accuracy is dominated by the majority class and rewards a model that is never bold enough to predict churn. F1 requires choosing a threshold before optimisation. AUC-ROC makes no threshold assumption and directly evaluates the quality of the probability scores the model outputs : which is what matters when the goal is to rank customers, not just classify them.

### Revenue-Weighted Prioritisation Score

After the final model produces a churn probability for each customer, a revenue at risk score is computed as:

```
revenue_at_risk = churn_probability × MonthlyCharges
```

This score re-ranks the customer list in a way that reflects business impact rather than model confidence. It answers the question a retention team actually faces: given a fixed number of outreach slots, which customers should we contact first? A customer with a 55% churn probability paying $95/month has a revenue at risk score of $52.25. A customer with a 90% churn probability paying $20/month has a score of $18.00. Pure churn probability would prioritise the second customer; revenue at risk correctly prioritises the first.

The top 10% of customers ranked by revenue at risk are identified as the primary intervention tier: the accounts where a retention offer has the highest expected return.

---

## Key Findings

| Metric | Logistic Regression | XGBoost (tuned) |
|---|---|---|
| Accuracy | 0.791 | 0.823 |
| Precision (churn class) | 0.641 | 0.689 |
| Recall (churn class) | 0.762 | 0.798 |
| F1 (churn class) | 0.696 | 0.740 |
| AUC-ROC | 0.848 | 0.887 |

XGBoost was selected as the final model. The gain in AUC-ROC (+0.039) reflects a meaningfully better probability ranking, which directly improves the quality of the revenue-weighted prioritisation step. In practical terms: for every 100 customers flagged for retention outreach, XGBoost correctly identifies roughly 8 more true churners than Logistic Regression.

**EDA findings that shaped the analysis:**

- Month-to-month contracts have a churn rate of approximately 43%, compared to 11% for one-year and under 3% for two-year contracts. Contract type is the single strongest predictor in the dataset.
- Electronic check payment is associated with a churn rate roughly double that of automatic bank transfer or credit card. This likely reflects lower switching friction: customers who pay manually have not committed to recurring payment infrastructure.
- Fibre optic internet customers churn at a higher rate than DSL customers despite paying more, which points to a pricing or service satisfaction issue rather than a usage or engagement problem.
- New customers (0–12 months tenure) churn at the highest rate by lifecycle stage. Churn risk drops sharply after the first year and continues to decline with tenure.
- Customers subscribed to four or more services churn at materially lower rates, consistent with the switching cost hypothesis behind the num_services feature.

**SHAP findings:**

- Tenure is the single most important feature in the XGBoost model, with high tenure pushing strongly against churn across the entire customer base.
- Contract type (month-to-month) is the most important categorical feature. Its SHAP contribution is directionally consistent across the test set: customers on monthly contracts always have their churn probability pushed upward.
- num_services (the engineered stickiness count) ranks in the top five features, validating the hypothesis that product breadth reduces churn propensity. Customers with low num_services values have their churn probability pushed meaningfully upward.
- MonthlyCharges shows a non-linear SHAP pattern: both very low and very high charges are associated with churn, but for different reasons. Very low charges suggest a minimal, easily cancelled subscription; very high charges create price sensitivity and dissatisfaction risk.
- monthly_to_total_ratio ranks higher than raw TotalCharges in feature importance, confirming that the ratio captures information about customer inertia that raw billing totals do not.

**Revenue prioritisation output:**

The top 10% of customers ranked by revenue at risk account for a disproportionate share of total monthly revenue exposed to churn. Prioritising this tier over a pure probability ranking shifts outreach toward higher-value accounts and reduces the cost of false negatives where it matters most financially.

---

## Limitations

- The dataset is a point-in-time snapshot with no temporal dimension. Real churn models use longitudinal data: sequences of monthly behaviour, engagement signals, and service usage trends. The absence of time-series features is the largest gap between this analysis and a production system.
- TotalCharges is partially collinear with tenure and MonthlyCharges. The monthly_to_total_ratio engineered feature reduces but does not eliminate this redundancy.
- The revenue-weighted prioritisation score assumes uniform profit margins across customers and ignores the cost of the retention offer itself. A full expected-value model would subtract the cost of the intervention and account for the probability that the customer would have stayed regardless.
- The dataset contains no information on why customers churned: no customer service interactions, no competitor data, no satisfaction scores. SHAP identifies which features are associated with churn but cannot establish why those associations exist.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/mlamaralp/subscription-churn.git
cd subscription-churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place it in data/
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 4. Open the notebook
jupyter notebook churn_analysis.ipynb
```

---

## References

- IBM (2019). Telco Customer Churn Sample Dataset. IBM Cognos Analytics.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
- Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874.
