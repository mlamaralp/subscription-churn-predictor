# Subscription Churn Prediction : Revenue-Weighted Retention Prioritisation

Predicting customer churn and quantifying the expected revenue at risk for each account, using the IBM Telco Customer Churn dataset (7,043 customers, 20 features). The goal is not just to classify who will leave, but to rank customers by the financial cost of losing them : turning a binary classifier into a prioritisation tool for retention teams.

---

## Why This Problem

Every major technology company that operates a subscription product runs some version of this analysis. Microsoft 365, Xbox Game Pass, and Azure all depend on recurring revenue that evaporates the moment a customer cancels. Google One, YouTube Premium, and Google Workspace face the same problem. Churn prediction is not an academic exercise at these companies: it is a live system that feeds retention campaigns, pricing decisions, and product roadmaps.

The standard approach : build a classifier, report accuracy, stop : does not answer the question a product or revenue team actually needs to act on: which customers should we contact first, and what is the financial cost of getting it wrong? This project adds that layer. Each customer's churn probability is multiplied by their monthly charges to produce an expected monthly revenue at risk score, which re-ranks the output of the model into a prioritised intervention list. A customer with a 60% churn probability and $100/month in charges is worth more to retain than one with an 80% probability and $20/month.

---

## Research Questions

1. **Which customer and contract characteristics are the strongest predictors of churn?** Across demographics, service subscriptions, billing type, and contract length, which features drive the classification most and in which direction?
2. **How does a gradient-boosted model compare to a logistic regression baseline on an imbalanced binary target?** With a 26.5% churn rate, does a tuned XGBoost model add measurable value over a simple linear baseline, and at what cost in false positives?
3. **Can churn probability be combined with revenue data to produce an actionable prioritisation score?** Does re-ranking customers by expected revenue at risk change which accounts a retention team would contact first?

---

## Dataset

**Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) on Kaggle (IBM sample dataset).

The dataset contains 7,043 customer records across 20 features covering demographics, subscribed services, contract and billing details, and a binary churn label. It was chosen because its feature structure : contract type, payment method, tenure, monthly charges : maps directly onto the subscription dynamics of any recurring-revenue product. The known data quality issue (11 blank values in TotalCharges for customers with zero tenure) is documented and handled explicitly in preprocessing.

| Feature group | Features |
| ------------- | -------- |
| Demographics  | Gender, SeniorCitizen, Partner, Dependents |
| Services      | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Account       | tenure, Contract, PaperlessBilling, PaymentMethod |
| Billing       | MonthlyCharges, TotalCharges |
| Target        | Churn (Yes / No) |

Overall churn rate: 26.5% (1,869 out of 7,043 customers). Moderately imbalanced : addressed through class weighting rather than resampling.

---

## Repository Structure

```
subscription-churn-predictor/
├── churn_analysis.py        # Full pipeline: EDA, feature engineering, modelling, revenue scoring, SHAP
├── data/                    # Not tracked: place the Kaggle CSV here
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── images/                  # Generated automatically on first run
├── requirements.txt
└── README.md
```

---

## Tools

| Area              | Stack                 |
| ----------------- | --------------------- |
| Language          | Python 3.10+          |
| Data manipulation | Pandas, NumPy         |
| Visualisation     | Matplotlib, Seaborn   |
| Modelling         | Scikit-learn, XGBoost |
| Explainability    | SHAP                  |

---

## Methodology

### Data Cleaning

TotalCharges contains 11 blank string values corresponding to customers with zero tenure (new accounts with no billing history). These were converted to 0 rather than dropped or imputed with the column mean, since zero is the correct value: a customer who has never completed a billing cycle has zero total charges. Dropping these rows would silently remove new customers from the dataset and bias the tenure distribution.

customerID was dropped before modelling as a row identifier with no predictive signal. SeniorCitizen was already encoded as a binary integer (0/1) and required no transformation. Binary Yes/No columns were label-encoded. InternetService, Contract, and PaymentMethod : the three multi-class categorical features : were one-hot encoded. Class imbalance (26.5% churn) was addressed through `class_weight='balanced'` in Logistic Regression and `scale_pos_weight` in XGBoost, rather than SMOTE or other resampling. With a 26.5% minority class, synthetic oversampling adds less value than class weighting and risks generating implausible combinations of contract type, tenure, and billing values.

The data was split 80/20 into training (5,634 customers) and test (1,409 customers) sets, stratified on the churn label (`random_state=42`). All metrics below are computed on the test set.

### Feature Engineering

**tenure_group:** Tenure in months (range: 0–72) was binned into five groups: New (0–12), Developing (13–24), Established (25–36), Loyal (37–48), and Long-term (49–72). It is used in the EDA to show churn by customer lifecycle stage; the models use the raw numeric tenure.

**num_services:** A count of how many of the seven optional services each customer has subscribed to (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, MultipleLines). The hypothesis is that customers embedded in more services have a higher switching cost and lower churn propensity : a pattern common in bundled subscription products.

**monthly_to_total_ratio:** MonthlyCharges divided by TotalCharges, representing the share of the customer's total spend that their most recent bill represents. For long-tenure customers this ratio approaches zero; for new customers it approaches one (customers with zero total charges are set to 1.0). It acts as a proxy for how little billing history a customer has built up.

### Model 1: Logistic Regression (Baseline)

Logistic Regression provides a transparent, interpretable baseline. Its coefficients directly represent log-odds, making it possible to verify that the model has learned directionally sensible relationships before moving to a more complex approach. Features are standardised. Parameters: `max_iter=1000`, `class_weight='balanced'`, `random_state=42`.

### Model 2: XGBoost with Grid Search

XGBoost was chosen as the second model because gradient-boosted trees are a common standard for tabular churn prediction, handle moderate imbalance through `scale_pos_weight`, and work natively with SHAP's TreeExplainer.

A grid search was run over learning_rate (0.01, 0.1, 0.3), max_depth (3, 5, 7), n_estimators (100, 300), and subsample (0.8, 1.0) : 36 combinations, each scored using 5-fold stratified cross-validation on AUC-ROC. The best configuration was `learning_rate=0.01`, `max_depth=5`, `n_estimators=300`, `subsample=0.8`, with a cross-validated AUC-ROC of 0.848.

**Why AUC-ROC and not accuracy or F1?** With 26.5% churn, accuracy is dominated by the majority class. F1 requires choosing a threshold before optimisation. AUC-ROC makes no threshold assumption and directly evaluates the quality of the probability scores the model outputs : which is what matters when the goal is to rank customers, not just classify them.

### Revenue-Weighted Prioritisation Score

After the final model produces a churn probability for each customer, a revenue at risk score is computed as:

```
revenue_at_risk = churn_probability × MonthlyCharges
```

This score re-ranks the customer list in a way that reflects business impact rather than model confidence. It answers the question a retention team actually faces: given a fixed number of outreach slots, which customers should we contact first? A customer with a 55% churn probability paying $95/month has a revenue at risk score of $52.25. A customer with a 90% churn probability paying $20/month has a score of $18.00. Pure churn probability would prioritise the second customer; revenue at risk correctly prioritises the first.

The top 10% of test-set customers ranked by revenue at risk are identified as the primary intervention tier.

---

## Key Findings

### Model comparison (test set, 1,409 customers)

| Metric                  | Logistic Regression | XGBoost (tuned) |
| ----------------------- | ------------------- | --------------- |
| Accuracy                | 0.737               | 0.763           |
| Precision (churn class) | 0.503               | 0.536           |
| Recall (churn class)    | 0.802               | 0.799           |
| F1 (churn class)        | 0.618               | 0.642           |
| AUC-ROC                 | 0.848               | 0.847           |

Both models rank customers equally well (AUC-ROC ≈ 0.85). XGBoost is slightly better on accuracy, precision and F1 at the same recall, and was kept as the final model for the revenue-at-risk step and for SHAP explainability. The result suggests that, on this dataset, most of the predictive signal sits in a few strong effects (contract type, tenure, services, billing) that a linear model already captures; tuned gradient boosting adds little ranking power. Class weighting favours recall, which is why precision and accuracy are modest for both models at the default 0.5 threshold.

### EDA findings

- Month-to-month contracts have a churn rate of 42.7%, compared to 11.3% for one-year and 2.8% for two-year contracts. Contract type is the single strongest predictor in the dataset.
- Electronic check payment is associated with a churn rate of 45.3%, roughly three times that of automatic bank transfer (16.7%) or credit card (15.2%). This likely reflects lower switching friction: customers who pay manually have not committed to recurring payment infrastructure.
- Fibre optic internet customers churn at 41.9%, more than double DSL customers (19.0%), despite paying more, which points to a pricing or service satisfaction issue rather than a usage problem.
- New customers (0–12 months tenure) churn at 47.4%. Churn falls steadily with tenure, to 9.5% for customers with 49+ months.
- Churn is highest for customers with one or two add-on services (34.5% and 37.7%) and falls from three services onwards, to 11.7% at six services and 5.3% at all seven, consistent with the switching-cost hypothesis behind `num_services`.

### SHAP findings

Top features by mean absolute SHAP value (XGBoost, test set):

| Rank | Feature                        | Mean \|SHAP\| |
| ---- | ------------------------------ | ------------- |
| 1    | Contract_Month-to-month        | 0.786         |
| 2    | monthly_to_total_ratio         | 0.271         |
| 3    | MonthlyCharges                 | 0.214         |
| 4    | OnlineSecurity                 | 0.212         |
| 5    | InternetService_Fiber optic    | 0.188         |
| 6    | PaymentMethod_Electronic check | 0.139         |
| 7    | TechSupport                    | 0.130         |
| 8    | Contract_One year              | 0.102         |
| 9    | tenure                         | 0.097         |
| 10   | PaperlessBilling               | 0.089         |

- Month-to-month contract is by far the most important feature, almost three times the next one.
- monthly_to_total_ratio, the engineered tenure-related feature, ranks second, above raw TotalCharges (which does not appear in the top ten), confirming that the ratio captures information about customer inertia that raw billing totals do not.
- Service and pricing features (OnlineSecurity, TechSupport, MonthlyCharges, fibre optic, electronic check) fill most of the remaining top ten. Raw tenure ranks ninth, as much of its information is carried by the ratio and the contract type.

### Revenue prioritisation output

On the test set (1,409 customers), total monthly revenue at risk is $41,204. The top 10% of customers by revenue at risk (140 customers) hold **$10,795, or 26.2% of the total**. 60.7% of customers in this tier actually churned, compared with 26.5% overall, and the revenue actually lost in the tier was $7,823 per month. Prioritising this tier over a pure probability ranking shifts outreach toward higher-value accounts and reduces the cost of false negatives where it matters most financially.

---

## Limitations

- XGBoost does not outperform the logistic regression baseline in AUC-ROC (0.847 vs 0.848). The tuned model is kept for explainability and slightly higher precision, not for better ranking.
- Classification metrics use the default 0.5 threshold; a different threshold would trade precision against recall. Revenue at risk is computed on the 20% test set only, so the 26.2% figure carries sampling noise.
- The dataset is a point-in-time snapshot with no temporal dimension. Real churn models use longitudinal data: sequences of monthly behaviour, engagement signals, and service usage trends. The absence of time-series features is the largest gap between this analysis and a production system.
- TotalCharges is partially collinear with tenure and MonthlyCharges. The monthly_to_total_ratio engineered feature reduces but does not eliminate this redundancy.
- The revenue-weighted prioritisation score assumes uniform profit margins across customers and ignores the cost of the retention offer itself. A full expected-value model would subtract the cost of the intervention and account for the probability that the customer would have stayed regardless.
- The dataset contains no information on why customers churned: no customer service interactions, no competitor data, no satisfaction scores. SHAP identifies which features are associated with churn but cannot establish why those associations exist.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/mlamaralp/subscription-churn-predictor.git
cd subscription-churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place it in data/
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 4. Run the full pipeline (figures are saved to images/)
python churn_analysis.py
```

---

## References

- IBM (2019). Telco Customer Churn Sample Dataset. IBM Cognos Analytics.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
- Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874.
