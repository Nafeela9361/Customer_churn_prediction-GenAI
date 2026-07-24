# Customer Churn Prediction — Generative AI for Business Applications

A machine learning project that predicts which bank customers are likely to churn (attrite), enabling proactive retention strategies.

## 📌 Overview

Customer retention is critical for bank profitability, since acquiring a new customer typically costs more than retaining an existing one. This notebook builds and compares several classification models to identify customers at risk of churning, so the bank can intervene early with targeted offers and improved service.

## 💼 Business Context

- Customer retention drives bank growth and profit; customers can easily switch banks given many available choices.
- Losing customers directly reduces income, and replacing them is more expensive than keeping them.
- The goal is to flag at-risk customers early so the bank can act before they leave.

## 📊 Dataset

- **Source file:** `Churn Prediction.csv` (loaded from Google Drive in Colab)
- **Size:** 10,127 rows × 21 columns, no missing values, no duplicate records
- **Target variable:** `Attrition_Flag` — `Existing Customer` vs. `Attrited Customer`
- **Feature groups:**
  - **Identifier:** `CLIENTNUM` (dropped before modeling)
  - **Demographics:** `Customer_Age`, `Gender`, `Dependent_count`, `Education_Level`, `Marital_Status`
  - **Account/behavioral:** `Income_Category`, `Card_Category`, `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio`

## 🧭 Notebook Workflow

1. **Setup & Data Loading** — Import pandas, numpy, matplotlib, seaborn, and scikit-learn; mount Google Drive and load the CSV.
2. **Data Overview** — Shape, dtypes, summary statistics, and duplicate check.
3. **Exploratory Data Analysis (EDA)**
   - *Univariate:* bar charts for categorical features, box plots for numerical features (outliers found in `Total_Ct_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Amt_Chng_Q4_Q1`, `Avg_Open_To_Buy`, `Credit_Limit`).
   - *Bivariate:* churn rate broken down by age, gender, income level, education level, transaction amount, credit limit, card category, and months inactive.
4. **Data Preprocessing**
   - Drop `CLIENTNUM`.
   - Binary-encode the target (`Existing Customer` → 0, `Attrited Customer` → 1).
   - Ordinal-encode `Education_Level` and `Income_Category` by hierarchy.
   - One-hot (n-1 dummy) encode `Marital_Status` and `Card_Category`.
   - Label-encode `Gender`.
   - Stratified 70/30 train-test split.
5. **Model Building** — Trains and evaluates three classifiers on Accuracy, Precision, Recall, and F1 (train vs. test):
   - **Decision Tree** — perfect on train, drops on test → overfits.
   - **Random Forest** — strong test performance, still overfits somewhat.
   - **Gradient Boosting** — best balance between train and test performance; selected as the final model.
6. **Final Model Selection & Feature Importance** — Gradient Boosting is chosen for its generalization. Top drivers of churn: `Total_Trans_Ct`, `Total_Trans_Amt`, and `Total_Revolving_Bal`.
7. **Actionable Insights** — Business recommendations targeting low transaction count/amount customers (engagement campaigns, spending incentives, cross-selling).
8. **Appendix: Hyperparameter Tuning** — `GridSearchCV` over `n_estimators`, `learning_rate`, `max_depth`, and `subsample`, optimized for recall, to further reduce missed churners (false negatives).
9. **Prediction on Unseen Data** — Demonstrates inference with the tuned model on held-out test samples.
10. **Model Comparison Visualization & Conclusion** — Bar chart comparing F1-scores across models; summary of results and future development ideas (feature engineering, class-imbalance handling, real-time API deployment).

## 🎯 Evaluation Criterion

Because missing an actual churner (a false negative) is costly, the primary objective is to **maximize recall** while keeping precision and F1 reasonable — catching as many true churn cases as possible.

## 🏆 Results Summary

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Notes |
|---|---|---|---|---|---|
| Decision Tree | 0.93 | 0.78 | 0.81 | — | Overfits (perfect train scores) |
| Random Forest | 0.96 | 0.92 | 0.80 | 0.85 | Better generalization, mild overfitting |
| Gradient Boosting | 0.96 | 0.93 | 0.81 | 0.86 | Best train/test balance; selected model |
| Gradient Boosting (Tuned) | — | — | Improved | — | Recall-optimized via GridSearchCV |

## 🛠️ Requirements

- Python 3
- Google Colab (uses `google.colab.drive` to mount Drive) — or adapt the data-loading cell to read the CSV locally
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## ▶️ How to Run

1. Place `Churn Prediction.csv` in the expected path (update the `pd.read_csv(...)` path if not using Colab/Drive).
2. Run all cells top to bottom — EDA cells, preprocessing, model training, tuning, and evaluation are executed sequentially.
3. Review the model comparison table/chart and feature-importance plot for final results.

## 💡 Key Business Recommendations

- **Boost engagement** for customers with low transaction counts (rewards for activity, loyalty education).
- **Encourage higher spending** for low-transaction-amount customers (personalized promotions, cross-selling).
- **Monitor revolving balance and inactivity** trends, since these strongly correlate with churn risk; prioritize outreach around 3 months of inactivity, where attrition risk peaks.

## 🚀 Future Development

- Further hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Additional feature engineering and feature selection
- Handling class imbalance (e.g., SMOTE, class weighting)
- Deploying the trained model as a real-time REST API (Flask/FastAPI)
