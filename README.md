# 🏦 Kaggle Playground Series 2025 — Loan Repayment Prediction (Version 3)

**Competition Link:**  
[Loan Repayment Prediction — Kaggle Playground Series 2025](https://www.kaggle.com/competitions/playground-series-s3e25)

---

## 🎯 Overview

The **goal** of this competition is to predict the **probability that a borrower will repay their loan**  
(`loan_paid_back = 1`).

Each row in the dataset represents a borrower with financial and credit attributes.  
The task is a **binary classification problem**, and the performance is measured using **ROC-AUC**.

---

## 🧾 Evaluation Metric

Submissions are evaluated on the **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**  
between the predicted probabilities and the true `loan_paid_back` values.

Higher AUC → Better model discrimination between repayers and defaulters.

---

## 🧠 Approach Summary

This repository implements a **Hybrid Stacking Pipeline** using **LightGBM** and **CatBoost** with **Optuna Hyperparameter Optimization** and **SHAP Explainability**.

### 🔹 Key Steps

1. **Data Preprocessing**
   - Missing values imputed using mean (`SimpleImputer`).
   - Standard scaling applied (`StandardScaler`).
   - Only numeric features used for stability and simplicity.

2. **Feature Engineering**
   Additional derived features created to capture borrower behavior and risk:
   - `debt_ratio = loan_amount / annual_income`
   - `income_interest = annual_income / interest_rate`
   - `credit_interest = credit_score * interest_rate`
   - `log_income = log1p(annual_income)`
   - `loan_to_credit = loan_amount / credit_score`
   - `credit_sq = credit_score ** 2`

3. **Model Training**
   - **Optuna** automatically tunes LightGBM’s hyperparameters via 3-fold CV.
   - **LightGBM** and **CatBoost** are trained using **5-fold Stratified OOF Stacking**:
     - Each fold produces out-of-fold (OOF) predictions.
     - Averaged test predictions ensure generalization.
   - **Final Meta LightGBM Model** trained on fused features:
     ```
     [original features + oof_pred_lgb + oof_pred_cat]
     ```

4. **Explainability (SHAP)**
   - Uses `shap.TreeExplainer` for interpretability.
   - SHAP summary and dependence plots reveal the most impactful financial indicators.

5. **Submission**
   - Produces final `submission.csv` file with:
     ```csv
     id,loan_paid_back
     1001,0.8234
     1002,0.1321
     ...
     ```
   - File ready for Kaggle submission.

---

## ⚙️ Model Configurations

### **LightGBM (Optuna-tuned)**
| Parameter | Range / Value |
|------------|---------------|
| `learning_rate` | 0.005–0.1 (log search) |
| `num_leaves` | 16–128 |
| `max_depth` | 3–12 |
| `feature_fraction` | 0.5–1.0 |
| `bagging_fraction` | 0.5–1.0 |
| `lambda_l1/l2` | 0.0–2.0 |
| `n_estimators` | 2000 |
| `objective` | binary |
| `metric` | auc |

### **CatBoost (Hand-tuned)**
| Parameter | Value |
|------------|--------|
| `iterations` | 3000 |
| `learning_rate` | 0.02 |
| `depth` | 7 |
| `l2_leaf_reg` | 3 |
| `border_count` | 128 |
| `random_seed` | 42 |

### **Final LightGBM Meta-Model**
| Parameter | Value |
|------------|--------|
| `n_estimators` | 2500 |
| `learning_rate` | 0.015 |
| `num_leaves` | 80 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `reg_alpha/lambda` | 0.1 / 0.1 |
| `objective` | binary |
| `metric` | auc |

---

## 📈 Results

| Stage | Description | AUC (ROC) |
|--------|--------------|------------|
| Baseline (numeric only) | Simple LGBM | 0.689 |
| + Feature Engineering | Ratios, logs, interactions | 0.735 |
| + Optuna Hyper-Tuning | Stronger LightGBM | 0.78 |
| + Hybrid Fusion (LGB + CatBoost) | Final Stacked Model | **0.8113** ✅ |

### 🏆 Kaggle Public Leaderboard
- **Version:** V3  
- **Public Score (AUC):** **0.81131**  
- **Ranking:** **2037 / 2461 participants**

---

## 🧩 SHAP Explainability Insights

| Feature | SHAP Impact | Description |
|----------|--------------|--------------|
| `credit_score` | ↑ Higher → more likely to repay | Strongest predictor |
| `debt_ratio` | ↑ Higher → less likely to repay | Indicates credit stress |
| `interest_rate` | ↑ Higher → higher default risk | Cost of borrowing |
| `annual_income` | ↑ Higher → more likely to repay | Financial capacity |
| `income_interest` | Non-linear effect | Interaction between income and interest |

---

## 🧪 How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn lightgbm catboost shap matplotlib optuna

