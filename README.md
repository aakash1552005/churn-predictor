# Customer Churn Predictor

Predicts whether a telecom customer will leave or stay
using XGBoost — trained on 7,043 real customer records
with SHAP explainability for business stakeholders.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![Accuracy](https://img.shields.io/badge/Accuracy-83.69%25-brightgreen)

## Live Demo
👉 [Click here to try the app](https://churn-predictor-hapbyyp9fex58buros9v6c.streamlit.app/)

---

## Project Overview

Every telecom company loses revenue when customers switch
to competitors. This project predicts which customers are
at risk of churning BEFORE they leave — giving the business
time to intervene with targeted retention offers.

| Model | Accuracy | Recall (Churn) | F1 Score |
|---|---|---|---|
| Baseline (No model) | 73.5% | 0% | 0% |
| **XGBoost + SMOTE** | **83.69%** | **87%** | **0.84** |

> Identified 894 out of 1,033 at-risk customers —
> potentially saving $447,000 in annual revenue.

---

## Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customers x 21 features
- **Target:** Churn (Yes = left, No = stayed)
- **Class imbalance:** 73.5% No Churn vs 26.5% Churn

| Feature | Description |
|---|---|
| tenure | How many months as a customer |
| Contract | Month-to-month, One year, Two year |
| MonthlyCharges | Monthly bill amount |
| TotalCharges | Total amount paid |
| InternetService | DSL, Fiber optic, None |
| TechSupport | Has tech support or not |
| OnlineSecurity | Has online security or not |
| PaymentMethod | How customer pays |

---

## Handling Class Imbalance
```
Problem:
No Churn  73.5% (5,163 customers)
Churn     26.5% (1,869 customers)

A lazy model predicting No Churn always
achieves 73.5% accuracy but catches ZERO churners!

Solution — SMOTE Oversampling:
No Churn  50% (5,163 customers)
Churn     50% (5,163 synthetic + real samples)

Result — model learns both classes equally!
```

---

## Model Performance
```
Confusion Matrix:
                 Predicted
                 No Churn  Churn
Actual No Churn [  835   |  198 ]
       Churn    [  139   |  894 ]

Correctly identified churners: 894 out of 1,033
Missed churners: 139
```

---

## SHAP Explainability

Top features by importance:
```
Contract          52% most important
MonthlyCharges    14% higher bill = higher risk
tenure             9% new customers = higher risk
InternetService    8% Fiber Optic = highest risk
TechSupport        6% no support = higher risk
```

---

## Business Impact
```
At-risk customers identified : 894
Average revenue per customer : $500/year
Revenue protected            : $447,000
Net benefit after discounts  : $441,060 saved
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| Pandas | Data manipulation |
| Scikit-learn | Preprocessing and metrics |
| Imbalanced-learn | SMOTE oversampling |
| XGBoost | Classification model |
| SHAP | Model explainability |
| Joblib | Model serialization |
| Streamlit | Web application |

---

## Project Structure
```
churn-predictor/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── xgb_model.pkl           # Trained XGBoost model (341 KB)
├── feature_names.pkl       # Feature names
├── README.md               # Project documentation
└── .gitignore              # Ignored files
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/aakash1552005/churn-predictor.git
cd churn-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## Retention Suggestions Built In

App automatically suggests actions:
- Month-to-month contract -> offer annual discount
- Monthly charges above $70 -> offer loyalty discount
- Tenure less than 12 months -> assign dedicated support
- Fiber Optic service -> check service quality

---

## Future Improvements

- [ ] Add SHAP waterfall plot per prediction
- [ ] Add customer segment dashboard
- [ ] Build REST API using FastAPI
- [ ] Add email alerts for high risk customers

---

## Author

**Aakash S S**
- 📧 aakash1552005@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/aakash-s-s-88b91a301)
- 🐙 [GitHub](https://github.com/aakash1552005)

---

## My AI/ML Portfolio

- Project 1 ✅ [House Price Predictor](https://house-price-predictor-mpgei6k8kext3prxmwealo.streamlit.app/)
- Project 2 ✅ [Customer Churn Predictor](https://churn-predictor-hapbyyp9fex58buros9v6c.streamlit.app/)
- Project 3 ⏳ Analytics Dashboard
- Project 4 ⏳ ML Pipeline
- Project 5 ⏳ Autonomous Analyst Agent
- Project 6 ⏳ End-to-End AI Product

---

## If you found this useful please star the repo!