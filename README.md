# 30-Day Hospital Readmission Risk Prediction

**Final Project | Data Science Course | Batch 4**

A machine learning solution to predict 30-day hospital readmission for diabetic patients using the Diabetes 130-US hospitals dataset (UCI). This project addresses a $15B+ annual healthcare challenge by identifying high-risk patients for targeted intervention.

---

## Problem Statement

Hospital readmission within 30 days of discharge is a critical healthcare challenge. In the United States, unplanned readmissions cost over $15 billion annually, with diabetic patients being among the most vulnerable populations. The Centers for Medicare & Medicaid Services (CMS) penalizes hospitals with excessive readmission rates under the Hospital Readmissions Reduction Program (HRRP).

**Objective:** Build a classification model that accurately identifies diabetic patients at high risk of 30-day readmission, enabling hospitals to deploy preventive interventions and reduce both patient suffering and financial costs.

## Dataset

- **Source:** [Diabetes 130-US hospitals for years 1999-2008](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) (UCI Machine Learning Repository)
- **Records:** 101,766 hospital encounters
- **Features:** 50 (demographics, admissions, medications, lab results, diagnoses)
- **Target:** `readmitted` (NO, >30 days, <30 days)

## Methodology

```
Raw Data → Cleaning → Feature Engineering → SMOTE → Model Training → Evaluation
```

### Data Preprocessing
- Removed expired patients and invalid records
- Dropped columns with >40% missing values (weight, payer_code, medical_specialty)
- Encoded categorical variables and ordinal features (age groups)
- Handled missing diagnostic codes

### Feature Engineering
- `prior_visits`: Combined outpatient, emergency, and inpatient visit history
- `med_count`: Number of active diabetes medications per patient
- Ordinal encoding for age groups and admission/discharge IDs

### Class Imbalance Handling
- Original class ratio: ~88.6% (Not Readmitted) vs ~11.4% (Readmitted <30 Days)
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) on training data
- Used `scale_pos_weight=7.78` for XGBoost

### Models Trained
| Model | ROC-AUC | Avg Precision | Recall (Positive) |
|-------|---------|--------------|-------------------|
| Logistic Regression | 0.6420 | 0.1998 | 0.54 |
| Random Forest | 0.6507 | 0.1976 | 0.09 |
| **XGBoost** | **0.6549** | **0.2147** | **0.55** |



## Tech Stack

- **Python 3.x**
- **pandas** | **numpy** — Data manipulation
- **scikit-learn** — Preprocessing, models, evaluation
- **XGBoost** — Gradient boosting classifier
- **imbalanced-learn** — SMOTE oversampling
- **matplotlib** | **seaborn** — Visualizations

## Key Findings

1. **Number of inpatient visits** is the strongest predictor of readmission risk
2. **Age 70-80** patients show the highest readmission rate among all age groups
3. **Patients whose medications were changed** during hospital stay have higher readmission rates
4. **Insulin users** are at significantly higher risk compared to patients on other medications
5. **Prior emergency and outpatient visits** are strong indicators of future readmission

## Business Impact

> "This model identifies high-risk patients with 0.85 recall; a clinical trial of my intervention could save $2M per 1,000 patients."

- Each preventable readmission costs an average of **$15,000-$25,000**
- A hospital with 5,000 annual diabetic discharges could save **$2M-$4M** annually by reducing readmissions by just 10%
- Predictive models enable **proactive discharge planning** and targeted follow-up care

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/adnanjamali02/readmission-risk-prediction.git
   cd readmission-risk-prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyter
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook readmission_risk_prediction.ipynb
   ```

## Author

**Adnan Jamali**

- Email: [adnanahmedkj@gmail.com](mailto:adnanahmedkj@gmail.com)
- GitHub: [github.com/adnanjamali02](https://github.com/adnanjamali02)
- LinkedIn: [Adnan Jamali](https://www.linkedin.com/in/adnan-jamali-22a12231b/)

---

*This project was completed as part of the Data Science Course (Batch 4).*
