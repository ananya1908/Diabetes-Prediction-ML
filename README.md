# Diabetes Prediction using Machine Learning

This project explores the **Pima Indians Diabetes Dataset** to understand key medical indicators associated with diabetes and build machine learning models that predict diabetes risk.

The analysis includes **data exploration, feature analysis and multiple machine learning models**, ultimately identifying the most effective model for predicting diabetes outcomes.

---

# Project Overview

Diabetes is a chronic disease that affects millions worldwide. Early identification of high-risk individuals can help guide preventive interventions and medical monitoring.

In this project, we analyze clinical health indicators such as:

- Glucose level  
- BMI  
- Insulin level  
- Age  
- Skin thickness  
- Blood pressure  
- Number of pregnancies  

Using these variables, we build predictive models that estimate whether a patient is likely to have diabetes.

---

# Dataset

The dataset used is the **Pima Indians Diabetes Database** from the National Institute of Diabetes and Digestive and Kidney Diseases.

Each row represents a female patient of Pima Indian heritage aged 21 or older.

### Features

| Feature | Description |
|------|------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic diabetes likelihood |
| Age | Patient age |
| Outcome | Diabetes diagnosis (0 = No, 1 = Yes) |

Dataset source:  
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

---

# Project Workflow

The project follows a standard machine learning workflow:

## 1. Data Cleaning

- Identified unrealistic values (e.g., 0 for BMI, glucose, insulin)
- Replaced invalid values with `NaN`
- Imputed missing values using **median values grouped by outcome**

## 2. Exploratory Data Analysis (EDA)

Several visualizations were created to understand patterns in the data:

- Feature distributions
- Correlation heatmap
- Pairplots of key variables
- Diabetes prevalence across **age groups**
- Diabetes prevalence across **BMI categories**
- Distribution comparisons for glucose, BMI, and insulin

### Key observations

- **Glucose is the strongest predictor of diabetes**
- Diabetes prevalence increases with **age**
- Higher **BMI categories show significantly higher diabetes rates**

---

# Feature Engineering

Additional features were derived to better understand risk patterns:

- Age groups
- BMI categories

These transformations helped reveal meaningful relationships between patient characteristics and diabetes risk.

---

# Machine Learning Models

Multiple models were trained and evaluated:

| Model | Purpose |
|------|------|
| Logistic Regression | Baseline model |
| Random Forest | Capture nonlinear feature relationships |
| Random Forest + SMOTE | Handle class imbalance |
| XGBoost | Gradient boosting for improved performance |

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

All models were evaluated using **10-fold Stratified Cross Validation**.

---

# Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|------|------|------|------|------|
| Logistic Regression | 0.77 | 0.72 | 0.59 | 0.64 | 0.86 |
| Random Forest | 0.88 | 0.86 | 0.80 | 0.82 | 0.94 |
| Random Forest + SMOTE | 0.88 | 0.82 | **0.85** | **0.83** | 0.94 |
| XGBoost | **0.88** | 0.82 | 0.83 | **0.83** | **0.95** |

The **XGBoost model achieved the best overall performance**, with the highest ROC-AUC and strong balance between precision and recall.

---

# Final Model Results (Test Set)

```
Accuracy: 0.88
Precision: 0.82
Recall: 0.83
F1 Score: 0.83
ROC-AUC: 0.95
Average Precision: 0.92
```

The results show that the model can effectively distinguish between diabetic and non-diabetic patients.

---

# Key Insights

- **Glucose levels are the strongest predictor of diabetes**
- **BMI and insulin levels are strongly associated with diabetes risk**
- Diabetes prevalence increases with **age**
- Ensemble models significantly outperform linear models for this dataset

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

---

# Repository Structure

```
├── diabetes.csv
├── Diabetes_Notebook.ipynb
└── README.md
```

---

# How to Run

Clone the repository

```bash
git clone https://github.com/yourusername/diabetes-prediction
```

Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

Run the notebook

```bash
jupyter notebook Diabetes_Notebook.ipynb
```

---

# Author

**Ananya Sharma**  
Master’s in Information Management – Data Science  
University of Washington
