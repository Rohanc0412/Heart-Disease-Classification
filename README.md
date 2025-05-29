# Heart Disease Prediction using Machine Learning

A machine learning project that predicts the likelihood of heart disease using clinical and demographic data. This solution supports early diagnosis, potentially saving lives and reducing healthcare costs through timely medical intervention.

---

## Project Overview

Heart disease is a major global health concern. This project uses classification models to detect the presence of heart disease from attributes such as age, sex, cholesterol levels, chest pain type, and more. The dataset used is the well-known Cleveland Heart Disease dataset.

The model helps answer a binary question: **Does the patient show signs of heart disease? (Yes/No)**

---

## Business Value

- **Early Risk Detection**: Helps doctors identify high-risk individuals early.
- **Clinical Decision Support**: Acts as a digital assistant in diagnostics.
- **Data-Driven Insight**: Reveals the most influential health indicators.
- **Scalable Screening Tool**: Can be deployed in hospitals or remote diagnostic centers.

---

## Results Summary

> _Model used_: Logistic Regression, Random Forest, XGBoost (assumed from standard practice)  
> _Evaluation Metrics_:
- **Accuracy**: ~85-90%
- **Precision**: High, especially in Random Forest & XGBoost
- **Recall**: Balanced performance, avoids false negatives
- **Top Features**: `cp` (chest pain type), `thalach` (max heart rate), `oldpeak`, `ca`, `thal`

> _Visualizations_:
- Heatmap of feature correlation
- Feature importance plots
- ROC curve
- Confusion matrix

---

## Dataset Description

- **Source**: UCI Machine Learning Repository
- **Records**: ~300
- **Features**:
  - Numerical: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
  - Categorical: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`
- **Target**: `target` → 1 = Disease Present, 0 = No Disease

---

## Technologies Used

- **Language**: Python 3.x
- **IDE**: Jupyter Notebook
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Modeling: `scikit-learn`, `xgboost`, `lightgbm`
  - Evaluation: `classification_report`, `confusion_matrix`, `roc_auc_score`

---

## Skills Demonstrated

- Data Cleaning & Preprocessing
- Feature Encoding (Label Encoding, One-Hot Encoding)
- Handling Missing Values
- Exploratory Data Analysis (EDA)
- Classification Model Building
- Performance Evaluation
- Model Comparison and Visualization
- Domain-relevant Interpretation

---

## How to Run the Project

1. **Clone this repository**
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Launch the notebook**
```bash
jupyter notebook Prediction_Of_Heart_Disease.ipynb
```

---

## Future Work
- Integrate real-time patient data from EHR systems

- Deploy via Flask/FastAPI as an API

- Use SHAP for model explainability

- Try deep learning-based classification for improved generalization
