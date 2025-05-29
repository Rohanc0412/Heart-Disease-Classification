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

**_Model used_:** Logistic Regression, Random Forest, KNN  <br><br>
<img src="https://github.com/Rohanc0412/Heart-Disease-Classification/blob/56acb58d5c204474888a386687819b32e2090a52/assets/Screenshot%202025-05-29%20172452.png" alt="Heart Disease Prediction" width="800"/>


**_Visualizations Performed_**:
- Heatmap of feature correlation :<br><br>
  <img src="https://github.com/Rohanc0412/Heart-Disease-Classification/blob/2e97a711260dad51081e17edaaeff9c61fdb6203/assets/Screenshot%202025-05-29%20173003.png" alt="Heart Disease Prediction" width="500"/>

- Feature importance plot of Random forest Classifier: <br><br>
  <img src="https://github.com/Rohanc0412/Heart-Disease-Classification/blob/e514062422db15822fb87d94dd77c35a3c3632c8/assets/Screenshot%202025-05-29%20174433.png" alt="Heart Disease Prediction" width="500"/>
  
- ROC curve of Random forest classifier:<br><br>
  <img src="https://github.com/Rohanc0412/Heart-Disease-Classification/blob/e514062422db15822fb87d94dd77c35a3c3632c8/assets/Screenshot%202025-05-29%20174405.png" alt="Heart Disease Prediction" width="500"/>
  
- Confusion matrix of Random forest classifier:<br><br>
  <img src="https://github.com/Rohanc0412/Heart-Disease-Classification/blob/e514062422db15822fb87d94dd77c35a3c3632c8/assets/Screenshot%202025-05-29%20174421.png" alt="Heart Disease Prediction" width="500"/>

> _Other visualizations were also performed during EDA_

---

## Dataset Description

- **Source**: [Heart Disease Classification Dataset](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
- **Records**: ~300
- **Features**: `age`, `trestbps`, `chol`, `thalach`, `oldpeak``sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`
- **Target**: `target` → 1 = Disease Present, 0 = No Disease

---

## Technologies Used

- **Language**: Python
- **IDE**: Jupyter Notebook
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Modeling: `LogisticRegression`, `KNeighborsClassifier`, `RandomForestClassifier`
  - Hyper-parameter Tuning: `RandomizedSearchCV`,  `GridSearchCV`
  - Evaluation: `classification_report`, `confusion_matrix`, `roc_auc_score`, `f1_score`, `recall_score`, `precision_score`, `RocCurveDisplay` 

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
2. **Launch the notebook**
```bash
jupyter notebook Prediction_Of_Heart_Disease.ipynb
```

---

## Future Work
- Integrate real-time patient data from EHR systems

- Deploy via Flask/FastAPI as an API

- Use SHAP for model explainability

- Try deep learning-based classification for improved generalization
