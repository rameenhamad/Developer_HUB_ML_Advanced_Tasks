## Task 1: End-to-End ML Pipeline with Scikit-learn
### Objective: 
Build a reusable and production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.
### Dataset:
Dataset Name: Telco Customer Churn
Includes customer demographic, service usage, and account information used to predict churn behavior.
### Data Preprocessing:
- Removed missing and duplicate entries.
- Converted categorical columns into numeric form using One-Hot Encoding.
- Scaled numerical columns with StandardScaler.
- Combined preprocessing steps using ColumnTransformer within a Scikit-learn Pipeline.
### Model Development:
Implemented two classifiers:
  - Logistic Regression
  - Random Forest Classifier
Integrated models into pipelines for end-to-end processing (preprocessing → training → evaluation).
### Hyperparameter Tuning:
Used **GridSearchCV** for systematic hyperparameter optimization.
Explored **penalties**, **solvers**, and **regularization** strengths for Logistic Regression.
Tuned **tree depth**, **estimators**, and feature selection strategies for Random Forest.
### Model Evaluation:
- Logistic Regression Accuracy: **~81.9 %**
- Random Forest Accuracy: **~73.5 %**
**Logistic Regression** showed **better** generalization on unseen data.
### Model Export:
Saved optimized models using **joblib.dump()** for deployment:

logistic_regression_churn.pkl

random_forest_churn.pkl
