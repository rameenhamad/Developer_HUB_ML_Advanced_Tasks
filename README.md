# Task 1: End-to-End ML Pipeline with Scikit-learn
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
- logistic_regression_churn.pkl
- random_forest_churn.pkl

# Task 2: Multimodal ML – Housing Price Prediction (Images + Tabular Data)
### Objective:
Develop a multimodal machine learning model to predict house prices by **combining image-based and tabular** (structured) data features for improved predictive accuracy.
### Dataset:
**Structured Data:** Housing Sales Dataset containing attributes such as bedrooms, bathrooms, square footage, and city codes.
**Image Data:** Corresponding house images from a custom dataset (socal_pics/).
Each sample links tabular attributes with its image through a unique image_id.
### Data Preparation:
- Added image_path column to map each record to its corresponding image file.
- Split data into training (80%) and validation (20%) sets.
- Selected key features: ["sqft", "bed", "n_citi", "bath"].
- Applied ImageDataGenerator for image normalization and batching.
### Feature Fusion
Defined a **custom data generator** to synchronize image and tabular inputs.
Implemented **dual-input** architecture:
CNN branch for image feature extraction.
Dense network for tabular features.
Combined both feature streams using **Concatenate()** layer before regression output.
### Model Architecture
**Image branch:**
- Two **convolutional blocks** with **ReLU activation** and **max pooling**.
- **Flattened** outputs for feature extraction.
**Tabular branch:**
**Dense** layers with **Batch Normalization** and **Dropout** for regularization.
**Fusion:**
**Combined** dense layers → Batch Normalization → Dropout → Linear output neuron.
**Total parameters:** ≈ 4.26 million, trainable: 4.26 million.
### Training Setup:
**Optimizer:** Adam
**Loss:** Mean Squared Error (MSE)
**Metrics:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
**Callbacks:** 
- EarlyStopping to prevent overfitting
- ReduceLROnPlateau for adaptive learning rate scheduling
**Epochs:** 15  Batch Size: 64  Image Size: 128×128
### Model Evaluation	
**MAE:**
- Training: ~696 K USD
- Validation: ~703 K USD
**RMSE:**
- Training: ~785K USD
- Validation: ~796K USD
### Results Visualization
Plotted **RMSE** vs **Epochs** for both **training** and **validation** phases.
Model shows **stable convergence without overfitting** across modalities.
