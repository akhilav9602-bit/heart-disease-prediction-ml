# Heart Disease Prediction Using Machine Learning

## 1. Project Overview

This project uses machine learning classification algorithms to predict the presence of heart disease based on patient health attributes.

Two machine learning algorithms, Logistic Regression and Random Forest, are trained and evaluated to compare their performance.

## 2. Objectives

- Load and explore the heart disease dataset.
- Perform data preprocessing and handle duplicate records.
- Convert categorical features into numerical values.
- Scale numerical features.
- Train Logistic Regression and Random Forest models.
- Evaluate and compare model performance.
- Save the trained model and preprocessing files.
- Predict outcomes for new patient data.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## 4. Dataset

The project uses `heart.csv`, which contains patient health attributes and a target column named `HeartDisease`.

The target variable represents:
- `0` – No heart disease
- `1` – Heart disease

The dataset includes features such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, maximum heart rate, and other health-related attributes.

## 5. Project Workflow

### Step 1: Data Loading
Load the dataset using Pandas.

### Step 2: Data Preprocessing
- Check for missing values.
- Identify and remove duplicate records.
- Encode categorical variables using one-hot encoding.
- Separate features and the target variable.
- Split the dataset into training and testing sets.
- Apply StandardScaler to scale the features.

### Step 3: Model Training
Train the following classification algorithms:
- Logistic Regression
- Random Forest Classifier

### Step 4: Model Evaluation
Evaluate the models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Score

### Step 5: Model Comparison
Compare the performance of both models using evaluation metrics and ROC curves.

### Step 6: Model Saving
Save the trained Logistic Regression model, scaler, and feature column names using Joblib.

### Step 7: Prediction
Use the saved model to predict outcomes for new patient data.

## 6. Model Performance

The following results were obtained during model evaluation on the test dataset.

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 88.59% | 86.96% |
| Precision | 87.16% | 87.50% |
| Recall | 93.14% | 89.22% |
| F1-score | 90.05% | 88.35% |
| ROC-AUC | 0.9297 | 0.9314 |

Logistic Regression achieved higher accuracy, recall, and F1-score in this evaluation. Random Forest achieved slightly higher precision and ROC-AUC.

These results are specific to this train-test split and do not establish clinical effectiveness.

## 7. Project Structure

```text
heart_db/
│
├── data/
│   └── heart.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── data_loader.py
├── preprocessing.py
├── train_models.py
├── evaluate_models.py
├── save_model.py
├── main.py
├── predict.py
├── README.md
└── .gitignore