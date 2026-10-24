# Heart Disease Prediction Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to predict the presence of heart disease based on patient health information. It includes data preprocessing, model training, evaluation, prediction, and experiment tracking using MLflow.

## Project Objectives

- Analyze heart disease data.
- Clean and preprocess the dataset.
- Train and compare machine learning classification models.
- Evaluate model performance using multiple metrics.
- Save trained models for future predictions.
- Track experiments and manage a registered model using MLflow.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- MLflow

## Dataset

The dataset contains patient health information and a target variable, `HeartDisease`, indicating the presence or absence of heart disease.

The dataset includes features such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, and exercise-related information.

**Note:** Ensure that the dataset source and licence permit its use and redistribution before sharing the dataset publicly.

## Project Workflow

1. Load the dataset.
2. Inspect missing values and duplicate records.
3. Remove duplicate records.
4. Encode categorical variables.
5. Split the data into training and testing sets.
6. Scale the features using StandardScaler.
7. Train Logistic Regression and Random Forest models.
8. Evaluate model performance.
9. Save model-related files.
10. Use the trained model to make predictions.
11. Track experiments and metrics using MLflow.
12. Register a trained model in the MLflow Model Registry.

## Machine Learning Models

Two classification algorithms were trained and evaluated:

- Logistic Regression
- Random Forest Classifier

## Model Evaluation Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 88.59% | 86.96% |
| Precision | 87.16% | 87.50% |
| Recall | 93.14% | 89.22% |
| F1-score | 90.05% | 88.35% |
| ROC-AUC | 92.97% | 93.14% |

Logistic Regression achieved higher accuracy, recall, and F1-score in this evaluation. Random Forest achieved slightly higher precision and ROC-AUC.

These results are based on the current train-test split and do not establish clinical reliability.

## MLflow Experiment Tracking

MLflow was used to track and compare machine learning experiments.

### MLflow Features Practised

- Created an experiment named `Heart Disease Prediction`.
- Logged model parameters.
- Logged evaluation metrics.
- Logged trained models.
- Compared model performance across experiments.
- Registered the Logistic Regression model as `HeartDiseasePrediction`, Version 1.

### Run MLflow Experiments

After installing the required dependencies and activating your Python environment, run:

```bash
python train_with_mlflow.py
```

To launch the MLflow UI using the local tracking database, run:

```bash
python -m mlflow ui --backend-store-uri "sqlite:///C:/Users/akhil/OneDrive/Documents/heart_db/mlflow.db"
```

Then open http://127.0.0.1:5000 in your browser.

**Note:** The tracking database path above is specific to the original development computer. Update it to match your own project location if you clone this repository.

## Project Structure

```text
heart-disease-prediction-ml/
│
├── data/
│   └── heart.csv
│
├── models/
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── best_model.pkl
│
├── data_loader.py
├── preprocessing.py
├── train_models.py
├── evaluate_models.py
├── save_model.py
├── main.py
├── predict.py
├── train_with_mlflow.py
├── README.md
└── .gitignore
```

*The `models/` files are generated during project execution and may not be included in the repository.*

## How to Run the Project

1. Install Python.
2. Clone or download this repository.
3. Create and activate a virtual environment.
4. Install the required Python libraries.
5. Ensure the dataset is available at `data/heart.csv`.
6. Run the main project:

```bash
python main.py
```

7. To run MLflow experiment tracking:

```bash
python train_with_mlflow.py
```

## Key Learnings

- Data cleaning and preprocessing
- Categorical feature encoding
- Feature scaling
- Classification model training
- Model evaluation and comparison
- Saving and loading trained models
- Making predictions with a trained model
- Experiment tracking and model registration using MLflow

## Disclaimer

This project is for educational purposes only. Its predictions are not medical advice and should not be used to diagnose or make treatment decisions about heart disease.
