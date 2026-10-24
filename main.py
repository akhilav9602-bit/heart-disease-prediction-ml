import matplotlib.pyplot as plt

from data_loader import load_data
from preprocessing import preprocess_data
from train_models import train_model
from evaluate_models import evaluate_model
from save_model import save_model


# Step 1: Load the dataset
df = load_data()


# Step 2: Preprocess the dataset and split into training and testing sets
X_train, X_test, y_train, y_test = preprocess_data(df)


# Step 3: Train Logistic Regression
logistic_model = train_model(
    X_train,
    y_train,
    model_type="logistic"
)


# Step 4: Save the Logistic Regression model
save_model(logistic_model)


# Step 5: Evaluate Logistic Regression
evaluate_model(
    logistic_model,
    X_test,
    y_test,
    model_name="Logistic Regression"
)


# Step 6: Train Random Forest
rf_model = train_model(
    X_train,
    y_train,
    model_type="random_forest"
)


# Step 7: Evaluate Random Forest
evaluate_model(
    rf_model,
    X_test,
    y_test,
    model_name="Random Forest"
)


# Step 8: Compare both models using ROC curves
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()