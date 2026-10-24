import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_loader import load_data
from preprocessing import preprocess_data
from train_models import train_model


# 1. Set the MLflow experiment
mlflow.set_experiment("Heart Disease Prediction")


# 2. Load the dataset
df = load_data()


# 3. Preprocess the dataset
X_train, X_test, y_train, y_test = preprocess_data(df)


# 4. Define the models to train
models = {
    "Logistic Regression": "logistic",
    "Random Forest": "random_forest"
}


# 5. Train and track each model
for model_name, model_type in models.items():

    with mlflow.start_run(run_name=model_name):

        print(f"\nTraining {model_name}...")

        # Train the model
        model = train_model(
            X_train,
            y_train,
            model_type
        )

        # Log model parameters
        mlflow.log_param("model_type", model_name)

        if model_type == "logistic":
            mlflow.log_param("random_state", 42)

        elif model_type == "random_forest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)

        # Generate predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        # Log metrics in MLflow
        mlflow.log_metrics(metrics)

        # Log the trained model
           # Log the trained model
        if model_type == "random_forest":
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )

        # Display results
        print(f"\n{model_name} Results:")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print(f"\n{model_name} logged successfully in MLflow!")


print("\nAll MLflow experiments completed successfully!")
