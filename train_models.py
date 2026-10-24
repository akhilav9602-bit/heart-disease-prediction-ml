from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, model_type="logistic"):

    # Select the model
    if model_type == "logistic":
        model = LogisticRegression(random_state=42)

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    else:
        raise ValueError("Invalid model type")

    # Train the model
    model.fit(X_train, y_train)

    print(f"{model_type} model trained successfully!")

    return model