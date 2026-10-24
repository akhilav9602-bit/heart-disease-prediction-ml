
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):

    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum())

    # Check for duplicate rows
    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert categorical columns into numerical columns
    df = pd.get_dummies(
        df,
        columns=[
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope"
        ],
        drop_first=True,
        dtype=int
    )
    # Save the feature column names
    X = df.drop("HeartDisease", axis=1)

    joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)
    print("\nDataset after encoding:")
    print(df.head())

    print("\nUpdated columns:")
    print(df.columns.tolist())

    # Separate features and target
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler for future predictions
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nScaler saved successfully!")

    print("\nTraining data scaled successfully!")
    print("Testing data scaled successfully!")

    print("\nTraining features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)

    return X_train, X_test, y_train, y_test
