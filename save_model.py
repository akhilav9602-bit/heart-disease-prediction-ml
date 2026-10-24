import joblib
import os

def save_model(model, filename="models/best_model.pkl"):
    # Create the models folder
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(model, filename)

    print(f"Model saved successfully at {filename}")