import joblib
import pandas as pd

# Step 1: Load the saved model, scaler, and feature columns
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("Model, scaler, and feature columns loaded successfully!")

# Step 2: Enter sample patient details
patient = {
    "Age": 55,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 140,
    "Cholesterol": 250,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "Y",
    "Oldpeak": 1.5,
    "ST_Slope": "Flat"
}

# Step 3: Encode categorical features
patient_data = pd.DataFrame([{
    "Age": patient["Age"],
    "RestingBP": patient["RestingBP"],
    "Cholesterol": patient["Cholesterol"],
    "FastingBS": patient["FastingBS"],
    "MaxHR": patient["MaxHR"],
    "Oldpeak": patient["Oldpeak"],
    "Sex_M": int(patient["Sex"] == "M"),
    "ChestPainType_ATA": int(patient["ChestPainType"] == "ATA"),
    "ChestPainType_NAP": int(patient["ChestPainType"] == "NAP"),
    "ChestPainType_TA": int(patient["ChestPainType"] == "TA"),
    "RestingECG_Normal": int(patient["RestingECG"] == "Normal"),
    "RestingECG_ST": int(patient["RestingECG"] == "ST"),
    "ExerciseAngina_Y": int(patient["ExerciseAngina"] == "Y"),
    "ST_Slope_Flat": int(patient["ST_Slope"] == "Flat"),
    "ST_Slope_Up": int(patient["ST_Slope"] == "Up")
}])

# Step 4: Arrange features in the correct order
patient_data = patient_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# Step 5: Scale the patient data
patient_scaled = scaler.transform(patient_data)

# Step 6: Predict heart disease
prediction = model.predict(patient_scaled)[0]
probability = model.predict_proba(patient_scaled)[0][1]

# Step 7: Display the results
print("\nPrediction:", prediction)
print(
    "Estimated probability of heart disease:",
    round(probability * 100, 2),
    "%"
)

if prediction == 1:
    print("Model classification: Heart disease")
else:
    print("Model classification: No heart disease")