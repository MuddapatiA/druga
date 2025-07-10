from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

# Initialize the FastAPI app
app = FastAPI(title="Drug A Eligibility Predictor")

# --- Load Model and Columns ---
# Construct paths relative to the current script
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'drug_a_predictor.joblib')
columns_path = os.path.join(base_dir, 'models', 'model_columns.json')

# Load the model and columns
model = joblib.load(model_path)
with open(columns_path, 'r') as f:
    model_columns = json.load(f)

# --- Define the input data structure using Pydantic ---
class PatientFeatures(BaseModel):
    patient_age: int
    patient_gender: str
    num_conditions: int
    physician_type: str
    physician_state: str
    location_type: str
    num_contraindications: int
    patient_is_high_risk: int

# --- API Endpoint for Prediction ---
@app.post("/predict")
def predict(features: PatientFeatures):
    """
    Accepts patient features in JSON format and returns the
    likelihood of NOT being treated with Drug A.
    """
    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([features.dict()])

    # One-hot encode the categorical features
    input_encoded = pd.get_dummies(input_df)

    # Align columns with the model's training columns
    # This adds missing columns and removes extra ones
    final_df = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction (returns probability of class 1: Treated)
    probability_treated = model.predict_proba(final_df)[:, 1][0]

    # Calculate the likelihood of not being treated
    likelihood_not_treated = 1.0 - probability_treated

    return {
        "likelihood_of_not_being_treated": round(likelihood_not_treated, 4),
        "alert_physician": bool(likelihood_not_treated > 0.5) # Example threshold
    }