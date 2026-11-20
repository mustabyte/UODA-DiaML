
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Locate the saved model and configuration files
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
CONFIG_PATH = MODELS_DIR / "model_config.joblib"


# Load the trained model and prediction settings
def load_prediction_model():
    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)

    return model, config


# Validate and organize the eight input measurements
def prepare_input(patient_data, feature_names):
    if not isinstance(patient_data, dict):
        raise ValueError("Patient data must be a dictionary.")

    # Check for missing or unexpected measurements
    missing = set(feature_names) - set(patient_data)
    extra = set(patient_data) - set(feature_names)

    if missing or extra:
        raise ValueError(
            f"Missing features: {sorted(missing)}; "
            f"Unexpected features: {sorted(extra)}"
        )

    # Arrange measurements in the original training order
    input_df = pd.DataFrame([
        {feature: patient_data[feature] for feature in feature_names}
    ])

    # Convert all measurements into numeric values
    try:
        input_df = input_df.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("All measurements must be numeric.") from error

    # Reject missing, infinite, or negative values
    values = input_df.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("All measurements must be finite numbers.")

    if (values < 0).any():
        raise ValueError("Measurements cannot be negative.")

    return input_df


# Predict the outcome using our saved Random Forest pipeline
def predict_diabetes(patient_data, model=None, config=None):
    if model is None or config is None:
        model, config = load_prediction_model()

    # Prepare input in the correct feature order
    input_df = prepare_input(
        patient_data,
        config["feature_names"]
    )

    # Calculate the model's positive-class score
    probability = float(model.predict_proba(input_df)[0, 1])

    # Apply the threshold chosen during model development
    threshold = float(config["threshold"])
    prediction = int(probability >= threshold)

    # Return prediction information for the Streamlit application
    return {
        "prediction": prediction,
        "prediction_label": (
            "Predicted Positive" if prediction == 1
            else "Predicted Negative"
        ),
        "probability": round(probability, 4),
        "probability_percent": round(probability * 100, 2),
        "threshold": threshold,
        "model_name": config["model_name"]
    }
