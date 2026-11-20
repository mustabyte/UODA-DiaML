
from pathlib import Path
from math import isclose, isfinite

import joblib
import pandas as pd
import pytest


# Locate the saved machine learning artifacts.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_pipeline.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.joblib"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

DECISION_THRESHOLD = 0.40

VALID_INPUT = {
    "Pregnancies": 2,
    "Glucose": 145,
    "BloodPressure": 78,
    "SkinThickness": 25,
    "Insulin": 110,
    "BMI": 29.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35,
}

LOWER_PROBABILITY_INPUT = {
    "Pregnancies": 1,
    "Glucose": 95,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 23.5,
    "DiabetesPedigreeFunction": 0.25,
    "Age": 25,
}


# Load the trained model once for the test session.
@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.is_file(), "Saved model file is missing."

    pipeline = joblib.load(MODEL_PATH)

    assert hasattr(pipeline, "predict_proba")
    assert 1 in pipeline.classes_

    return pipeline


# Convert a measurement dictionary into the model's expected format.
def make_dataframe(measurements):
    return pd.DataFrame(
        [measurements],
        columns=FEATURE_COLUMNS,
    )


# Retrieve the positive-class probability.
def get_positive_probability(model, measurements):
    dataframe = make_dataframe(measurements)

    classes = list(model.classes_)
    positive_index = classes.index(1)

    return float(
        model.predict_proba(dataframe)[0][positive_index]
    )


# Verify that both saved artifacts exist.
def test_model_artifacts_exist():
    assert MODEL_PATH.is_file()
    assert CONFIG_PATH.is_file()


# Verify the model supports binary probability prediction.
def test_model_supports_probability_prediction(model):
    dataframe = make_dataframe(VALID_INPUT)

    probabilities = model.predict_proba(dataframe)

    assert probabilities.shape == (1, 2)
    assert isclose(
        float(probabilities[0].sum()),
        1.0,
        abs_tol=1e-10,
    )


# Verify the positive-class probability is valid.
def test_probability_is_between_zero_and_one(model):
    probability = get_positive_probability(
        model,
        VALID_INPUT,
    )

    assert isfinite(probability)
    assert 0.0 <= probability <= 1.0


# Verify identical measurements produce identical predictions.
def test_prediction_is_reproducible(model):
    first_probability = get_positive_probability(
        model,
        VALID_INPUT,
    )

    second_probability = get_positive_probability(
        model,
        VALID_INPUT,
    )

    assert isclose(
        first_probability,
        second_probability,
        abs_tol=1e-12,
    )


# Verify the model handles different valid measurements.
def test_model_responds_to_input_changes(model):
    first_probability = get_positive_probability(
        model,
        VALID_INPUT,
    )

    second_probability = get_positive_probability(
        model,
        LOWER_PROBABILITY_INPUT,
    )

    assert not isclose(
        first_probability,
        second_probability,
        abs_tol=1e-12,
    )


# Verify probabilities are classified using the selected threshold.
@pytest.mark.parametrize(
    "probability, expected_class",
    [
        (0.00, 0),
        (0.39, 0),
        (0.40, 1),
        (0.57, 1),
        (1.00, 1),
    ],
)
def test_classification_threshold(probability, expected_class):
    predicted_class = int(
        probability >= DECISION_THRESHOLD
    )

    assert predicted_class == expected_class


# Verify the model reproduces the previously observed positive result.
def test_known_positive_prediction(model):
    probability = get_positive_probability(
        model,
        VALID_INPUT,
    )

    predicted_class = int(
        probability >= DECISION_THRESHOLD
    )

    assert predicted_class == 1
    assert round(probability * 100, 1) == 57.1


# Verify the model reproduces the previously observed negative result.
def test_known_negative_prediction(model):
    probability = get_positive_probability(
        model,
        LOWER_PROBABILITY_INPUT,
    )

    predicted_class = int(
        probability >= DECISION_THRESHOLD
    )

    assert predicted_class == 0
    assert round(probability * 100, 1) == 3.3


# Verify the saved configuration can be loaded.
def test_model_configuration_is_valid():
    assert CONFIG_PATH.is_file()

    config = joblib.load(CONFIG_PATH)

    assert isinstance(config, dict)
    assert len(config) > 0
