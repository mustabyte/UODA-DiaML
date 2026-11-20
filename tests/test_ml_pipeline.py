
import unittest

import numpy as np
import pandas as pd

from ml.preprocessing import FEATURE_COLUMNS, create_preprocessor
from ml.predict import (
    load_prediction_model,
    prepare_input,
    predict_diabetes
)


# Test preprocessing and prediction integration
class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the saved model once for all tests
        cls.model, cls.config = load_prediction_model()

        # Create a sample patient record
        cls.patient = {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 80,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 35
        }

    # Check that the saved model and configuration are valid
    def test_model_loading(self):
        self.assertTrue(hasattr(self.model, "predict_proba"))
        self.assertEqual(self.config["threshold"], 0.40)
        self.assertEqual(len(self.config["feature_names"]), 8)

    # Check that preprocessing handles invalid zero measurements
    def test_preprocessing(self):

        # Create two records with valid measurements
        sample = pd.DataFrame([
            self.patient,
            {
                **self.patient,
                "Glucose": 150,
                "BMI": 32.0
            }
        ])

        # Set the first record's glucose value to an invalid zero
        sample.loc[0, "Glucose"] = 0

        # Apply preprocessing to the sample records
        processed = create_preprocessor().fit_transform(sample)

        # Verify that all eight features are preserved
        self.assertEqual(processed.shape, (2, 8))

        # Verify that no missing or infinite values remain
        self.assertTrue(np.isfinite(processed).all())

        # Verify that invalid glucose was replaced with the valid median
        imputed = create_preprocessor(scale=False).fit_transform(sample)
        glucose_index = FEATURE_COLUMNS.index("Glucose")

        self.assertAlmostEqual(imputed[0, glucose_index], 150.0)

    # Check that input features follow the correct training order
    def test_input_order(self):

        # Reverse the input dictionary's feature order
        reversed_patient = dict(reversed(list(self.patient.items())))

        # Prepare the input using the expected feature order
        prepared = prepare_input(
            reversed_patient,
            FEATURE_COLUMNS
        )

        self.assertEqual(
            prepared.columns.tolist(),
            FEATURE_COLUMNS
        )

    # Check that a valid patient record produces a prediction
    def test_prediction(self):

        # Generate a prediction using the saved model
        result = predict_diabetes(
            self.patient,
            self.model,
            self.config
        )

        # Verify that the prediction result contains valid values
        self.assertIn(result["prediction"], [0, 1])
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)
        self.assertEqual(result["model_name"], "Random Forest")
        self.assertEqual(result["threshold"], 0.40)

    # Check that missing measurements are rejected
    def test_missing_feature(self):

        # Remove glucose from the patient record
        incomplete_patient = self.patient.copy()
        incomplete_patient.pop("Glucose")

        # Verify that missing measurements raise an error
        with self.assertRaises(ValueError):
            prepare_input(incomplete_patient, FEATURE_COLUMNS)

    # Check that negative measurements are rejected
    def test_negative_value(self):

        # Create a patient record with an invalid negative BMI
        invalid_patient = self.patient.copy()
        invalid_patient["BMI"] = -25

        # Verify that negative values raise an error
        with self.assertRaises(ValueError):
            prepare_input(invalid_patient, FEATURE_COLUMNS)

    # Check that nonnumeric measurements are rejected
    def test_nonnumeric_value(self):

        # Create a patient record with an invalid age
        invalid_patient = self.patient.copy()
        invalid_patient["Age"] = "unknown"

        # Verify that nonnumeric values raise an error
        with self.assertRaises(ValueError):
            prepare_input(invalid_patient, FEATURE_COLUMNS)


# Run the tests when this file is executed directly
if __name__ == "__main__":
    unittest.main()
