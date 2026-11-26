import unittest

import numpy as np
import pandas as pd

from ml.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    INVALID_ZERO_COLUMNS,
    InvalidZeroTransformer,
    load_dataset,
    validate_dataset,
    create_preprocessor
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a valid sample dataset
        self.dataset = pd.DataFrame(
            [
                [2, 120, 70, 25, 80, 28.5, 0.5, 35, 0],
                [3, 150, 80, 30, 100, 32.0, 0.6, 42, 1]
            ],
            columns=FEATURE_COLUMNS + [TARGET_COLUMN]
        )

    # Verify that the original dataset loads correctly
    def test_dataset_loading(self):
        dataset = load_dataset()

        self.assertEqual(dataset.shape, (768, 9))
        self.assertEqual(
            dataset.columns.tolist(),
            FEATURE_COLUMNS + [TARGET_COLUMN]
        )

    # Verify that valid datasets are accepted
    def test_valid_dataset(self):
        result = validate_dataset(self.dataset)

        self.assertEqual(result.shape, (2, 9))

    # Verify that empty datasets are rejected
    def test_empty_dataset(self):
        with self.assertRaises(ValueError):
            validate_dataset(self.dataset.iloc[0:0])

    # Verify that missing columns are rejected
    def test_missing_column(self):
        invalid = self.dataset.drop(columns=["Glucose"])

        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    # Verify that invalid outcome labels are rejected
    def test_invalid_outcome(self):
        invalid = self.dataset.copy()
        invalid.loc[0, TARGET_COLUMN] = 2

        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    # Verify that missing target values are rejected
    def test_missing_outcome(self):
        invalid = self.dataset.copy()
        invalid.loc[0, TARGET_COLUMN] = np.nan

        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    # Verify that negative measurements are rejected
    def test_negative_measurement(self):
        invalid = self.dataset.copy()
        invalid.loc[0, "BMI"] = -20

        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    # Verify that infinite measurements are rejected
    def test_infinite_measurement(self):
        invalid = self.dataset.copy()
        invalid.loc[0, "BMI"] = np.inf

        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    # Verify that nonnumeric measurements are rejected
    def test_nonnumeric_measurement(self):
        invalid = self.dataset.copy()
        invalid["Glucose"] = ["unknown", "150"]

        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    # Verify that invalid zeros become missing values
    def test_invalid_zero_conversion(self):
        sample = self.dataset[FEATURE_COLUMNS].copy()
        sample.loc[0, INVALID_ZERO_COLUMNS] = 0

        transformed = InvalidZeroTransformer().fit_transform(sample)

        self.assertTrue(
            transformed.loc[0, INVALID_ZERO_COLUMNS].isna().all()
        )

    # Verify that zero pregnancies remain valid
    def test_valid_zero_preservation(self):
        sample = self.dataset[FEATURE_COLUMNS].copy()
        sample.loc[0, "Pregnancies"] = 0

        transformed = InvalidZeroTransformer().fit_transform(sample)

        self.assertEqual(transformed.loc[0, "Pregnancies"], 0)

    # Verify that preprocessing does not modify the original input
    def test_original_data_preservation(self):
        sample = self.dataset[FEATURE_COLUMNS].copy()
        sample.loc[0, "Glucose"] = 0

        original = sample.copy(deep=True)

        InvalidZeroTransformer().fit_transform(sample)

        pd.testing.assert_frame_equal(sample, original)

    # Verify that missing measurements are imputed
    def test_missing_value_imputation(self):
        sample = self.dataset[FEATURE_COLUMNS].copy()
        sample.loc[0, "BMI"] = np.nan

        processed = create_preprocessor(
            scale=False
        ).fit_transform(sample)

        self.assertTrue(np.isfinite(processed).all())
        self.assertEqual(processed.shape, (2, 8))

    # Verify preprocessing configuration
    def test_scaling_configuration(self):
        scaled = create_preprocessor(scale=True)
        unscaled = create_preprocessor(scale=False)

        self.assertIn("scaler", scaled.named_steps)
        self.assertNotIn("scaler", unscaled.named_steps)


if __name__ == "__main__":
    unittest.main()