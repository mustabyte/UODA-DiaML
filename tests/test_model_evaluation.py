import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from ml.preprocessing import (
    load_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

from ml.predict import load_prediction_model

from ml.evaluate_models import (
    validate_threshold,
    get_predictions,
    calculate_specificity,
    evaluate_model,
    get_confusion_matrix,
    get_roc_curve,
    get_precision_recall_curve
)


class TestModelEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the saved model and original dataset
        cls.model, cls.config = load_prediction_model()

        dataset = load_dataset()

        X = dataset[FEATURE_COLUMNS]
        y = dataset[TARGET_COLUMN]

        # Reproduce the original testing partition
        _, cls.X_test, _, cls.y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        cls.threshold = cls.config["threshold"]

    # Verify valid classification thresholds
    def test_valid_threshold(self):
        self.assertEqual(validate_threshold(0.40), 0.40)
        self.assertEqual(validate_threshold(0), 0.0)
        self.assertEqual(validate_threshold(1), 1.0)

    # Reject thresholds outside the valid range
    def test_invalid_threshold(self):
        for threshold in [-0.1, 1.1, 2]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    validate_threshold(threshold)

    # Reject nonnumeric thresholds
    def test_nonnumeric_threshold(self):
        for threshold in ["invalid", None]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    validate_threshold(threshold)

    # Reject nonfinite thresholds
    def test_nonfinite_threshold(self):
        for threshold in [np.nan, np.inf, -np.inf]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    validate_threshold(threshold)

    # Verify prediction and probability dimensions
    def test_prediction_dimensions(self):
        predictions, probabilities = get_predictions(
            self.model,
            self.X_test,
            self.threshold
        )

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(len(probabilities), len(self.X_test))

    # Verify that predictions contain binary classes
    def test_binary_predictions(self):
        predictions, _ = get_predictions(
            self.model,
            self.X_test,
            self.threshold
        )

        self.assertTrue(
            np.isin(predictions, [0, 1]).all()
        )

    # Verify that probabilities remain within the valid range
    def test_probability_range(self):
        _, probabilities = get_predictions(
            self.model,
            self.X_test,
            self.threshold
        )

        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all())
        self.assertTrue((probabilities <= 1).all())

    # Verify classification against the selected threshold
    def test_threshold_classification(self):
        predictions, probabilities = get_predictions(
            self.model,
            self.X_test,
            self.threshold
        )

        expected = (
            probabilities >= self.threshold
        ).astype(int)

        np.testing.assert_array_equal(predictions, expected)

    # Verify that all evaluation metrics are available
    def test_evaluation_metrics(self):
        metrics = evaluate_model(
            self.model,
            self.X_test,
            self.y_test,
            self.threshold
        )

        expected_metrics = {
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "specificity",
            "balanced_accuracy",
            "average_precision"
        }

        self.assertEqual(set(metrics.keys()), expected_metrics)

    # Verify the range of all evaluation metrics
    def test_metric_ranges(self):
        metrics = evaluate_model(
            self.model,
            self.X_test,
            self.y_test,
            self.threshold
        )

        for name, value in metrics.items():
            with self.subTest(metric=name):
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)

    # Verify the specificity calculation
    def test_specificity(self):
        actual = [0, 0, 0, 0, 1, 1]
        predicted = [0, 0, 0, 1, 1, 0]

        specificity = calculate_specificity(
            actual,
            predicted
        )

        self.assertAlmostEqual(specificity, 0.75)

    # Verify the confusion matrix dimensions and total
    def test_confusion_matrix(self):
        matrix = get_confusion_matrix(
            self.model,
            self.X_test,
            self.y_test,
            self.threshold
        )

        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.sum(), len(self.y_test))

    # Verify ROC curve coordinates
    def test_roc_curve(self):
        result = get_roc_curve(
            self.model,
            self.X_test,
            self.y_test
        )

        self.assertEqual(
            len(result["fpr"]),
            len(result["tpr"])
        )

        self.assertEqual(
            len(result["fpr"]),
            len(result["thresholds"])
        )

        self.assertEqual(result["fpr"][0], 0.0)
        self.assertEqual(result["tpr"][0], 0.0)
        self.assertEqual(result["fpr"][-1], 1.0)
        self.assertEqual(result["tpr"][-1], 1.0)

    # Verify Precision-Recall curve coordinates
    def test_precision_recall_curve(self):
        result = get_precision_recall_curve(
            self.model,
            self.X_test,
            self.y_test
        )

        self.assertEqual(
            len(result["precision"]),
            len(result["recall"])
        )

        self.assertEqual(
            len(result["precision"]),
            len(result["thresholds"]) + 1
        )

        self.assertGreaterEqual(
            result["average_precision"],
            0
        )

        self.assertLessEqual(
            result["average_precision"],
            1
        )

    # Verify balanced accuracy against recall and specificity
    def test_balanced_accuracy(self):
        metrics = evaluate_model(
            self.model,
            self.X_test,
            self.y_test,
            self.threshold
        )

        expected = (
            metrics["recall"] + metrics["specificity"]
        ) / 2

        self.assertAlmostEqual(
            metrics["balanced_accuracy"],
            expected
        )


if __name__ == "__main__":
    unittest.main()