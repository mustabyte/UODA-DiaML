
from dataclasses import dataclass

import numpy as np
import pandas as pd

from components.batch_validation import (
    BatchValidationResult,
    ROW_NUMBER_COLUMN,
    prepare_batch_dataframe,
)

from ml.predict import load_prediction_model


# Store batch prediction results and metadata
@dataclass
class BatchPredictionResult:
    results: pd.DataFrame
    total_uploaded: int
    total_processed: int
    total_excluded: int
    total_warnings: int
    positive_count: int
    negative_count: int
    average_probability: float
    threshold: float
    model_name: str

    @property
    def positive_rate(self):
        if self.total_processed == 0:
            return 0.0

        return (
            self.positive_count
            / self.total_processed
            * 100
        )

    @property
    def negative_rate(self):
        if self.total_processed == 0:
            return 0.0

        return (
            self.negative_count
            / self.total_processed
            * 100
        )

    def get_summary(self):
        return {
            "total_uploaded": self.total_uploaded,
            "total_processed": self.total_processed,
            "total_excluded": self.total_excluded,
            "total_warnings": self.total_warnings,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_rate": self.positive_rate,
            "negative_rate": self.negative_rate,
            "average_probability": (
                self.average_probability
            ),
            "threshold": self.threshold,
            "model_name": self.model_name,
        }


# Validate the supplied batch validation result
def validate_batch_result(validation_result):
    if not isinstance(
        validation_result,
        BatchValidationResult,
    ):
        raise TypeError(
            "A BatchValidationResult object is required."
        )

    if not validation_result.is_file_valid:
        raise ValueError(
            "The uploaded dataset has structural errors."
        )

    if not validation_result.can_predict:
        raise ValueError(
            "No valid records are available for prediction."
        )

    if validation_result.valid_data.empty:
        raise ValueError(
            "The accepted dataset is empty."
        )

    if ROW_NUMBER_COLUMN not in (
        validation_result.valid_data.columns
    ):
        raise ValueError(
            "Original file row numbers are missing."
        )

    if (
        len(validation_result.valid_data)
        != validation_result.valid_rows
    ):
        raise ValueError(
            "The accepted record count is inconsistent."
        )

    if (
        validation_result.valid_data[
            ROW_NUMBER_COLUMN
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate file row numbers were detected."
        )


# Prepare accepted measurements for model inference
def prepare_batch_input(validation_result, feature_names):
    dataframe = prepare_batch_dataframe(
        validation_result
    )

    missing_features = (
        set(feature_names) - set(dataframe.columns)
    )

    extra_features = (
        set(dataframe.columns) - set(feature_names)
    )

    if missing_features or extra_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}; "
            f"Unexpected features: {sorted(extra_features)}"
        )

    dataframe = dataframe.loc[
        :, feature_names
    ].copy()

    try:
        dataframe = dataframe.apply(
            pd.to_numeric,
            errors="raise",
        )

    except (ValueError, TypeError) as error:
        raise ValueError(
            "All measurements must be numeric."
        ) from error

    values = dataframe.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "All measurements must be finite numbers."
        )

    if (values < 0).any():
        raise ValueError(
            "Measurements cannot be negative."
        )

    return dataframe


# Locate the configured positive class
def get_positive_class_index(model, config):
    positive_class = config.get(
        "positive_class",
        1,
    )

    classes = np.asarray(model.classes_)

    matching_indices = np.flatnonzero(
        classes == positive_class
    )

    if len(matching_indices) != 1:
        raise ValueError(
            "The configured positive class was not "
            "found in the trained model."
        )

    return int(matching_indices[0])


# Validate model prediction probabilities
def validate_probabilities(
    probabilities,
    expected_count,
):
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "The model returned an invalid probability shape."
        )

    if len(probabilities) != expected_count:
        raise ValueError(
            "The number of probabilities does not "
            "match the accepted record count."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "The model returned non-finite probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Prediction probabilities must be "
            "between zero and one."
        )

    return probabilities


# Generate predictions for all accepted records
def predict_batch(
    validation_result,
    model=None,
    config=None,
):
    validate_batch_result(validation_result)

    if model is None or config is None:
        model, config = load_prediction_model()

    feature_names = list(
        config["feature_names"]
    )

    threshold = float(
        config["threshold"]
    )

    model_name = config["model_name"]

    if (
        not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "The configured prediction threshold is invalid."
        )

    input_df = prepare_batch_input(
        validation_result,
        feature_names,
    )

    positive_index = get_positive_class_index(
        model,
        config,
    )

    probability_matrix = np.asarray(
        model.predict_proba(input_df)
    )

    if (
        probability_matrix.ndim != 2
        or probability_matrix.shape[0] != len(input_df)
        or positive_index >= probability_matrix.shape[1]
    ):
        raise ValueError(
            "The model returned an invalid "
            "probability matrix."
        )

    probabilities = probability_matrix[
        :, positive_index
    ]

    probabilities = validate_probabilities(
        probabilities,
        len(input_df),
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    classifications = np.where(
        predictions == 1,
        "Positive",
        "Negative",
    )

    results = input_df.reset_index(
        drop=True
    ).copy()

    results.insert(
        0,
        ROW_NUMBER_COLUMN,
        validation_result.valid_data[
            ROW_NUMBER_COLUMN
        ].to_numpy(),
    )

    results["Prediction"] = predictions

    results["Classification"] = classifications

    results["Probability"] = probabilities

    results["Probability (%)"] = np.round(
        probabilities * 100,
        2,
    )

    results["Threshold"] = threshold

    results["Model"] = model_name

    positive_count = int(
        (predictions == 1).sum()
    )

    negative_count = int(
        (predictions == 0).sum()
    )

    return BatchPredictionResult(
        results=results,
        total_uploaded=validation_result.total_rows,
        total_processed=len(results),
        total_excluded=validation_result.invalid_rows,
        total_warnings=validation_result.warning_rows,
        positive_count=positive_count,
        negative_count=negative_count,
        average_probability=float(
            probabilities.mean()
        ),
        threshold=threshold,
        model_name=model_name,
    )
