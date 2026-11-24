
import numpy as np
import pandas as pd
import pytest

from components.validation import FEATURE_COLUMNS

from components.batch_validation import (
    validate_batch_input,
)

from ml.predict import (
    load_prediction_model,
    predict_diabetes,
)

from ml.batch_predict import (
    BatchPredictionResult,
    predict_batch,
)


@pytest.fixture(scope="module")
def trained_model():
    return load_prediction_model()


@pytest.fixture
def positive_record():
    return {
        "Pregnancies": 2,
        "Glucose": 145,
        "BloodPressure": 78,
        "SkinThickness": 25,
        "Insulin": 110,
        "BMI": 29.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 35,
    }


@pytest.fixture
def negative_record():
    return {
        "Pregnancies": 1,
        "Glucose": 95,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 23.5,
        "DiabetesPedigreeFunction": 0.25,
        "Age": 25,
    }


@pytest.fixture
def valid_batch(
    positive_record,
    negative_record,
):
    dataframe = pd.DataFrame([
        positive_record,
        negative_record,
    ])

    return validate_batch_input(dataframe)


def test_batch_result_type(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    assert isinstance(
        result,
        BatchPredictionResult,
    )


def test_batch_prediction_count(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    assert result.total_processed == 2
    assert len(result.results) == 2


def test_batch_prediction_columns(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    expected_columns = [
        "File Row",
        *FEATURE_COLUMNS,
        "Prediction",
        "Classification",
        "Probability",
        "Probability (%)",
        "Threshold",
        "Model",
    ]

    assert (
        list(result.results.columns)
        == expected_columns
    )


def test_probability_range(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    assert result.results[
        "Probability"
    ].between(0, 1).all()


def test_classification_threshold(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    expected = (
        result.results["Probability"]
        >= config["threshold"]
    ).astype(int)

    assert (
        result.results["Prediction"] == expected
    ).all()


def test_classification_labels(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    expected = np.where(
        result.results["Prediction"] == 1,
        "Positive",
        "Negative",
    )

    assert np.array_equal(
        result.results["Classification"],
        expected,
    )


def test_positive_negative_counts(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    assert (
        result.positive_count
        + result.negative_count
        == result.total_processed
    )


def test_classification_rates(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    assert (
        result.positive_rate
        + result.negative_rate
    ) == pytest.approx(100)


def test_average_probability(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    assert (
        result.average_probability
        == pytest.approx(
            result.results["Probability"].mean()
        )
    )


def test_model_metadata(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    assert (
        result.model_name
        == config["model_name"]
    )

    assert (
        result.threshold
        == config["threshold"]
    )


def test_individual_batch_consistency(
    positive_record,
    negative_record,
    trained_model,
):
    model, config = trained_model

    records = [
        positive_record,
        negative_record,
    ]

    validation = validate_batch_input(
        pd.DataFrame(records)
    )

    batch_result = predict_batch(
        validation,
        model,
        config,
    )

    for index, record in enumerate(records):
        individual = predict_diabetes(
            record,
            model,
            config,
        )

        batch_row = batch_result.results.iloc[
            index
        ]

        assert (
            batch_row["Prediction"]
            == individual["prediction"]
        )

        assert round(
            float(batch_row["Probability"]),
            4,
        ) == individual["probability"]


def test_invalid_records_excluded(
    positive_record,
    negative_record,
    trained_model,
):
    model, config = trained_model

    invalid_record = {
        **positive_record,
        "Glucose": None,
    }

    dataframe = pd.DataFrame([
        positive_record,
        invalid_record,
        negative_record,
    ])

    validation = validate_batch_input(
        dataframe
    )

    result = predict_batch(
        validation,
        model,
        config,
    )

    assert result.total_uploaded == 3
    assert result.total_processed == 2
    assert result.total_excluded == 1


def test_original_file_rows_preserved(
    positive_record,
    negative_record,
    trained_model,
):
    model, config = trained_model

    invalid_record = {
        **positive_record,
        "BMI": -20,
    }

    validation = validate_batch_input(
        pd.DataFrame([
            positive_record,
            invalid_record,
            negative_record,
        ])
    )

    result = predict_batch(
        validation,
        model,
        config,
    )

    assert result.results[
        "File Row"
    ].tolist() == [2, 4]


def test_warning_records_processed(
    positive_record,
    trained_model,
):
    model, config = trained_model

    warning_record = {
        **positive_record,
        "Glucose": 280,
    }

    validation = validate_batch_input(
        pd.DataFrame([warning_record])
    )

    result = predict_batch(
        validation,
        model,
        config,
    )

    assert result.total_processed == 1
    assert result.total_warnings == 1
    assert result.total_excluded == 0


def test_duplicate_records_processed(
    positive_record,
    trained_model,
):
    model, config = trained_model

    validation = validate_batch_input(
        pd.DataFrame([
            positive_record,
            positive_record,
        ])
    )

    result = predict_batch(
        validation,
        model,
        config,
    )

    assert result.total_processed == 2
    assert result.total_warnings == 1

    assert result.results[
        "File Row"
    ].tolist() == [2, 3]

    assert (
        result.results.iloc[0]["Probability"]
        == pytest.approx(
            result.results.iloc[1]["Probability"]
        )
    )


def test_single_record_batch(
    positive_record,
    trained_model,
):
    model, config = trained_model

    validation = validate_batch_input(
        pd.DataFrame([positive_record])
    )

    result = predict_batch(
        validation,
        model,
        config,
    )

    assert result.total_processed == 1


def test_no_valid_records(
    positive_record,
    trained_model,
):
    model, config = trained_model

    invalid_record = {
        **positive_record,
        "Glucose": None,
    }

    validation = validate_batch_input(
        pd.DataFrame([invalid_record])
    )

    with pytest.raises(
        ValueError,
        match="No valid records",
    ):
        predict_batch(
            validation,
            model,
            config,
        )


def test_invalid_file_structure(
    positive_record,
    trained_model,
):
    model, config = trained_model

    dataframe = pd.DataFrame([
        positive_record
    ]).drop(columns=["Age"])

    validation = validate_batch_input(
        dataframe
    )

    with pytest.raises(
        ValueError,
        match="structural errors",
    ):
        predict_batch(
            validation,
            model,
            config,
        )


def test_invalid_validation_object():
    with pytest.raises(TypeError):
        predict_batch({})


def test_invalid_threshold(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    invalid_config = {
        **config,
        "threshold": 1.5,
    }

    with pytest.raises(
        ValueError,
        match="threshold is invalid",
    ):
        predict_batch(
            valid_batch,
            model,
            invalid_config,
        )


def test_batch_summary(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    result = predict_batch(
        valid_batch,
        model,
        config,
    )

    summary = result.get_summary()

    assert summary["total_processed"] == 2
    assert summary["total_uploaded"] == 2
    assert summary["total_excluded"] == 0
    assert summary["threshold"] == config["threshold"]


def test_threshold_boundary(
    positive_record,
):
    class MockModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, dataframe):
            return np.array([
                [0.60001, 0.39999],
                [0.60, 0.40],
                [0.59999, 0.40001],
            ])

    config = {
        "feature_names": FEATURE_COLUMNS,
        "threshold": 0.40,
        "positive_class": 1,
        "model_name": "Test Model",
    }

    validation = validate_batch_input(
        pd.DataFrame([
            positive_record,
            positive_record,
            positive_record,
        ])
    )

    result = predict_batch(
        validation,
        MockModel(),
        config,
    )

    assert result.results[
        "Prediction"
    ].tolist() == [0, 1, 1]


def test_positive_class_index(
    valid_batch,
):
    class MockModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, dataframe):
            return np.array([
                [0.70, 0.30],
                [0.20, 0.80],
            ])

    config = {
        "feature_names": FEATURE_COLUMNS,
        "threshold": 0.40,
        "positive_class": 1,
        "model_name": "Test Model",
    }

    result = predict_batch(
        valid_batch,
        MockModel(),
        config,
    )

    assert result.results[
        "Prediction"
    ].tolist() == [1, 0]


def test_invalid_model_probabilities(
    valid_batch,
):
    class MockModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, dataframe):
            return np.array([
                [0.30, 0.70],
                [np.nan, np.nan],
            ])

    config = {
        "feature_names": FEATURE_COLUMNS,
        "threshold": 0.40,
        "positive_class": 1,
        "model_name": "Test Model",
    }

    with pytest.raises(
        ValueError,
        match="non-finite probabilities",
    ):
        predict_batch(
            valid_batch,
            MockModel(),
            config,
        )


def test_automatic_model_loading(
    valid_batch,
):
    result = predict_batch(
        valid_batch
    )

    assert result.total_processed == 2
    assert result.model_name == "Random Forest"


def test_batch_does_not_modify_validation(
    valid_batch,
    trained_model,
):
    model, config = trained_model

    original_data = (
        valid_batch.valid_data.copy(deep=True)
    )

    predict_batch(
        valid_batch,
        model,
        config,
    )

    pd.testing.assert_frame_equal(
        valid_batch.valid_data,
        original_data,
    )
