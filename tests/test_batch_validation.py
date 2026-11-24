
import pandas as pd
import pytest

from components.batch_validation import (
    MAX_BATCH_ROWS,
    validate_batch_input,
    prepare_batch_dataframe,
)

from components.validation import FEATURE_COLUMNS


# Create a valid measurement record for testing.
@pytest.fixture
def valid_record():
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


def test_valid_batch(valid_record):
    dataframe = pd.DataFrame([valid_record])

    result = validate_batch_input(dataframe)

    assert result.is_file_valid
    assert result.can_predict
    assert result.total_rows == 1
    assert result.valid_rows == 1
    assert result.invalid_rows == 0
    assert result.clean_rows == 1
    assert result.success_rate == 100


def test_missing_column(valid_record):
    record = valid_record.copy()
    record.pop("Glucose")

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert not result.is_file_valid
    assert not result.can_predict
    assert "Glucose" in result.missing_columns


def test_empty_dataset():
    dataframe = pd.DataFrame(
        columns=FEATURE_COLUMNS
    )

    result = validate_batch_input(dataframe)

    assert not result.is_file_valid
    assert not result.can_predict


def test_missing_measurement(valid_record):
    record = {
        **valid_record,
        "Glucose": None,
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.invalid_rows == 1
    assert result.valid_rows == 0
    assert result.errors.iloc[0]["Field"] == "Glucose"
    assert result.errors.iloc[0]["File Row"] == 2


def test_invalid_numeric_value(valid_record):
    record = {
        **valid_record,
        "Glucose": "abc",
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.invalid_rows == 1
    assert result.errors.iloc[0]["Field"] == "Glucose"


def test_negative_measurement(valid_record):
    record = {
        **valid_record,
        "BMI": -25,
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.invalid_rows == 1
    assert result.errors.iloc[0]["Field"] == "BMI"


def test_invalid_integer(valid_record):
    record = {
        **valid_record,
        "Age": 25.5,
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.invalid_rows == 1
    assert result.errors.iloc[0]["Field"] == "Age"


def test_multiple_errors_in_one_row(valid_record):
    record = {
        **valid_record,
        "Glucose": None,
        "BMI": -20,
        "Age": "abc",
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.invalid_rows == 1
    assert len(result.errors) == 3

    assert set(result.errors["Field"]) == {
        "Glucose",
        "BMI",
        "Age",
    }


def test_warning_record(valid_record):
    record = {
        **valid_record,
        "Glucose": 280,
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.valid_rows == 1
    assert result.invalid_rows == 0
    assert result.warning_rows == 1
    assert result.clean_rows == 0
    assert result.has_warnings
    assert result.can_predict


def test_mixed_valid_and_invalid_rows(valid_record):
    dataframe = pd.DataFrame([
        valid_record,
        {**valid_record, "Glucose": None},
        {**valid_record, "Glucose": 280},
    ])

    result = validate_batch_input(dataframe)

    assert result.total_rows == 3
    assert result.valid_rows == 2
    assert result.invalid_rows == 1
    assert result.warning_rows == 1
    assert result.clean_rows == 1
    assert result.can_predict

    assert result.status == "partially_valid"


def test_original_row_numbers(valid_record):
    dataframe = pd.DataFrame([
        valid_record,
        {**valid_record, "BMI": -10},
        {**valid_record, "Glucose": 100},
    ])

    result = validate_batch_input(dataframe)

    assert result.valid_data["File Row"].tolist() == [2, 4]

    assert result.invalid_data[
        "File Row"
    ].tolist() == [3]


def test_extra_columns(valid_record):
    dataframe = pd.DataFrame([
        {**valid_record, "Outcome": 1}
    ])

    result = validate_batch_input(dataframe)

    assert result.is_file_valid
    assert result.valid_rows == 1
    assert "Outcome" in result.extra_columns
    assert "Outcome" not in result.valid_data.columns


def test_column_normalization(valid_record):
    dataframe = pd.DataFrame([valid_record])

    dataframe.columns = [
        f" {column.lower()} "
        for column in dataframe.columns
    ]

    result = validate_batch_input(dataframe)

    assert result.is_file_valid
    assert result.valid_rows == 1

    assert list(result.valid_data.columns) == [
        "File Row",
        *FEATURE_COLUMNS,
    ]


def test_duplicate_columns(valid_record):
    dataframe = pd.DataFrame([valid_record])

    dataframe[" glucose "] = 150

    result = validate_batch_input(dataframe)

    assert not result.is_file_valid
    assert "Glucose" in result.duplicate_columns


def test_duplicate_records(valid_record):
    dataframe = pd.DataFrame([
        valid_record,
        valid_record,
    ])

    result = validate_batch_input(dataframe)

    assert result.valid_rows == 2
    assert result.invalid_rows == 0
    assert result.warning_rows == 1

    assert "Entire Record" in result.warnings[
        "Field"
    ].tolist()


def test_numeric_strings(valid_record):
    record = {
        feature: str(value)
        for feature, value in valid_record.items()
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.valid_rows == 1
    assert result.invalid_rows == 0


def test_infinite_measurement(valid_record):
    record = {
        **valid_record,
        "BMI": float("inf"),
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.invalid_rows == 1


def test_whitespace_measurement(valid_record):
    record = {
        **valid_record,
        "Glucose": "   ",
    }

    result = validate_batch_input(
        pd.DataFrame([record])
    )

    assert result.invalid_rows == 1

    assert result.errors.iloc[0][
        "Entered Value"
    ] == "(Empty)"


def test_maximum_batch_size(valid_record):
    dataframe = pd.DataFrame(
        [valid_record] * (MAX_BATCH_ROWS + 1)
    )

    result = validate_batch_input(dataframe)

    assert not result.is_file_valid
    assert not result.can_predict


def test_prepare_batch_dataframe(valid_record):
    dataframe = pd.DataFrame([
        valid_record,
        {**valid_record, "Glucose": None},
    ])

    result = validate_batch_input(dataframe)

    prepared = prepare_batch_dataframe(result)

    assert len(prepared) == 1
    assert list(prepared.columns) == FEATURE_COLUMNS
    assert "File Row" not in prepared.columns


def test_prepare_batch_without_valid_rows(valid_record):
    dataframe = pd.DataFrame([
        {**valid_record, "Glucose": None}
    ])

    result = validate_batch_input(dataframe)

    with pytest.raises(ValueError):
        prepare_batch_dataframe(result)


def test_incorrect_column_order(valid_record):
    dataframe = pd.DataFrame([valid_record])

    dataframe = dataframe[
        list(reversed(FEATURE_COLUMNS))
    ]

    result = validate_batch_input(dataframe)

    assert result.valid_rows == 1

    prepared = prepare_batch_dataframe(result)

    assert list(prepared.columns) == FEATURE_COLUMNS
