
from dataclasses import dataclass, field

import pandas as pd

from components.validation import (
    FEATURE_COLUMNS,
    is_missing_value,
    validate_prediction_input,
)


MAX_BATCH_ROWS = 5000

ROW_NUMBER_COLUMN = "File Row"

ERROR_COLUMNS = [
    ROW_NUMBER_COLUMN,
    "Field",
    "Entered Value",
    "Issue",
]

WARNING_COLUMNS = [
    ROW_NUMBER_COLUMN,
    "Field",
    "Entered Value",
    "Warning",
]


# Store all batch validation results.
@dataclass
class BatchValidationResult:
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    warning_rows: int = 0
    clean_rows: int = 0

    valid_data: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[ROW_NUMBER_COLUMN, *FEATURE_COLUMNS]
        )
    )

    invalid_data: pd.DataFrame = field(
        default_factory=pd.DataFrame
    )

    errors: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=ERROR_COLUMNS
        )
    )

    warnings: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=WARNING_COLUMNS
        )
    )

    file_errors: list = field(default_factory=list)
    file_warnings: list = field(default_factory=list)

    missing_columns: list = field(default_factory=list)
    extra_columns: list = field(default_factory=list)
    duplicate_columns: list = field(default_factory=list)

    @property
    def is_file_valid(self):
        return not self.file_errors

    @property
    def can_predict(self):
        return self.is_file_valid and self.valid_rows > 0

    @property
    def has_errors(self):
        return bool(self.file_errors) or not self.errors.empty

    @property
    def has_warnings(self):
        return (
            bool(self.file_warnings)
            or not self.warnings.empty
        )

    @property
    def success_rate(self):
        if self.total_rows == 0:
            return 0.0

        return self.valid_rows / self.total_rows * 100

    @property
    def status(self):
        if not self.is_file_valid:
            return "invalid_file"

        if self.valid_rows == 0:
            return "no_valid_rows"

        if self.invalid_rows > 0:
            return "partially_valid"

        if self.has_warnings:
            return "valid_with_warnings"

        return "valid"

    # Return statistics for the batch interface.
    def get_summary(self):
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "warning_rows": self.warning_rows,
            "clean_rows": self.clean_rows,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "success_rate": self.success_rate,
            "status": self.status,
        }


# Normalize column names to match the trained model.
def normalize_column_name(column):
    column = str(column).strip().lstrip("\ufeff").strip()

    feature_lookup = {
        feature.casefold(): feature
        for feature in FEATURE_COLUMNS
    }

    return feature_lookup.get(
        column.casefold(),
        column,
    )


# Format submitted values for validation reports.
def format_entered_value(value):
    if is_missing_value(value):
        return "(Empty)"

    return str(value)


# Validate the overall structure of an uploaded dataset.
def validate_batch_structure(dataframe, result):
    if not isinstance(dataframe, pd.DataFrame):
        result.file_errors.append(
            "The uploaded dataset must be a pandas DataFrame."
        )
        return None

    result.total_rows = len(dataframe)

    if dataframe.empty:
        result.file_errors.append(
            "The uploaded dataset contains no records."
        )
        return None

    if result.total_rows > MAX_BATCH_ROWS:
        result.file_errors.append(
            f"The dataset contains {result.total_rows:,} records. "
            f"A maximum of {MAX_BATCH_ROWS:,} records is allowed."
        )
        return None

    dataframe = dataframe.copy()

    original_columns = list(dataframe.columns)

    normalized_columns = [
        normalize_column_name(column)
        for column in original_columns
    ]

    dataframe.columns = normalized_columns

    # Detect duplicate columns after normalization.
    duplicated = dataframe.columns[
        dataframe.columns.duplicated(keep=False)
    ]

    result.duplicate_columns = list(
        dict.fromkeys(duplicated.tolist())
    )

    if result.duplicate_columns:
        result.file_errors.append(
            "Duplicate column names detected: "
            + ", ".join(result.duplicate_columns)
        )

    # Identify missing required measurements.
    result.missing_columns = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in dataframe.columns
    ]

    if result.missing_columns:
        result.file_errors.append(
            "Missing required columns: "
            + ", ".join(result.missing_columns)
        )

    # Identify columns that are not used by the model.
    result.extra_columns = [
        column
        for column in dataframe.columns
        if column not in FEATURE_COLUMNS
    ]

    if result.extra_columns:
        result.file_warnings.append(
            "The following extra columns will be ignored: "
            + ", ".join(result.extra_columns)
        )

    changed_columns = [
        f"{original!r} → {normalized!r}"
        for original, normalized in zip(
            original_columns,
            normalized_columns,
        )
        if original != normalized
    ]

    if changed_columns:
        result.file_warnings.append(
            "Column names were automatically normalized: "
            + ", ".join(changed_columns)
        )

    if result.file_errors:
        return None

    # Keep only the required features in the correct order.
    return dataframe.loc[:, FEATURE_COLUMNS].copy()


# Validate every measurement in every uploaded record.
def validate_batch_input(dataframe):
    result = BatchValidationResult()

    prepared_data = validate_batch_structure(
        dataframe,
        result,
    )

    if prepared_data is None:
        return result

    valid_records = []
    invalid_records = []
    error_records = []
    warning_records = []

    seen_records = {}

    # Start at row 2 because row 1 contains column headers.
    for row_number, (_, row) in enumerate(
        prepared_data.iterrows(),
        start=2,
    ):
        measurements = {
            feature: row[feature]
            for feature in FEATURE_COLUMNS
        }

        validation = validate_prediction_input(
            measurements
        )

        row_warnings = []

        # Collect every invalid measurement in the record.
        for feature, message in validation.errors.items():
            error_records.append({
                ROW_NUMBER_COLUMN: row_number,
                "Field": feature,
                "Entered Value": format_entered_value(
                    measurements.get(feature)
                ),
                "Issue": message,
            })

        # Collect unusual but acceptable measurements.
        for feature, message in validation.warnings.items():
            row_warnings.append({
                ROW_NUMBER_COLUMN: row_number,
                "Field": feature,
                "Entered Value": format_entered_value(
                    measurements.get(feature)
                ),
                "Warning": message,
            })

        if not validation.is_valid:
            invalid_records.append({
                ROW_NUMBER_COLUMN: row_number,
                **measurements,
                "Error Count": len(validation.errors),
                "Errors": "; ".join(
                    validation.errors.values()
                ),
            })

            warning_records.extend(row_warnings)
            continue

        cleaned_data = validation.cleaned_data

        # Detect identical records after numeric conversion.
        record_signature = tuple(
            cleaned_data[feature]
            for feature in FEATURE_COLUMNS
        )

        if record_signature in seen_records:
            original_row = seen_records[record_signature]

            row_warnings.append({
                ROW_NUMBER_COLUMN: row_number,
                "Field": "Entire Record",
                "Entered Value": "Duplicate measurements",
                "Warning": (
                    f"This record contains the same measurements "
                    f"as file row {original_row}. "
                    "The record has been retained."
                ),
            })

        else:
            seen_records[record_signature] = row_number

        # Count accepted records with and without warnings.
        if row_warnings:
            result.warning_rows += 1
        else:
            result.clean_rows += 1

        warning_records.extend(row_warnings)

        valid_records.append({
            ROW_NUMBER_COLUMN: row_number,
            **cleaned_data,
        })

    # Convert collected records into reusable DataFrames.
    result.valid_data = pd.DataFrame(
        valid_records,
        columns=[ROW_NUMBER_COLUMN, *FEATURE_COLUMNS],
    )

    result.invalid_data = pd.DataFrame(
        invalid_records,
        columns=[
            ROW_NUMBER_COLUMN,
            *FEATURE_COLUMNS,
            "Error Count",
            "Errors",
        ],
    )

    result.errors = pd.DataFrame(
        error_records,
        columns=ERROR_COLUMNS,
    )

    result.warnings = pd.DataFrame(
        warning_records,
        columns=WARNING_COLUMNS,
    )

    result.valid_rows = len(valid_records)
    result.invalid_rows = len(invalid_records)

    return result


# Prepare only accepted measurements for the ML pipeline.
def prepare_batch_dataframe(validation_result):
    if not isinstance(
        validation_result,
        BatchValidationResult,
    ):
        raise TypeError(
            "Expected a BatchValidationResult object."
        )

    if not validation_result.can_predict:
        raise ValueError(
            "No valid batch records are available for prediction."
        )

    return validation_result.valid_data.loc[
        :,
        FEATURE_COLUMNS,
    ].copy()
