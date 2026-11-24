
from io import BytesIO

import pandas as pd
import pytest

from components.validation import FEATURE_COLUMNS

from components.batch_validation import (
    MAX_BATCH_ROWS,
    validate_batch_input,
)

from components.batch_files import (
    MAX_FILE_SIZE_BYTES,
    MAX_BATCH_COLUMNS,
    SAMPLE_RECORDS,
    create_template_dataframe,
    generate_csv_template,
    generate_excel_template,
    read_batch_file,
    format_file_size,
)


def test_blank_template():
    dataframe = create_template_dataframe()

    assert dataframe.empty
    assert list(dataframe.columns) == FEATURE_COLUMNS


def test_sample_template():
    dataframe = create_template_dataframe(True)

    assert len(dataframe) == len(SAMPLE_RECORDS)
    assert list(dataframe.columns) == FEATURE_COLUMNS


def test_blank_csv_template():
    dataframe = read_batch_file(
        "blank.csv",
        generate_csv_template(),
    )

    assert dataframe.empty
    assert list(dataframe.columns) == FEATURE_COLUMNS


def test_sample_csv_template():
    dataframe = read_batch_file(
        "sample.csv",
        generate_csv_template(True),
    )

    validation = validate_batch_input(dataframe)

    assert validation.is_file_valid
    assert validation.valid_rows == len(SAMPLE_RECORDS)
    assert validation.invalid_rows == 0


def test_blank_excel_template():
    dataframe = read_batch_file(
        "blank.xlsx",
        generate_excel_template(),
    )

    assert dataframe.empty
    assert list(dataframe.columns) == FEATURE_COLUMNS


def test_sample_excel_template():
    dataframe = read_batch_file(
        "sample.xlsx",
        generate_excel_template(True),
    )

    validation = validate_batch_input(dataframe)

    assert validation.is_file_valid
    assert validation.valid_rows == len(SAMPLE_RECORDS)
    assert validation.invalid_rows == 0


def test_unsupported_file_type():
    with pytest.raises(
        ValueError,
        match="Unsupported file format",
    ):
        read_batch_file(
            "sample.txt",
            b"sample content",
        )


def test_empty_file():
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        read_batch_file(
            "sample.csv",
            b"",
        )


def test_file_size_limit():
    content = b"x" * (MAX_FILE_SIZE_BYTES + 1)

    with pytest.raises(
        ValueError,
        match="size limit",
    ):
        read_batch_file(
            "large.csv",
            content,
        )


def test_duplicate_csv_headers():
    headers = ",".join(
        [*FEATURE_COLUMNS, "Glucose"]
    )

    values = ",".join(
        map(str, [*SAMPLE_RECORDS[0], 150])
    )

    content = (
        f"{headers}\n{values}\n"
    ).encode("utf-8")

    dataframe = read_batch_file(
        "duplicate.csv",
        content,
    )

    validation = validate_batch_input(dataframe)

    assert not validation.is_file_valid
    assert "Glucose" in validation.duplicate_columns


def test_duplicate_excel_headers():
    dataframe = pd.DataFrame(
        [[*SAMPLE_RECORDS[0], 150]],
        columns=[*FEATURE_COLUMNS, "Glucose"],
    )

    buffer = BytesIO()

    dataframe.to_excel(
        buffer,
        index=False,
        engine="openpyxl",
    )

    parsed = read_batch_file(
        "duplicate.xlsx",
        buffer.getvalue(),
    )

    validation = validate_batch_input(parsed)

    assert not validation.is_file_valid
    assert "Glucose" in validation.duplicate_columns


def test_blank_csv_record():
    headers = ",".join(FEATURE_COLUMNS)

    first = ",".join(
        map(str, SAMPLE_RECORDS[0])
    )

    second = ",".join(
        map(str, SAMPLE_RECORDS[1])
    )

    content = (
        f"{headers}\n{first}\n\n{second}\n"
    ).encode("utf-8")

    dataframe = read_batch_file(
        "blank_row.csv",
        content,
    )

    validation = validate_batch_input(dataframe)

    assert validation.total_rows == 3
    assert validation.valid_rows == 2
    assert validation.invalid_rows == 1

    assert validation.valid_data[
        "File Row"
    ].tolist() == [2, 4]


def test_blank_excel_record():
    dataframe = pd.DataFrame(
        [
            SAMPLE_RECORDS[0],
            [None] * len(FEATURE_COLUMNS),
            SAMPLE_RECORDS[1],
        ],
        columns=FEATURE_COLUMNS,
    )

    buffer = BytesIO()

    dataframe.to_excel(
        buffer,
        index=False,
        engine="openpyxl",
    )

    parsed = read_batch_file(
        "blank_row.xlsx",
        buffer.getvalue(),
    )

    validation = validate_batch_input(parsed)

    assert validation.total_rows == 3
    assert validation.invalid_rows == 1

    assert validation.valid_data[
        "File Row"
    ].tolist() == [2, 4]


def test_incomplete_csv_record():
    headers = ",".join(FEATURE_COLUMNS)

    values = ",".join(
        map(str, SAMPLE_RECORDS[0][:-1])
    )

    content = (
        f"{headers}\n{values}\n"
    ).encode("utf-8")

    dataframe = read_batch_file(
        "incomplete.csv",
        content,
    )

    validation = validate_batch_input(dataframe)

    assert validation.invalid_rows == 1
    assert "Age" in validation.errors["Field"].tolist()


def test_extra_csv_values():
    headers = ",".join(FEATURE_COLUMNS)

    values = ",".join(
        map(str, [*SAMPLE_RECORDS[0], 999])
    )

    content = (
        f"{headers}\n{values}\n"
    ).encode("utf-8")

    with pytest.raises(
        ValueError,
        match="more values",
    ):
        read_batch_file(
            "extra_values.csv",
            content,
        )


def test_csv_row_limit():
    headers = ",".join(FEATURE_COLUMNS)

    values = ",".join(
        map(str, SAMPLE_RECORDS[0])
    )

    content = (
        headers
        + "\n"
        + (values + "\n") * (MAX_BATCH_ROWS + 1)
    ).encode("utf-8")

    with pytest.raises(
        ValueError,
        match="maximum",
    ):
        read_batch_file(
            "large_dataset.csv",
            content,
        )


def test_excel_row_limit():
    dataframe = pd.DataFrame(
        [SAMPLE_RECORDS[0]] * (MAX_BATCH_ROWS + 1),
        columns=FEATURE_COLUMNS,
    )

    buffer = BytesIO()

    dataframe.to_excel(
        buffer,
        index=False,
        engine="openpyxl",
    )

    with pytest.raises(
        ValueError,
        match="maximum",
    ):
        read_batch_file(
            "large_dataset.xlsx",
            buffer.getvalue(),
        )


def test_column_limit():
    headers = ",".join(
        f"Column{i}"
        for i in range(MAX_BATCH_COLUMNS + 1)
    )

    content = (
        headers + "\n"
    ).encode("utf-8")

    with pytest.raises(
        ValueError,
        match="columns",
    ):
        read_batch_file(
            "wide_dataset.csv",
            content,
        )


def test_unnamed_column():
    content = (
        "Pregnancies,Glucose,,Age\n"
        "2,145,29.5,35\n"
    ).encode("utf-8")

    with pytest.raises(
        ValueError,
        match="column must have a name",
    ):
        read_batch_file(
            "unnamed_column.csv",
            content,
        )


def test_invalid_excel_file():
    with pytest.raises(
        ValueError,
        match="Unable to read",
    ):
        read_batch_file(
            "invalid.xlsx",
            b"not an excel file",
        )


def test_invalid_csv_encoding():
    with pytest.raises(
        ValueError,
        match="UTF-8",
    ):
        read_batch_file(
            "invalid_encoding.csv",
            b"\xff\xfe\x80",
        )


def test_excel_first_worksheet():
    buffer = BytesIO()

    first = create_template_dataframe(True)
    second = pd.DataFrame({"Unused": [1, 2]})

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        first.to_excel(
            writer,
            sheet_name="Measurements",
            index=False,
        )

        second.to_excel(
            writer,
            sheet_name="Other Data",
            index=False,
        )

    dataframe = read_batch_file(
        "multiple_sheets.xlsx",
        buffer.getvalue(),
    )

    assert list(dataframe.columns) == FEATURE_COLUMNS
    assert len(dataframe) == len(SAMPLE_RECORDS)


def test_file_size_formatting():
    assert format_file_size(500) == "500 B"
    assert format_file_size(1024) == "1.0 KB"

    assert format_file_size(
        1024 * 1024
    ) == "1.00 MB"
