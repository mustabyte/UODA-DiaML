
import csv
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

from components.validation import FEATURE_COLUMNS
from components.batch_validation import MAX_BATCH_ROWS


MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_EXCEL_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_BATCH_COLUMNS = 100

SUPPORTED_EXTENSIONS = [".csv", ".xlsx"]

SAMPLE_RECORDS = [
    [2, 145, 78, 25, 110, 29.5, 0.45, 35],
    [1, 95, 70, 20, 80, 23.5, 0.25, 25],
    [3, 130, 82, 30, 125, 31.2, 0.62, 42],
    [0, 110, 72, 22, 90, 24.8, 0.31, 28],
    [5, 180, 88, 35, 210, 36.4, 0.85, 52],
]


# Create blank or sample measurement templates.
def create_template_dataframe(include_samples=False):
    return pd.DataFrame(
        SAMPLE_RECORDS if include_samples else [],
        columns=FEATURE_COLUMNS,
    )


# Generate a downloadable CSV template.
def generate_csv_template(include_samples=False):
    dataframe = create_template_dataframe(include_samples)

    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# Generate a formatted Excel template.
def generate_excel_template(include_samples=False):
    dataframe = create_template_dataframe(include_samples)

    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        dataframe.to_excel(
            writer,
            sheet_name="Measurements",
            index=False,
        )

        worksheet = writer.sheets["Measurements"]

        for cell in worksheet[1]:
            cell.font = Font(
                bold=True,
                color="FFFFFF",
            )

            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="0F766E",
            )

            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
            )

            worksheet.column_dimensions[
                cell.column_letter
            ].width = max(
                len(str(cell.value)) + 5,
                16,
            )

        worksheet.row_dimensions[1].height = 28
        worksheet.freeze_panes = "A2"

    return buffer.getvalue()


# Format uploaded file sizes.
def format_file_size(size):
    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"

    return f"{size / (1024 * 1024):.2f} MB"


# Validate file format and size.
def validate_uploaded_file(filename, content):
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file format. Upload a CSV "
            "or Excel (.xlsx) file."
        )

    if not content:
        raise ValueError(
            "The uploaded file is empty."
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"The uploaded file exceeds the "
            f"{MAX_FILE_SIZE_MB} MB size limit."
        )

    return extension


# Preserve original headers and submitted values.
def build_dataframe(headers, records):
    if not headers:
        raise ValueError(
            "The dataset does not contain column headers."
        )

    if len(headers) > MAX_BATCH_COLUMNS:
        raise ValueError(
            f"A maximum of {MAX_BATCH_COLUMNS} "
            "columns is allowed."
        )

    headers = [
        str(header) if header is not None else ""
        for header in headers
    ]

    if any(not header.strip() for header in headers):
        raise ValueError(
            "Every dataset column must have a name."
        )

    return pd.DataFrame(
        records,
        columns=headers,
        dtype=object,
    )


# Read CSV files without renaming duplicate headers.
def read_csv_file(content):
    try:
        decoded = content.decode("utf-8-sig")

    except UnicodeDecodeError:
        raise ValueError(
            "Unable to decode the CSV file. "
            "Save it using UTF-8 encoding."
        )

    reader = csv.reader(
        StringIO(decoded, newline=""),
        strict=True,
    )

    try:
        headers = next(reader, None)

        if not headers:
            raise ValueError(
                "The CSV file does not contain column headers."
            )

        column_count = len(headers)

        if column_count > MAX_BATCH_COLUMNS:
            raise ValueError(
                f"A maximum of {MAX_BATCH_COLUMNS} "
                "columns is allowed."
            )

        records = []

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if len(records) >= MAX_BATCH_ROWS:
                raise ValueError(
                    f"A maximum of {MAX_BATCH_ROWS:,} "
                    "records is allowed."
                )

            if len(row) > column_count:
                raise ValueError(
                    f"File row {row_number} contains more "
                    "values than the header defines."
                )

            if len(row) < column_count:
                row.extend(
                    [None] * (column_count - len(row))
                )

            records.append(row)

    except csv.Error as error:
        raise ValueError(
            f"Invalid CSV structure: {error}"
        )

    return build_dataframe(headers, records)


# Check the expanded size of an Excel workbook.
def validate_excel_archive(content):
    with ZipFile(BytesIO(content)) as archive:
        total_size = sum(
            item.file_size
            for item in archive.infolist()
        )

        if total_size > MAX_EXCEL_UNCOMPRESSED_BYTES:
            raise ValueError(
                "The expanded Excel file is too large "
                "to process safely."
            )


# Read the first Excel worksheet.
def read_excel_file(content):
    workbook = None

    try:
        validate_excel_archive(content)

        workbook = load_workbook(
            BytesIO(content),
            read_only=True,
            data_only=True,
        )

        if not workbook.worksheets:
            raise ValueError(
                "The Excel file contains no worksheets."
            )

        worksheet = workbook.worksheets[0]

        if (
            worksheet.max_column is not None
            and worksheet.max_column > MAX_BATCH_COLUMNS
        ):
            raise ValueError(
                f"A maximum of {MAX_BATCH_COLUMNS} "
                "columns is allowed."
            )

        if (
            worksheet.max_row is not None
            and worksheet.max_row > MAX_BATCH_ROWS + 1
        ):
            raise ValueError(
                f"A maximum of {MAX_BATCH_ROWS:,} "
                "records is allowed."
            )

        rows = worksheet.iter_rows(
            max_col=MAX_BATCH_COLUMNS + 1,
            values_only=True,
        )

        headers = next(rows, None)

        if not headers:
            raise ValueError(
                "The Excel worksheet contains no headers."
            )

        headers = list(headers)

        while headers and headers[-1] is None:
            headers.pop()

        if not headers:
            raise ValueError(
                "The Excel worksheet contains no headers."
            )

        column_count = len(headers)

        if column_count > MAX_BATCH_COLUMNS:
            raise ValueError(
                f"A maximum of {MAX_BATCH_COLUMNS} "
                "columns is allowed."
            )

        records = []

        for row_number, row in enumerate(
            rows,
            start=2,
        ):
            if len(records) >= MAX_BATCH_ROWS:
                raise ValueError(
                    f"A maximum of {MAX_BATCH_ROWS:,} "
                    "records is allowed."
                )

            row = list(row)

            extra_values = row[column_count:]

            if any(
                value is not None
                for value in extra_values
            ):
                raise ValueError(
                    f"File row {row_number} contains values "
                    "outside the defined header columns."
                )

            row = row[:column_count]

            if len(row) < column_count:
                row.extend(
                    [None] * (column_count - len(row))
                )

            records.append(row)

        return build_dataframe(
            headers,
            records,
        )

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            "Unable to read the Excel file. Make sure "
            "it is a valid, unencrypted .xlsx workbook."
        ) from error

    finally:
        if workbook is not None:
            workbook.close()


# Select the appropriate file parser.
def read_batch_file(filename, content):
    extension = validate_uploaded_file(
        filename,
        content,
    )

    if extension == ".csv":
        return read_csv_file(content)

    return read_excel_file(content)
