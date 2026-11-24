
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd

from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter

from components.validation import FEATURE_COLUMNS


HEADER_COLOR = "0F766E"
TITLE_COLOR = "134E4A"
WHITE = "FFFFFF"
TEXT_COLOR = "1E293B"
BORDER_COLOR = "E2E8F0"
ALTERNATE_COLOR = "F8FAFC"

POSITIVE_FILL = "FEE2E2"
POSITIVE_TEXT = "B91C1C"

NEGATIVE_FILL = "CCFBF1"
NEGATIVE_TEXT = "0F766E"

WARNING_FILL = "FEF3C7"
WARNING_TEXT = "92400E"


# Generate a timestamp for exported filenames.
def generate_export_filename(extension):
    extension = extension.lower().lstrip(".")

    if extension not in ("csv", "xlsx"):
        raise ValueError("Unsupported export format.")

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    return f"UODA_DiaML_Batch_{timestamp}.{extension}"


# Prevent spreadsheet formula injection from uploaded text.
def sanitize_export_value(value):
    if isinstance(value, str):
        stripped = value.lstrip()

        if (
            stripped.startswith(("=", "+", "-", "@"))
            or value.startswith(("\t", "\r", "\n"))
        ):
            return "'" + value

    return value


# Sanitize text while preserving numeric measurements.
def sanitize_export_dataframe(dataframe):
    cleaned = dataframe.copy()

    for column in cleaned.columns:
        cleaned[column] = cleaned[column].map(
            sanitize_export_value
        )

    return cleaned


# Prepare the complete prediction dataset for export.
def build_prediction_export(
    result,
    validation,
    batch_id=None,
):
    dataframe = result.results.copy()

    warning_rows = set()

    if not validation.warnings.empty:
        warning_rows = set(
            validation.warnings["File Row"].tolist()
        )

    dataframe["Validation"] = (
        dataframe["File Row"]
        .isin(warning_rows)
        .map({
            True: "Warning",
            False: "Clean",
        })
    )

    dataframe["Model"] = result.model_name
    dataframe["Model Version"] = "v1.0"
    dataframe["Threshold"] = result.threshold

    dataframe["Batch ID"] = (
        str(batch_id) if batch_id else ""
    )

    columns = [
        "File Row",
        *FEATURE_COLUMNS,
        "Classification",
        "Probability",
        "Probability (%)",
        "Validation",
        "Model",
        "Model Version",
        "Threshold",
        "Batch ID",
    ]

    return dataframe.loc[:, columns].copy()


# Build the workbook summary worksheet.
def build_summary_export(
    result,
    validation,
    batch_id=None,
    filename=None,
):
    generated_at = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    records = [
        ("Project", "UODA DiaML"),
        ("Report Type", "Batch Prediction Results"),
        ("Generated At", generated_at),
        ("Source File", filename or "Not specified"),
        ("Batch ID", str(batch_id) if batch_id else "Not saved"),
        ("Database Status", "Saved" if batch_id else "Not saved"),
        ("Model", result.model_name),
        ("Model Version", "v1.0"),
        ("Classification Threshold", result.threshold),
        ("Uploaded Records", validation.total_rows),
        ("Processed Records", result.total_processed),
        ("Positive Predictions", result.positive_count),
        ("Negative Predictions", result.negative_count),
        ("Excluded Records", result.total_excluded),
        ("Warning Records", result.total_warnings),
        ("Clean Records", validation.clean_rows),
        ("Acceptance Rate", validation.success_rate / 100),
        ("Positive Rate", result.positive_rate / 100),
        ("Negative Rate", result.negative_rate / 100),
        ("Average Probability", result.average_probability),
        ("Validation Errors", len(validation.errors)),
        ("Validation Warnings", len(validation.warnings)),
        (
            "Disclaimer",
            "Educational use only. Predictions are not "
            "medical diagnoses and must not be used "
            "for clinical decisions.",
        ),
    ]

    return pd.DataFrame(
        records,
        columns=["Metric", "Value"],
    )


# Generate the downloadable CSV file.
def generate_batch_csv(
    result,
    validation,
    batch_id=None,
):
    dataframe = build_prediction_export(
        result,
        validation,
        batch_id,
    )

    dataframe = sanitize_export_dataframe(
        dataframe
    )

    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# Prepare an empty worksheet with meaningful headers.
def prepare_optional_dataframe(
    dataframe,
    columns,
):
    if dataframe.empty:
        return pd.DataFrame(columns=columns)

    return dataframe.copy()


# Apply the shared workbook title and header styles.
def style_worksheet(
    worksheet,
    title,
    subtitle,
    header_row=4,
):
    last_column = worksheet.max_column

    worksheet.sheet_view.showGridLines = False
    worksheet.sheet_properties.tabColor = HEADER_COLOR

    worksheet.merge_cells(
        start_row=1,
        start_column=1,
        end_row=1,
        end_column=last_column,
    )

    title_cell = worksheet.cell(
        row=1,
        column=1,
        value=title,
    )

    title_cell.font = Font(
        name="Calibri",
        size=18,
        bold=True,
        color=WHITE,
    )

    title_cell.fill = PatternFill(
        "solid",
        fgColor=TITLE_COLOR,
    )

    title_cell.alignment = Alignment(
        vertical="center",
        horizontal="left",
        indent=1,
    )

    worksheet.row_dimensions[1].height = 36

    worksheet.merge_cells(
        start_row=2,
        start_column=1,
        end_row=2,
        end_column=last_column,
    )

    subtitle_cell = worksheet.cell(
        row=2,
        column=1,
        value=subtitle,
    )

    subtitle_cell.font = Font(
        name="Calibri",
        size=11,
        italic=True,
        color="64748B",
    )

    subtitle_cell.alignment = Alignment(
        vertical="center",
        indent=1,
    )

    worksheet.row_dimensions[2].height = 25
    worksheet.row_dimensions[header_row].height = 30

    for cell in worksheet[header_row]:
        cell.fill = PatternFill(
            "solid",
            fgColor=HEADER_COLOR,
        )

        cell.font = Font(
            name="Calibri",
            size=11,
            bold=True,
            color=WHITE,
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    worksheet.freeze_panes = f"A{header_row + 1}"


# Apply alternating rows and readable column widths.
def style_table_body(
    worksheet,
    header_row=4,
):
    bottom_border = Border(
        bottom=Side(
            style="thin",
            color=BORDER_COLOR,
        )
    )

    for row in worksheet.iter_rows(
        min_row=header_row + 1
    ):
        for cell in row:
            cell.font = Font(
                name="Calibri",
                size=11,
                color=TEXT_COLOR,
            )

            cell.alignment = Alignment(
                vertical="center",
                wrap_text=True,
            )

            cell.border = bottom_border

            if cell.row % 2 == 0:
                cell.fill = PatternFill(
                    "solid",
                    fgColor=ALTERNATE_COLOR,
                )

        worksheet.row_dimensions[row[0].row].height = 24

    for column_number in range(
        1,
        worksheet.max_column + 1,
    ):
        header = worksheet.cell(
            row=header_row,
            column=column_number,
        ).value

        maximum_length = len(str(header or ""))

        for row_number in range(
            header_row + 1,
            worksheet.max_row + 1,
        ):
            value = worksheet.cell(
                row=row_number,
                column=column_number,
            ).value

            if value is not None:
                maximum_length = max(
                    maximum_length,
                    len(str(value)),
                )

        width = min(
            max(maximum_length + 4, 15),
            65,
        )

        worksheet.column_dimensions[
            get_column_letter(column_number)
        ].width = width


# Highlight prediction and validation statuses.
def style_prediction_statuses(
    worksheet,
    header_row=4,
):
    headers = {
        cell.value: cell.column
        for cell in worksheet[header_row]
    }

    for row_number in range(
        header_row + 1,
        worksheet.max_row + 1,
    ):
        for field in ("Classification", "Validation"):
            column_number = headers.get(field)

            if column_number is None:
                continue

            cell = worksheet.cell(
                row=row_number,
                column=column_number,
            )

            if cell.value == "Positive":
                fill_color = POSITIVE_FILL
                text_color = POSITIVE_TEXT

            elif cell.value in ("Negative", "Clean"):
                fill_color = NEGATIVE_FILL
                text_color = NEGATIVE_TEXT

            elif cell.value == "Warning":
                fill_color = WARNING_FILL
                text_color = WARNING_TEXT

            else:
                continue

            cell.fill = PatternFill(
                "solid",
                fgColor=fill_color,
            )

            cell.font = Font(
                name="Calibri",
                size=11,
                bold=True,
                color=text_color,
            )

            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
            )

        for field in ("Probability", "Threshold"):
            if field in headers:
                worksheet.cell(
                    row=row_number,
                    column=headers[field],
                ).number_format = "0.00%"

        if "Probability (%)" in headers:
            worksheet.cell(
                row=row_number,
                column=headers["Probability (%)"],
            ).number_format = '0.00"%"'

        if "BMI" in headers:
            worksheet.cell(
                row=row_number,
                column=headers["BMI"],
            ).number_format = "0.00"

        if "DiabetesPedigreeFunction" in headers:
            worksheet.cell(
                row=row_number,
                column=headers["DiabetesPedigreeFunction"],
            ).number_format = "0.000"


# Apply custom formatting to the summary worksheet.
def style_summary_worksheet(worksheet):
    worksheet.column_dimensions["A"].width = 33
    worksheet.column_dimensions["B"].width = 70

    percentage_metrics = {
        "Classification Threshold",
        "Acceptance Rate",
        "Positive Rate",
        "Negative Rate",
        "Average Probability",
    }

    for row_number in range(
        5,
        worksheet.max_row + 1,
    ):
        metric_cell = worksheet.cell(
            row=row_number,
            column=1,
        )

        value_cell = worksheet.cell(
            row=row_number,
            column=2,
        )

        metric_cell.font = Font(
            name="Calibri",
            size=11,
            bold=True,
            color=TEXT_COLOR,
        )

        if metric_cell.value in percentage_metrics:
            value_cell.number_format = "0.00%"

        if metric_cell.value == "Disclaimer":
            worksheet.row_dimensions[
                row_number
            ].height = 65

            value_cell.fill = PatternFill(
                "solid",
                fgColor=WARNING_FILL,
            )

            value_cell.font = Font(
                name="Calibri",
                size=11,
                color=WARNING_TEXT,
            )


# Format the detailed validation worksheets.
def style_validation_worksheet(
    worksheet,
    warning=False,
):
    worksheet.sheet_properties.tabColor = (
        "F59E0B" if warning else "EF4444"
    )

    headers = {
        cell.value: cell.column
        for cell in worksheet[4]
    }

    for field in ("Issue", "Warning", "Errors"):
        column_number = headers.get(field)

        if column_number is None:
            continue

        worksheet.column_dimensions[
            get_column_letter(column_number)
        ].width = 70

        for row_number in range(
            5,
            worksheet.max_row + 1,
        ):
            cell = worksheet.cell(
                row=row_number,
                column=column_number,
            )

            cell.fill = PatternFill(
                "solid",
                fgColor=(
                    WARNING_FILL
                    if warning
                    else POSITIVE_FILL
                ),
            )

            cell.font = Font(
                name="Calibri",
                size=11,
                color=(
                    WARNING_TEXT
                    if warning
                    else POSITIVE_TEXT
                ),
            )

            text_length = len(str(cell.value or ""))

            worksheet.row_dimensions[
                row_number
            ].height = max(
                32,
                min(
                    180,
                    16 * (text_length // 65 + 1),
                ),
            )


# Generate the complete formatted Excel workbook.
def generate_batch_excel(
    result,
    validation,
    batch_id=None,
    filename=None,
):
    summary = build_summary_export(
        result,
        validation,
        batch_id,
        filename,
    )

    predictions = build_prediction_export(
        result,
        validation,
        batch_id,
    )

    excluded = prepare_optional_dataframe(
        validation.invalid_data,
        [
            "File Row",
            *FEATURE_COLUMNS,
            "Error Count",
            "Errors",
        ],
    )

    errors = prepare_optional_dataframe(
        validation.errors,
        [
            "File Row",
            "Field",
            "Entered Value",
            "Issue",
        ],
    )

    warnings = prepare_optional_dataframe(
        validation.warnings,
        [
            "File Row",
            "Field",
            "Entered Value",
            "Warning",
        ],
    )

    worksheets = [
        (
            "Summary",
            "Batch Prediction Summary",
            summary,
        ),
        (
            "Predictions",
            "Accepted Prediction Records",
            predictions,
        ),
        (
            "Excluded Records",
            "Excluded Measurement Records",
            excluded,
        ),
        (
            "Validation Errors",
            "Detailed Validation Errors",
            errors,
        ),
        (
            "Validation Warnings",
            "Detailed Validation Warnings",
            warnings,
        ),
    ]

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        for sheet_name, title, dataframe in worksheets:
            safe_dataframe = sanitize_export_dataframe(
                dataframe
            )

            safe_dataframe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=3,
            )

            worksheet = writer.sheets[sheet_name]

            style_worksheet(
                worksheet,
                title=f"UODA DiaML | {title}",
                subtitle=(
                    "Educational Diabetes Prediction System"
                ),
            )

            style_table_body(worksheet)

            if sheet_name != "Summary":
                last_column = get_column_letter(
                    worksheet.max_column
                )

                worksheet.auto_filter.ref = (
                    f"A4:{last_column}{worksheet.max_row}"
                )

            if sheet_name == "Summary":
                style_summary_worksheet(worksheet)

            elif sheet_name == "Predictions":
                style_prediction_statuses(worksheet)

            elif sheet_name in (
                "Excluded Records",
                "Validation Errors",
                "Validation Warnings",
            ):
                style_validation_worksheet(
                    worksheet,
                    warning=(
                        sheet_name == "Validation Warnings"
                    ),
                )

    return output.getvalue()
