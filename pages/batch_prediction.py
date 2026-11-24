
from hashlib import sha256

import pandas as pd
import streamlit as st

from components.validation import (
    FEATURE_COLUMNS,
    FEATURE_RULES,
)

from components.batch_validation import (
    MAX_BATCH_ROWS,
    validate_batch_input,
)

from components.batch_files import (
    MAX_FILE_SIZE_MB,
    MAX_BATCH_COLUMNS,
    format_file_size,
    generate_csv_template,
    generate_excel_template,
    read_batch_file,
)


from ml.predict import load_prediction_model
from ml.batch_predict import predict_batch
from components.batch_results import render_batch_results


PREVIEW_ROWS = 10


# Apply page-specific styles.
def apply_batch_styles():
    st.html("""
<style>
.batch-summary-card {
    background: rgba(128, 150, 175, 0.055);
    border: 1px solid rgba(128, 150, 175, 0.22);
    border-radius: 12px;
    padding: 22px;
    min-height: 142px;
    width: 100%;
    max-width: 100%;
    min-width: 0;
    overflow: hidden;
    box-sizing: border-box;
    transition: border-color 0.2s ease;
}

.batch-summary-card:hover {
    border-color: rgba(20, 184, 166, 0.65);
}

.batch-card-label {
    font-size: 14px;
    font-weight: 500;
    opacity: 0.85;
    line-height: 1.4;
    overflow-wrap: anywhere;
}

.batch-card-value {
    font-size: 36px;
    font-weight: 650;
    line-height: 1.2;
    margin-top: 14px;
    letter-spacing: -0.8px;
    font-variant-numeric: tabular-nums;
}

.batch-card-description {
    font-size: 12px;
    opacity: 0.65;
    margin-top: 10px;
    line-height: 1.4;
    overflow-wrap: anywhere;
}

.batch-validation-bar {
    display: flex;
    width: 100%;
    height: 10px;
    border-radius: 6px;
    overflow: hidden;
    background: rgba(128, 150, 175, 0.15);
    margin-top: 12px;
    margin-bottom: 16px;
}

.batch-validation-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 24px;
    font-size: 13px;
}

.batch-legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
}

.batch-legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

.batch-file-status {
    font-size: 23px;
    font-weight: 600;
    margin-top: 5px;
    line-height: 1.3;
}

.batch-file-meta {
    font-size: 13px;
    opacity: 0.7;
    margin-bottom: 4px;
}

@media (max-width: 768px) {
    .batch-summary-card {
        padding: 16px;
        min-height: 135px;
    }

    .batch-card-value {
        font-size: 30px;
    }

    .batch-card-label {
        font-size: 12px;
    }

    .batch-file-status {
        font-size: 18px;
    }
}
</style>
""")


# Initialize the batch workflow state.
def initialize_batch_state():
    defaults = {
        "diaml_batch_file_signature": None,
        "diaml_batch_filename": None,
        "diaml_batch_file_size": None,
        "diaml_batch_dataframe": None,
        "diaml_batch_validation": None,
        "diaml_batch_file_error": None,
        "diaml_batch_confirmed": False,
        "diaml_batch_validation_view": "Errors",
        "diaml_batch_results": None,
        "diaml_batch_prediction_error": None,
        "diaml_batch_saved_id": None,
        "diaml_batch_pending_id": None,
        "diaml_batch_save_result": None,
        "diaml_batch_save_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Reset the workflow when the uploaded file changes.
def clear_batch_state():
    keys = [
        "diaml_batch_file_signature",
        "diaml_batch_filename",
        "diaml_batch_file_size",
        "diaml_batch_dataframe",
        "diaml_batch_validation",
        "diaml_batch_file_error",
        "diaml_batch_results",
        "diaml_batch_prediction_error",
        "diaml_batch_saved_id",
        "diaml_batch_pending_id",
        "diaml_batch_save_result",
        "diaml_batch_save_error",
    ]

    for key in keys:
        st.session_state[key] = None

    st.session_state["diaml_batch_confirmed"] = False
    st.session_state["diaml_batch_validation_view"] = "Errors"

    dashboard_keys = [
        "diaml_batch_analysis_feature",
        "diaml_batch_record_search",
        "diaml_batch_classification_filter",
        "diaml_batch_warning_filter",
        "diaml_batch_sort",
    ]

    for key in dashboard_keys:
        st.session_state.pop(key, None)

    table_version = st.session_state.get(
        "diaml_batch_record_table_version",
        0,
    )

    st.session_state[
        "diaml_batch_record_table_version"
    ] = table_version + 1



# Cache the generated dataset templates.
@st.cache_data(show_spinner=False)
def load_templates():
    return {
        "blank_csv": generate_csv_template(False),
        "sample_csv": generate_csv_template(True),
        "blank_excel": generate_excel_template(False),
        "sample_excel": generate_excel_template(True),
    }


# Build the measurement requirements table.
def build_requirements_table():
    records = []

    for feature in FEATURE_COLUMNS:
        rule = FEATURE_RULES[feature]

        records.append({
            "Column Name": feature,
            "Measurement": rule["label"],
            "Unit": rule["unit"] or "—",
            "Minimum": rule["minimum"],
            "Maximum": rule["maximum"],
            "Data Type": (
                "Whole number"
                if rule["integer"]
                else "Decimal number"
            ),
        })

    return pd.DataFrame(records)


# Render the four downloadable dataset templates.
def render_template_downloads():
    templates = load_templates()

    csv_column, excel_column = st.columns(
        2,
        gap="medium",
    )

    with csv_column:
        with st.container(border=True):
            st.markdown("**CSV Format**")

            st.caption(
                "Comma-separated measurement dataset."
            )

            st.download_button(
                label="Blank Template",
                data=templates["blank_csv"],
                file_name="diaml_blank_template.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
                key="diaml_blank_csv_download",
                on_click="ignore",
            )

            st.download_button(
                label="Sample Dataset",
                data=templates["sample_csv"],
                file_name="diaml_sample_template.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
                key="diaml_sample_csv_download",
                on_click="ignore",
            )

    with excel_column:
        with st.container(border=True):
            st.markdown("**Excel Format**")

            st.caption(
                "Formatted Excel measurement workbook."
            )

            st.download_button(
                label="Blank Template",
                data=templates["blank_excel"],
                file_name="diaml_blank_template.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                icon=":material/download:",
                width="stretch",
                key="diaml_blank_excel_download",
                on_click="ignore",
            )

            st.download_button(
                label="Sample Dataset",
                data=templates["sample_excel"],
                file_name="diaml_sample_template.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                icon=":material/download:",
                width="stretch",
                key="diaml_sample_excel_download",
                on_click="ignore",
            )


# Display the required measurement columns.
def render_measurement_requirements():
    st.caption(
        "All eight measurement columns are required. "
        "These ranges are application input screening "
        "rules, not medical reference ranges."
    )

    st.dataframe(
        build_requirements_table(),
        hide_index=True,
        width="stretch",
        column_config={
            "Minimum": st.column_config.NumberColumn(
                format="%g",
            ),
            "Maximum": st.column_config.NumberColumn(
                format="%g",
            ),
        },
    )


# Display dataset preparation instructions.
def render_dataset_guidelines():
    st.markdown("""
- Use a CSV or Excel (.xlsx) file.
- Place column names in the first row.
- Include all eight required measurement columns.
- Enter one measurement record per row.
- Enter numeric values without measurement units.
- Avoid merged cells and additional title rows.
- Column order and capitalization may differ.
- Additional columns will be ignored.
- Invalid measurements exclude the affected record.
- Unusual but acceptable measurements generate warnings.
- Duplicate records are retained with a warning.
- Only the first Excel worksheet will be processed.
- Use synthetic or de-identified data only.
""")

    st.caption(
        f"Maximum file size: {MAX_FILE_SIZE_MB} MB · "
        f"Maximum records: {MAX_BATCH_ROWS:,} · "
        f"Maximum columns: {MAX_BATCH_COLUMNS}"
    )


# Display templates prominently before an upload.
def render_initial_resources():
    with st.container(border=True):
        st.subheader("Dataset Templates")

        st.caption(
            "Use a ready-made template to ensure your "
            "dataset follows the correct format."
        )

        render_template_downloads()

    with st.expander("Measurement Requirements"):
        render_measurement_requirements()

    with st.expander("Dataset Preparation Guidelines"):
        render_dataset_guidelines()


# Keep preparation resources compact after an upload.
def render_collapsed_resources():
    with st.expander(
        "Templates & Dataset Guidelines"
    ):
        st.markdown("**Dataset Templates**")

        st.caption(
            "Download a blank template or a sample dataset."
        )

        render_template_downloads()

        st.divider()

        st.markdown("**Measurement Requirements**")
        render_measurement_requirements()

        st.divider()

        st.markdown("**Dataset Preparation Guidelines**")
        render_dataset_guidelines()


# Display the primary file upload area.
def render_upload_section():
    with st.container(border=True):
        st.subheader("Upload Dataset")

        st.caption(
            "Upload your measurement dataset to get started."
        )

        uploaded_file = st.file_uploader(
            "Upload measurement dataset",
            type=["csv", "xlsx"],
            accept_multiple_files=False,
            key="diaml_batch_uploader",
            label_visibility="collapsed",
            max_upload_size=MAX_FILE_SIZE_MB,
        )

        left_column, right_column = st.columns(2)

        with left_column:
            st.caption(
                f"Maximum {MAX_BATCH_ROWS:,} records"
            )

        with right_column:
            st.caption(
                "8 required measurements"
            )

    return uploaded_file


# Parse and validate newly uploaded files.
def process_uploaded_file(uploaded_file):
    if uploaded_file is None:
        if (
            st.session_state["diaml_batch_file_signature"]
            is not None
        ):
            clear_batch_state()

        return

    content = uploaded_file.getvalue()

    signature = (
        uploaded_file.name,
        sha256(content).hexdigest(),
    )

    if (
        signature
        == st.session_state["diaml_batch_file_signature"]
    ):
        return

    clear_batch_state()

    st.session_state[
        "diaml_batch_file_signature"
    ] = signature

    st.session_state[
        "diaml_batch_filename"
    ] = uploaded_file.name

    st.session_state[
        "diaml_batch_file_size"
    ] = len(content)

    try:
        with st.spinner(
            "Reading and validating your dataset..."
        ):
            dataframe = read_batch_file(
                uploaded_file.name,
                content,
            )

            validation = validate_batch_input(
                dataframe
            )

        st.session_state[
            "diaml_batch_dataframe"
        ] = dataframe

        st.session_state[
            "diaml_batch_validation"
        ] = validation

        if validation.can_predict and not validation.has_errors:
            if not validation.warnings.empty:
                st.session_state[
                    "diaml_batch_validation_view"
                ] = "Warnings"

            else:
                st.session_state[
                    "diaml_batch_validation_view"
                ] = "Accepted"

    except ValueError as error:
        st.session_state[
            "diaml_batch_file_error"
        ] = str(error)

    except Exception:
        st.session_state[
            "diaml_batch_file_error"
        ] = (
            "An unexpected error occurred while "
            "processing the uploaded dataset."
        )


# Display the uploaded dataset status.
def render_file_information(dataframe, validation):
    filename = st.session_state[
        "diaml_batch_filename"
    ]

    file_size = st.session_state[
        "diaml_batch_file_size"
    ]

    if not validation.is_file_valid:
        status = "Invalid File"
        status_color = "#EF4444"

    elif not validation.can_predict:
        status = "No Valid Records"
        status_color = "#EF4444"

    elif validation.invalid_rows > 0:
        status = "Partially Valid"
        status_color = "#F59E0B"

    elif validation.has_warnings:
        status = "With Warnings"
        status_color = "#F59E0B"

    else:
        status = "Validated"
        status_color = "#14B8A6"

    with st.container(border=True):
        st.markdown("**Uploaded Dataset**")

        st.text(filename)

        file_type = (
            "Excel workbook"
            if filename.lower().endswith(".xlsx")
            else "CSV document"
        )

        st.caption(
            f"{format_file_size(file_size)} · {file_type}"
        )

        st.divider()

        records_column, columns_column, status_column = (
            st.columns(3)
        )

        with records_column:
            st.metric(
                "Records",
                f"{len(dataframe):,}",
            )

        with columns_column:
            st.metric(
                "Columns",
                f"{len(dataframe.columns):,}",
            )

        with status_column:
            st.html(
                f"""
<div class="batch-file-meta">File Status</div>
<div class="batch-file-status"
     style="color:{status_color};">
    {status}
</div>
"""
            )


# Prepare raw uploaded values for table display.
def prepare_display_dataframe(dataframe):
    display = dataframe.copy()

    used_names = set()
    unique_names = []

    for column in display.columns:
        name = str(column) or "Unnamed Column"
        candidate = name
        suffix = 2

        while candidate in used_names:
            candidate = f"{name} ({suffix})"
            suffix += 1

        unique_names.append(candidate)
        used_names.add(candidate)

    display.columns = unique_names

    return display.astype("string").fillna("")


# Display a collapsible preview of the dataset.
def render_dataset_preview(dataframe):
    with st.expander(
        f"Dataset Preview · {len(dataframe):,} records"
    ):
        if dataframe.empty:
            st.info(
                "The dataset contains headers but no records."
            )
            return

        preview = prepare_display_dataframe(
            dataframe.head(PREVIEW_ROWS)
        )

        row_label = "File Row"

        while row_label in preview.columns:
            row_label = f"Original {row_label}"

        preview.insert(
            0,
            row_label,
            range(2, len(preview) + 2),
        )

        st.caption(
            f"Showing the first {len(preview):,} "
            f"of {len(dataframe):,} uploaded records."
        )

        st.dataframe(
            preview,
            hide_index=True,
            width="stretch",
            column_config={
                row_label: st.column_config.NumberColumn(
                    format="%d",
                    width="small",
                    pinned=True,
                ),
            },
        )


# Render a summary card without decorative icons.
def render_summary_card(
    label,
    value,
    description,
    value_color=None,
):
    color_style = (
        f"color:{value_color};"
        if value_color
        else ""
    )

    st.html(
        f"""
<div class="batch-summary-card">
    <div class="batch-card-label">{label}</div>
    <div class="batch-card-value"
         style="{color_style}">
        {value}
    </div>
    <div class="batch-card-description">
        {description}
    </div>
</div>
"""
    )


# Display the six validation summary cards.
def render_validation_cards(validation):
    cards = [
        (
            "Total Records",
            f"{validation.total_rows:,}",
            "Uploaded measurement records",
            None,
        ),
        (
            "Valid Records",
            f"{validation.valid_rows:,}",
            "Eligible for prediction",
            "#14B8A6",
        ),
        (
            "Invalid Records",
            f"{validation.invalid_rows:,}",
            "Excluded from prediction",
            "#EF4444",
        ),
        (
            "Clean Records",
            f"{validation.clean_rows:,}",
            "No validation issues",
            None,
        ),
        (
            "Warning Records",
            f"{validation.warning_rows:,}",
            "Accepted with caution",
            "#F59E0B",
        ),
        (
            "Acceptance Rate",
            f"{validation.success_rate:.1f}%",
            "Percentage of accepted records",
            None,
        ),
    ]

    first_row = st.columns(3, gap="medium")

    for column, card in zip(
        first_row,
        cards[:3],
    ):
        with column:
            render_summary_card(*card)

    second_row = st.columns(3, gap="medium")

    for column, card in zip(
        second_row,
        cards[3:],
    ):
        with column:
            render_summary_card(*card)


# Display the validation outcome distribution.
def render_validation_distribution(validation):
    total = validation.total_rows

    if total == 0:
        return

    clean_rate = (
        validation.clean_rows / total * 100
    )

    warning_rate = (
        validation.warning_rows / total * 100
    )

    invalid_rate = (
        validation.invalid_rows / total * 100
    )

    with st.container(border=True):
        st.markdown("**Validation Distribution**")

        st.html(
            f"""
<div class="batch-validation-bar">
    <div style="
        width:{clean_rate:.4f}%;
        background:#14B8A6;
    "></div>

    <div style="
        width:{warning_rate:.4f}%;
        background:#F59E0B;
    "></div>

    <div style="
        width:{invalid_rate:.4f}%;
        background:#EF4444;
    "></div>
</div>

<div class="batch-validation-legend">
    <div class="batch-legend-item">
        <span class="batch-legend-dot"
              style="background:#14B8A6;"></span>
        Clean: {validation.clean_rows:,}
        ({clean_rate:.1f}%)
    </div>

    <div class="batch-legend-item">
        <span class="batch-legend-dot"
              style="background:#F59E0B;"></span>
        Warnings: {validation.warning_rows:,}
        ({warning_rate:.1f}%)
    </div>

    <div class="batch-legend-item">
        <span class="batch-legend-dot"
              style="background:#EF4444;"></span>
        Invalid: {validation.invalid_rows:,}
        ({invalid_rate:.1f}%)
    </div>
</div>
"""
        )


# Display file-level validation messages.
def render_file_messages(validation):
    for message in validation.file_errors:
        st.error(message)

    for message in validation.file_warnings:
        st.warning(message)


# Display detailed measurement errors.
def render_error_table(validation):
    if validation.errors.empty:
        st.success(
            "No measurement errors detected."
        )
        return

    st.caption(
        f"{validation.invalid_rows:,} records contain "
        f"{len(validation.errors):,} validation errors."
    )

    st.dataframe(
        validation.errors,
        hide_index=True,
        width="stretch",
        height=300,
        column_config={
            "File Row": st.column_config.NumberColumn(
                format="%d",
                width="small",
                pinned=True,
            ),
            "Field": st.column_config.TextColumn(
                width="medium",
            ),
            "Entered Value": st.column_config.TextColumn(
                width="small",
            ),
            "Issue": st.column_config.TextColumn(
                width="large",
            ),
        },
    )


# Display measurement and duplicate record warnings.
def render_warning_table(validation):
    if validation.warnings.empty:
        st.success(
            "No measurement warnings detected."
        )
        return

    st.caption(
        f"{len(validation.warnings):,} warnings detected. "
        "Warnings alone do not exclude records. "
        "Some warnings may belong to excluded records."
    )

    st.dataframe(
        validation.warnings,
        hide_index=True,
        width="stretch",
        height=300,
        column_config={
            "File Row": st.column_config.NumberColumn(
                format="%d",
                width="small",
                pinned=True,
            ),
            "Field": st.column_config.TextColumn(
                width="medium",
            ),
            "Entered Value": st.column_config.TextColumn(
                width="small",
            ),
            "Warning": st.column_config.TextColumn(
                width="large",
            ),
        },
    )


# Display accepted measurement records.
def render_accepted_table(validation):
    if validation.valid_data.empty:
        st.info(
            "No records are eligible for prediction."
        )
        return

    st.caption(
        f"{validation.valid_rows:,} accepted records. "
        "Only these records will be submitted to the "
        "machine learning pipeline."
    )

    st.dataframe(
        validation.valid_data,
        hide_index=True,
        width="stretch",
        height=350,
        column_config={
            "File Row": st.column_config.NumberColumn(
                format="%d",
                width="small",
                pinned=True,
            ),
            "BMI": st.column_config.NumberColumn(
                format="%.2f",
            ),
            "DiabetesPedigreeFunction": (
                st.column_config.NumberColumn(
                    format="%.3f",
                )
            ),
        },
    )


# Display excluded measurement records.
def render_excluded_table(validation):
    if validation.invalid_data.empty:
        st.success(
            "No records have been excluded."
        )
        return

    st.caption(
        f"{validation.invalid_rows:,} excluded records. "
        "Correct these measurements in the original "
        "dataset and upload the updated file."
    )

    excluded_data = prepare_display_dataframe(
        validation.invalid_data
    )

    excluded_data["File Row"] = (
        validation.invalid_data["File Row"]
    )

    excluded_data["Error Count"] = (
        validation.invalid_data["Error Count"]
    )

    st.dataframe(
        excluded_data,
        hide_index=True,
        width="stretch",
        height=350,
        column_config={
            "File Row": st.column_config.NumberColumn(
                format="%d",
                width="small",
                pinned=True,
            ),
            "Error Count": st.column_config.NumberColumn(
                format="%d",
                width="small",
            ),
            "Errors": st.column_config.TextColumn(
                width="large",
            ),
        },
    )


# Display validation details using a segmented control.
def render_validation_details(validation):
    with st.container(border=True):
        st.subheader("Validation Details")

        views = {
            "Errors": len(validation.errors),
            "Warnings": len(validation.warnings),
            "Accepted": validation.valid_rows,
            "Excluded": validation.invalid_rows,
        }

        selected_view = st.segmented_control(
            "Validation view",
            options=list(views.keys()),
            format_func=lambda view: (
                f"{view} ({views[view]:,})"
            ),
            key="diaml_batch_validation_view",
            selection_mode="single",
            label_visibility="collapsed",
            width="stretch",
        )

        selected_view = selected_view or "Errors"

        if selected_view == "Errors":
            render_error_table(validation)

        elif selected_view == "Warnings":
            render_warning_table(validation)

        elif selected_view == "Accepted":
            render_accepted_table(validation)

        elif selected_view == "Excluded":
            render_excluded_table(validation)


# Display the complete validation overview.
def render_validation_overview(validation):
    st.subheader("Validation Overview")

    render_file_messages(validation)

    if not validation.is_file_valid:
        st.info(
            "Correct the dataset structure and upload "
            "the updated file to continue."
        )
        return

    render_validation_cards(validation)

    render_validation_distribution(validation)

    render_validation_details(validation)

    if not validation.can_predict:
        st.error(
            "No valid records are available for prediction. "
            "Correct the measurement errors and upload "
            "the updated dataset."
        )

    elif validation.invalid_rows > 0:
        st.warning(
            f"{validation.valid_rows:,} records are eligible "
            f"for prediction. "
            f"{validation.invalid_rows:,} invalid records "
            "will be excluded."
        )

    elif validation.has_warnings:
        st.info(
            "All records are eligible for prediction. "
            "Review the validation warnings before continuing."
        )

    else:
        st.success(
            "All records passed validation successfully."
        )


# Cache the trained model and configuration.
@st.cache_resource(show_spinner=False)
def load_batch_model():
    return load_prediction_model()


# Execute predictions for all accepted records.
def execute_batch_prediction(validation):
    st.session_state["diaml_batch_results"] = None
    st.session_state["diaml_batch_prediction_error"] = None
    st.session_state["diaml_batch_saved_id"] = None

    if not validation.can_predict:
        st.session_state["diaml_batch_prediction_error"] = (
            "No valid records are available for prediction."
        )
        return

    if not st.session_state["diaml_batch_confirmed"]:
        st.session_state["diaml_batch_prediction_error"] = (
            "Review and confirm the dataset before continuing."
        )
        return

    try:
        with st.spinner(
            "Generating predictions for accepted records..."
        ):
            model, config = load_batch_model()

            result = predict_batch(
                validation,
                model,
                config,
            )

        st.session_state["diaml_batch_results"] = result

    except ValueError as error:
        st.session_state[
            "diaml_batch_prediction_error"
        ] = str(error)

    except Exception:
        st.session_state[
            "diaml_batch_prediction_error"
        ] = (
            "An unexpected error occurred while generating "
            "batch predictions. Please try again."
        )


# Require confirmation before running predictions.
def render_prediction_confirmation(validation):
    if not validation.can_predict:
        return

    with st.container(border=True):
        st.subheader("Ready for Prediction")

        process_column, exclude_column = st.columns(2)

        with process_column:
            st.metric(
                "Records to Process",
                f"{validation.valid_rows:,}",
            )

        with exclude_column:
            st.metric(
                "Records to Exclude",
                f"{validation.invalid_rows:,}",
            )

        if validation.warning_rows > 0:
            st.caption(
                f"{validation.warning_rows:,} accepted records "
                "contain validation warnings."
            )

        st.divider()

        prediction_completed = (
            st.session_state["diaml_batch_results"]
            is not None
        )

        confirmed = st.checkbox(
            "I have reviewed the validation results "
            "and agree to proceed with the accepted records.",
            key="diaml_batch_confirmed",
            disabled=prediction_completed,
        )

        if st.button(
            "Run Batch Prediction",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
            disabled=(
                not confirmed
                or prediction_completed
            ),
            key="diaml_run_batch_prediction",
        ):
            execute_batch_prediction(validation)
            st.rerun()

        prediction_error = st.session_state[
            "diaml_batch_prediction_error"
        ]

        if prediction_error:
            st.error(prediction_error)

        elif prediction_completed:
            st.success(
                "Batch prediction completed successfully."
            )

        elif confirmed:
            st.success(
                "Dataset reviewed and ready for prediction."
            )

        else:
            st.caption(
                "Review and confirm the validation results "
                "to enable batch prediction."
            )


# Initialize the page.
apply_batch_styles()
initialize_batch_state()

st.title("Batch Prediction")

st.caption(
    "Generate diabetes predictions for multiple "
    "measurement records simultaneously."
)

st.divider()

uploaded_file = render_upload_section()

process_uploaded_file(uploaded_file)


# Render the initial upload or validated dataset view.
if uploaded_file is None:
    render_initial_resources()

elif st.session_state["diaml_batch_file_error"]:
    st.error(
        st.session_state["diaml_batch_file_error"]
    )

    render_collapsed_resources()

else:
    dataframe = st.session_state[
        "diaml_batch_dataframe"
    ]

    validation = st.session_state[
        "diaml_batch_validation"
    ]

    if dataframe is not None and validation is not None:
        render_file_information(
            dataframe,
            validation,
        )

        render_dataset_preview(dataframe)

        render_validation_overview(validation)

        render_prediction_confirmation(validation)

        render_batch_results(
            st.session_state["diaml_batch_results"],
            validation,
        )

    render_collapsed_resources()


st.divider()

st.caption(
    "UODA DiaML | Educational Diabetes Prediction System | "
    "Predictions are not a substitute for professional "
    "medical advice or diagnosis."
)
