from uuid import uuid4

import streamlit as st

from components.batch_exports import (
    generate_batch_csv,
    generate_batch_excel,
    generate_export_filename,
)

from components.report_generator import (
    generate_batch_report,
    generate_report_filename,
)

from components.validation import FEATURE_COLUMNS, FEATURE_RULES
from database.repository import save_batch_predictions


# Convert prediction results into database-ready records.
def prepare_batch_records(result):
    records = []

    for _, row in result.results.iterrows():
        measurements = {}

        for feature in FEATURE_COLUMNS:
            value = row[feature]

            if FEATURE_RULES[feature]["integer"]:
                measurements[feature] = int(value)
            else:
                measurements[feature] = float(value)

        records.append({
            "measurements": measurements,
            "probability": float(row["Probability"]),
            "classification": str(row["Classification"]),
            "batch_row_number": int(row["File Row"]),
        })

    if not records:
        raise ValueError(
            "No prediction records are available to save."
        )

    return records


# Initialize the saving state for the current batch.
def initialize_batch_save_state():
    defaults = {
        "diaml_batch_pending_id": None,
        "diaml_batch_saved_id": None,
        "diaml_batch_save_result": None,
        "diaml_batch_save_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Save all predicted records using a single batch ID.
def execute_batch_save(result):
    initialize_batch_save_state()

    if st.session_state["diaml_batch_saved_id"] is not None:
        return

    st.session_state["diaml_batch_save_error"] = None

    if st.session_state["diaml_batch_pending_id"] is None:
        st.session_state["diaml_batch_pending_id"] = str(uuid4())

    batch_id = st.session_state["diaml_batch_pending_id"]

    try:
        records = prepare_batch_records(result)

        with st.spinner("Saving batch to Prediction History..."):
            save_result = save_batch_predictions(
                predictions=records,
                batch_id=batch_id,
                threshold=result.threshold,
                model_name=result.model_name,
                model_version="v1.0",
            )

        st.session_state["diaml_batch_saved_id"] = (
            save_result["batch_id"]
        )

        st.session_state["diaml_batch_save_result"] = (
            save_result
        )

    except ValueError as error:
        st.session_state["diaml_batch_save_error"] = str(error)

    except Exception:
        st.session_state["diaml_batch_save_error"] = (
            "Unable to save the batch to the database. "
            "Please check your connection and try again."
        )


# Display the successful batch saving information.
def render_saved_batch_information():
    save_result = st.session_state["diaml_batch_save_result"]

    if save_result is None:
        return

    if save_result["already_saved"]:
        st.success(
            "This batch has already been saved. "
            "No duplicate records were created."
        )

    else:
        st.success(
            f"Successfully saved "
            f"{save_result['saved_count']:,} prediction records."
        )

    st.caption("Batch ID")

    st.code(
        save_result["batch_id"],
        language=None,
    )

    first, second = st.columns(2)

    with first:
        st.metric(
            "Saved Records",
            f"{save_result['total_records']:,}",
        )

    with second:
        st.metric(
            "Status",
            "Saved",
        )

    st.page_link(
        "pages/history.py",
        label="View Prediction History",
        icon=":material/history:",
        width="stretch",
    )


# Render the batch saving controls.
def render_batch_save(result):
    initialize_batch_save_state()

    is_saved = (
        st.session_state["diaml_batch_saved_id"]
        is not None
    )

    with st.container(border=True):
        st.markdown("**Save to Prediction History**")

        if is_saved:
            st.caption(
                "This batch is available in Prediction History."
            )

        else:
            st.caption(
                f"{result.total_processed:,} prediction records "
                "are ready to save."
            )

        if st.button(
            "Batch Saved" if is_saved else "Save Batch",
            type="primary",
            icon=(
                ":material/check_circle:"
                if is_saved
                else ":material/database:"
            ),
            width="stretch",
            disabled=is_saved,
            key="diaml_batch_save_button",
        ):
            execute_batch_save(result)
            st.rerun()

        save_error = st.session_state["diaml_batch_save_error"]

        if save_error:
            st.error(save_error)

        if is_saved:
            render_saved_batch_information()


# Cache export files to avoid repeated generation.
@st.cache_data(show_spinner=False, max_entries=8, ttl=600)
def prepare_export_files(
    result,
    validation,
    batch_id,
    filename,
):
    return {
        "csv": generate_batch_csv(
            result,
            validation,
            batch_id,
        ),
        "excel": generate_batch_excel(
            result,
            validation,
            batch_id,
            filename,
        ),
        "csv_filename": generate_export_filename("csv"),
        "excel_filename": generate_export_filename("xlsx"),
    }


# Render downloadable batch prediction exports.
def render_batch_exports(result, validation):
    with st.container(border=True):
        st.markdown("**Export Results**")

        st.caption(
            "Download the complete batch results. "
            "Saving to Prediction History is not required."
        )

        batch_id = st.session_state.get(
            "diaml_batch_saved_id"
        )

        filename = st.session_state.get(
            "diaml_batch_filename"
        )

        try:
            with st.spinner("Preparing export files..."):
                exports = prepare_export_files(
                    result,
                    validation,
                    batch_id,
                    filename,
                )

        except Exception:
            st.error(
                "Unable to prepare the export files. "
                "Please try again."
            )
            return

        csv_column, excel_column, pdf_column = st.columns(3)

        with csv_column:
            st.markdown("**CSV**")

            st.caption(
                "All accepted prediction records."
            )

            st.download_button(
                "Download CSV",
                data=exports["csv"],
                file_name=exports["csv_filename"],
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
                on_click="ignore",
                key="diaml_batch_csv_download",
            )

        with excel_column:
            st.markdown("**Excel**")

            st.caption(
                "Complete formatted workbook."
            )

            st.download_button(
                "Download Excel",
                data=exports["excel"],
                file_name=exports["excel_filename"],
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                icon=":material/download:",
                width="stretch",
                on_click="ignore",
                key="diaml_batch_excel_download",
            )

        with pdf_column:
            st.markdown("**PDF**")

            st.caption(
                "Professional analytical report."
            )

            try:
                with st.spinner("Preparing PDF report..."):
                    pdf_data = generate_batch_report(
                        result=result,
                        validation=validation,
                        batch_id=batch_id,
                        filename=filename,
                    )

                if (
                    not isinstance(pdf_data, bytes)
                    or not pdf_data.startswith(b"%PDF-")
                ):
                    raise ValueError(
                        "Invalid PDF report generated."
                    )

                st.download_button(
                    "Download PDF",
                    data=pdf_data,
                    file_name=generate_report_filename(
                        "Batch_Prediction"
                    ),
                    mime="application/pdf",
                    icon=":material/download:",
                    width="stretch",
                    on_click="ignore",
                    key="diaml_batch_pdf_download",
                )

            except Exception:
                st.error(
                    "Unable to generate the PDF report. "
                    "Please try again."
                )

        st.divider()

        st.caption(
            "CSV contains all accepted predictions. "
            "Excel includes the summary, predictions, "
            "excluded records, validation errors and warnings."
        )

        st.caption(
            "PDF includes the processing summary, "
            "visual analytics, prediction records "
            "and detailed validation information."
        )


# Render all available batch actions.
def render_batch_actions(result, validation):
    st.subheader("Batch Actions")

    st.caption(
        "Save your completed batch or export prediction results."
    )

    render_batch_save(result)
    render_batch_exports(result, validation)