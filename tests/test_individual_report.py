
from datetime import datetime, timezone
from pathlib import Path

from components.report_generator import (
    generate_individual_report
)

from components.validation import (
    validate_prediction_input
)


# Define the output directory for generated test reports
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "report_previews"
)


# Define representative prediction measurements
POSITIVE_MEASUREMENTS = {
    "Pregnancies": 2,
    "Glucose": 145,
    "BloodPressure": 78,
    "SkinThickness": 25,
    "Insulin": 110,
    "BMI": 29.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35
}

NEGATIVE_MEASUREMENTS = {
    "Pregnancies": 1,
    "Glucose": 95,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 23.5,
    "DiabetesPedigreeFunction": 0.25,
    "Age": 25
}

WARNING_MEASUREMENTS = {
    "Pregnancies": 5,
    "Glucose": 280,
    "BloodPressure": 88,
    "SkinThickness": 35,
    "Insulin": 210,
    "BMI": 36.4,
    "DiabetesPedigreeFunction": 0.85,
    "Age": 52
}


def create_sample_report(
    measurements,
    probability,
    classification,
    prediction_id=None
):
    # Validate the sample measurements using application rules
    validation_result = validate_prediction_input(
        measurements
    )

    if not validation_result.is_valid:
        raise ValueError(
            "Invalid test measurements: "
            + str(validation_result.errors)
        )

    # Generate a report using representative prediction data
    return generate_individual_report(
        measurements=validation_result.cleaned_data,
        probability=probability,
        classification=classification,
        threshold=0.40,
        prediction_id=prediction_id,
        prediction_date=datetime.now(timezone.utc),
        model_name="Random Forest",
        model_version="v1.0",
        warnings=validation_result.warnings,
        source="individual"
    )


def test_positive_report():
    # Verify that a positive prediction produces PDF bytes
    pdf_bytes = create_sample_report(
        measurements=POSITIVE_MEASUREMENTS,
        probability=0.570641360863246,
        classification="Positive",
        prediction_id=1
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")


def test_negative_report():
    # Verify that a negative prediction produces PDF bytes
    pdf_bytes = create_sample_report(
        measurements=NEGATIVE_MEASUREMENTS,
        probability=0.0333451104600568,
        classification="Negative",
        prediction_id=2
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")


def test_report_with_warnings():
    # Verify that reports support unusual measurements
    validation_result = validate_prediction_input(
        WARNING_MEASUREMENTS
    )

    assert validation_result.is_valid
    assert validation_result.has_warnings

    pdf_bytes = create_sample_report(
        measurements=WARNING_MEASUREMENTS,
        probability=0.883860078906773,
        classification="Positive",
        prediction_id=3
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_unsaved_report():
    # Verify that a database identifier is not required
    pdf_bytes = create_sample_report(
        measurements=POSITIVE_MEASUREMENTS,
        probability=0.570641360863246,
        classification="Positive",
        prediction_id=None
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_inconsistent_classification():
    # Reject a classification that conflicts with the threshold
    try:
        create_sample_report(
            measurements=POSITIVE_MEASUREMENTS,
            probability=0.5706,
            classification="Negative",
            prediction_id=1
        )

    except ValueError as error:
        assert "Classification" in str(error)

    else:
        raise AssertionError(
            "An inconsistent classification was accepted."
        )


def test_missing_measurement():
    # Reject reports containing incomplete measurements
    incomplete_measurements = (
        POSITIVE_MEASUREMENTS.copy()
    )

    incomplete_measurements.pop("Glucose")

    try:
        generate_individual_report(
            measurements=incomplete_measurements,
            probability=0.5706,
            classification="Positive",
            threshold=0.40
        )

    except ValueError as error:
        assert "Glucose" in str(error)

    else:
        raise AssertionError(
            "An incomplete measurement set was accepted."
        )


def generate_report_previews():
    # Create the directory for visual report verification
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Define the four report preview scenarios
    scenarios = [
        {
            "filename": "Individual_Positive.pdf",
            "measurements": POSITIVE_MEASUREMENTS,
            "probability": 0.570641360863246,
            "classification": "Positive",
            "prediction_id": 1
        },
        {
            "filename": "Individual_Negative.pdf",
            "measurements": NEGATIVE_MEASUREMENTS,
            "probability": 0.0333451104600568,
            "classification": "Negative",
            "prediction_id": 2
        },
        {
            "filename": "Individual_With_Warnings.pdf",
            "measurements": WARNING_MEASUREMENTS,
            "probability": 0.883860078906773,
            "classification": "Positive",
            "prediction_id": 3
        },
        {
            "filename": "Individual_Unsaved.pdf",
            "measurements": POSITIVE_MEASUREMENTS,
            "probability": 0.570641360863246,
            "classification": "Positive",
            "prediction_id": None
        }
    ]

    # Generate and save each preview report
    for scenario in scenarios:
        pdf_bytes = create_sample_report(
            measurements=scenario["measurements"],
            probability=scenario["probability"],
            classification=scenario["classification"],
            prediction_id=scenario["prediction_id"]
        )

        output_path = (
            OUTPUT_DIR / scenario["filename"]
        )

        output_path.write_bytes(pdf_bytes)

        print(
            f"Generated: {output_path.name} "
            f"({len(pdf_bytes) / 1024:.2f} KB)"
        )

    print("\nAll report previews generated successfully.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


# Generate preview files only when executed directly
if __name__ == "__main__":
    generate_report_previews()
