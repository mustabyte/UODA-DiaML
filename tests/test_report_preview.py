
from pathlib import Path

from components.report_generator import (
    PDFReportBuilder,
    POSITIVE,
    POSITIVE_BG,
    NEGATIVE,
    NEGATIVE_BG,
    PRIMARY
)


def main():
    # Create sample measurements for the report preview
    measurements = {
        "Pregnancies": 2,
        "Glucose": 145,
        "BloodPressure": 78,
        "SkinThickness": 25,
        "Insulin": 110,
        "BMI": 29.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 35
    }

    # Initialize the sample report
    report = PDFReportBuilder(
        report_title="Individual Prediction Report",
        subtitle="Sample report for template verification."
    )

    # Add report metadata
    report.add_metadata([
        ("Prediction ID", "#001"),
        ("Model", "Random Forest"),
        ("Model Version", "v1.0")
    ])

    # Add the prediction summary cards
    report.add_section("Prediction Summary")

    report.add_cards(
        [
            {
                "label": "Classification",
                "value": "Positive",
                "description": "Model classification",
                "color": POSITIVE,
                "background": POSITIVE_BG
            },
            {
                "label": "Probability",
                "value": "57.06%",
                "description": "Positive-class probability",
                "color": PRIMARY,
                "background": NEGATIVE_BG
            }
        ],
        columns=2
    )

    # Add the probability visualization
    report.add_probability_bar(
        probability=0.5706,
        threshold=0.40,
        classification="Positive"
    )

    # Add the submitted measurements
    report.add_section("Submitted Measurements")
    report.add_measurements(measurements)

    # Add the educational disclaimer
    report.add_disclaimer()

    # Start a new page to verify the remaining components
    report.add_page_break()

    # Add the What-If probability comparison chart
    report.add_section("Probability Comparison")

    report.add_probability_comparison(
        original_probability=0.5706,
        simulated_probability=0.2840,
        threshold=0.40
    )

    report.add_caption(
        "Illustrative probabilities for layout testing only."
    )

    # Add the classification distribution chart
    report.add_section("Classification Distribution")

    report.add_classification_distribution(
        positive_count=3,
        negative_count=2
    )

    # Add the batch summary cards
    report.add_section("Batch Summary")

    report.add_cards([
        {
            "label": "Processed Records",
            "value": "5",
            "description": "Accepted records",
            "color": PRIMARY
        },
        {
            "label": "Positive",
            "value": "3",
            "description": "60% of predictions",
            "color": POSITIVE,
            "background": POSITIVE_BG
        },
        {
            "label": "Negative",
            "value": "2",
            "description": "40% of predictions",
            "color": NEGATIVE,
            "background": NEGATIVE_BG
        }
    ])

    # Add the disclaimer to the final section
    report.add_disclaimer()

    # Generate the PDF in memory
    pdf_bytes = report.build()

    # Save the preview inside the tests directory
    output_path = (
        Path(__file__).resolve().parent
        / "UODA_DiaML_Report_Preview.pdf"
    )

    output_path.write_bytes(pdf_bytes)

    # Display the generation result
    print("\nPDF generated successfully.")
    print(f"Saved to: {output_path.resolve()}")
    print(f"File size: {len(pdf_bytes) / 1024:.2f} KB")


# Run the preview only when executed directly
if __name__ == "__main__":
    main()
