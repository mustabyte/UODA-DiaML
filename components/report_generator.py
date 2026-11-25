from datetime import datetime, timezone
from io import BytesIO
from math import isfinite
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Flowable,
    PageBreak
)


# Define the project paths and report branding
BASE_DIR = Path(__file__).resolve().parents[1]
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

PROJECT_NAME = "UODA DiaML"
PROJECT_SUBTITLE = "Diabetes Prediction & Analytics System"

DISCLAIMER = (
    "UODA DiaML is an educational machine learning project. "
    "Predictions and simulated results are not medical diagnoses "
    "and must not be used for clinical decisions or treatment."
)


# Define the shared report color palette
PRIMARY = colors.HexColor("#0F766E")
SECONDARY = colors.HexColor("#134E4A")

TEXT = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#64748B")

WHITE = colors.HexColor("#FFFFFF")
BACKGROUND = colors.HexColor("#F8FAFC")
BORDER = colors.HexColor("#E2E8F0")

POSITIVE = colors.HexColor("#B91C1C")
POSITIVE_BG = colors.HexColor("#FEF2F2")

NEGATIVE = colors.HexColor("#0D9488")
NEGATIVE_BG = colors.HexColor("#F0FDFA")

WARNING = colors.HexColor("#D97706")
WARNING_BG = colors.HexColor("#FFFBEB")

NEUTRAL = colors.HexColor("#475569")


# Define compact page margins
LEFT_MARGIN = 36
RIGHT_MARGIN = 36
TOP_MARGIN = 76
BOTTOM_MARGIN = 50


# Define consistent measurement labels and units
MEASUREMENT_INFORMATION = {
    "Pregnancies": ("Pregnancies", "Count"),
    "Glucose": ("Glucose", "mg/dL"),
    "BloodPressure": ("Diastolic Blood Pressure", "mmHg"),
    "SkinThickness": ("Skin Thickness", "mm"),
    "Insulin": ("Insulin", "µU/mL"),
    "BMI": ("Body Mass Index", "kg/m²"),
    "DiabetesPedigreeFunction": (
        "Diabetes Pedigree Function",
        "Unitless"
    ),
    "Age": ("Age", "Years")
}


def get_current_timestamp():
    # Return the current report generation time in UTC
    return datetime.now(timezone.utc)


def format_timestamp(value=None):
    # Format timestamps consistently across reports
    if value is None:
        value = get_current_timestamp()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc).strftime(
            "%d %B %Y, %H:%M UTC"
        )

    return str(value)


def generate_report_filename(report_type):
    # Generate a timestamped PDF filename
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in str(report_type)
    ).strip("_")

    timestamp = get_current_timestamp().strftime(
        "%Y%m%d_%H%M%S"
    )

    return f"UODA_DiaML_{safe_name}_{timestamp}.pdf"


def validate_probability(value):
    # Ensure probabilities use the zero-to-one scale
    probability = float(value)

    if not isfinite(probability):
        raise ValueError("Probability must be finite.")

    if not 0 <= probability <= 1:
        raise ValueError(
            "Probability must be between 0 and 1."
        )

    return probability


def format_probability(value):
    return f"{validate_probability(value) * 100:.2f}%"


def normalize_classification(classification):
    # Standardize classification labels
    value = str(classification).strip().lower()

    if value in ("positive", "predicted positive", "1"):
        return "Positive"

    if value in ("negative", "predicted negative", "0"):
        return "Negative"

    raise ValueError("Invalid prediction classification.")


def get_classification_color(classification):
    classification = normalize_classification(classification)

    if classification == "Positive":
        return POSITIVE

    return NEGATIVE


def get_classification_background(classification):
    classification = normalize_classification(classification)

    if classification == "Positive":
        return POSITIVE_BG

    return NEGATIVE_BG


def format_measurement(feature, value):
    # Format measurements using their expected precision
    if value is None:
        return "Not available"

    if feature == "BMI":
        return f"{float(value):.1f}"

    if feature == "DiabetesPedigreeFunction":
        return f"{float(value):.3f}"

    return str(int(value))


def safe_text(value):
    # Escape text before inserting it into a PDF paragraph
    if value is None:
        return "Not available"

    return escape(str(value)).replace("\n", "<br/>")


def create_report_styles():
    # Define compact and consistent report typography
    return {
        "title": ParagraphStyle(
            name="ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=26,
            textColor=TEXT,
            spaceAfter=5
        ),

        "subtitle": ParagraphStyle(
            name="ReportSubtitle",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceAfter=5
        ),

        "section": ParagraphStyle(
            name="SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=SECONDARY,
            spaceBefore=6,
            spaceAfter=4,
            keepWithNext=True
        ),

        "body": ParagraphStyle(
            name="BodyText",
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=TEXT,
            spaceAfter=3
        ),

        "caption": ParagraphStyle(
            name="CaptionText",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            spaceAfter=3
        ),

        "table_header": ParagraphStyle(
            name="TableHeader",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=WHITE,
            alignment=TA_LEFT
        ),

        "table_cell": ParagraphStyle(
            name="TableCell",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=TEXT,
            alignment=TA_LEFT
        ),

        "card_label": ParagraphStyle(
            name="CardLabel",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=MUTED
        ),

        "card_value": ParagraphStyle(
            name="CardValue",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=TEXT
        ),

        "card_description": ParagraphStyle(
            name="CardDescription",
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=MUTED
        ),

        "disclaimer_title": ParagraphStyle(
            name="DisclaimerTitle",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=SECONDARY
        ),

        "disclaimer": ParagraphStyle(
            name="DisclaimerText",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=NEUTRAL
        )
    }


STYLES = create_report_styles()


class NumberedCanvas(canvas.Canvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        # Preserve each page until the total page count is known
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        # Render every page with its final page number
        total_pages = len(self._saved_page_states)

        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(total_pages)
            super().showPage()

        super().save()

    def draw_page_number(self, total_pages):
        page_width, _ = self._pagesize

        self.saveState()

        self.setFont("Helvetica", 7.5)
        self.setFillColor(MUTED)

        self.drawRightString(
            page_width - RIGHT_MARGIN,
            17,
            f"Page {self._pageNumber} of {total_pages}"
        )

        self.restoreState()


def draw_logo(pdf_canvas, x, y, width=130, height=28):
    # Display the existing logo without stretching it
    if not LOGO_PATH.exists():
        return False

    try:
        image = ImageReader(str(LOGO_PATH))
        image_width, image_height = image.getSize()

        scale = min(
            width / image_width,
            height / image_height
        )

        display_width = image_width * scale
        display_height = image_height * scale

        pdf_canvas.drawImage(
            image,
            x,
            y + (height - display_height) / 2,
            width=display_width,
            height=display_height,
            mask="auto"
        )

        return True

    except Exception:
        return False


def draw_page_header(pdf_canvas, document, report_type):
    # Draw a compact header with the existing horizontal logo
    page_width, page_height = document.pagesize

    pdf_canvas.saveState()

    logo_displayed = draw_logo(
        pdf_canvas,
        LEFT_MARGIN,
        page_height - 46,
        width=130,
        height=28
    )

    if not logo_displayed:
        pdf_canvas.setFillColor(SECONDARY)
        pdf_canvas.setFont("Helvetica-Bold", 17)

        pdf_canvas.drawString(
            LEFT_MARGIN,
            page_height - 35,
            PROJECT_NAME
        )

    pdf_canvas.setFillColor(MUTED)
    pdf_canvas.setFont("Helvetica", 7.5)

    pdf_canvas.drawString(
        LEFT_MARGIN,
        page_height - 53,
        PROJECT_SUBTITLE
    )

    # Display the report category on the right
    pdf_canvas.setFillColor(SECONDARY)

    report_label = str(report_type).upper()
    font_size = 8

    maximum_width = (
        page_width
        - LEFT_MARGIN
        - RIGHT_MARGIN
        - 220
    )

    while (
        pdf_canvas.stringWidth(
            report_label,
            "Helvetica-Bold",
            font_size
        ) > maximum_width
        and font_size > 6
    ):
        font_size -= 0.5

    pdf_canvas.setFont("Helvetica-Bold", font_size)

    pdf_canvas.drawRightString(
        page_width - RIGHT_MARGIN,
        page_height - 32,
        report_label
    )

    pdf_canvas.setFillColor(MUTED)
    pdf_canvas.setFont("Helvetica", 7.5)

    pdf_canvas.drawRightString(
        page_width - RIGHT_MARGIN,
        page_height - 45,
        "Educational Analytics"
    )

    # Separate the header from the report content
    pdf_canvas.setStrokeColor(PRIMARY)
    pdf_canvas.setLineWidth(1.3)

    pdf_canvas.line(
        LEFT_MARGIN,
        page_height - 62,
        page_width - RIGHT_MARGIN,
        page_height - 62
    )

    pdf_canvas.restoreState()


def draw_page_footer(pdf_canvas, document):
    # Draw a compact footer on every page
    page_width, _ = document.pagesize

    pdf_canvas.saveState()

    pdf_canvas.setStrokeColor(BORDER)
    pdf_canvas.setLineWidth(0.6)

    pdf_canvas.line(
        LEFT_MARGIN,
        39,
        page_width - RIGHT_MARGIN,
        39
    )

    pdf_canvas.setFillColor(MUTED)
    pdf_canvas.setFont("Helvetica", 7)

    pdf_canvas.drawString(
        LEFT_MARGIN,
        29,
        "Educational use only - Not a medical diagnosis."
    )

    pdf_canvas.setFillColor(SECONDARY)
    pdf_canvas.setFont("Helvetica-Bold", 7.5)

    pdf_canvas.drawString(
        LEFT_MARGIN,
        17,
        PROJECT_NAME
    )

    pdf_canvas.restoreState()


class ProbabilityBar(Flowable):

    def __init__(
        self,
        probability,
        threshold=0.40,
        classification=None
    ):
        super().__init__()

        self.probability = validate_probability(probability)
        self.threshold = validate_probability(threshold)

        if classification is None:
            classification = (
                "Positive"
                if self.probability >= self.threshold
                else "Negative"
            )

        self.classification = normalize_classification(
            classification
        )

        self.width = 0
        self.height = 62

    def wrap(self, available_width, available_height):
        self.width = available_width
        return self.width, self.height

    def draw(self):
        # Draw a compact probability visualization
        pdf_canvas = self.canv

        pdf_canvas.saveState()

        bar_x = 2
        bar_y = 17
        bar_width = self.width - 4
        bar_height = 12

        probability_width = bar_width * self.probability
        threshold_x = bar_x + bar_width * self.threshold

        classification_color = get_classification_color(
            self.classification
        )

        pdf_canvas.setFillColor(TEXT)
        pdf_canvas.setFont("Helvetica-Bold", 9)

        pdf_canvas.drawString(
            bar_x,
            51,
            "Predicted Positive-Class Probability"
        )

        pdf_canvas.setFillColor(classification_color)

        pdf_canvas.drawRightString(
            self.width - 2,
            51,
            format_probability(self.probability)
        )

        # Draw the probability track
        pdf_canvas.setFillColor(BORDER)

        pdf_canvas.roundRect(
            bar_x,
            bar_y,
            bar_width,
            bar_height,
            4,
            stroke=0,
            fill=1
        )

        if probability_width > 0:
            pdf_canvas.setFillColor(classification_color)

            pdf_canvas.roundRect(
                bar_x,
                bar_y,
                probability_width,
                bar_height,
                min(4, probability_width / 2),
                stroke=0,
                fill=1
            )

        # Mark the classification threshold
        pdf_canvas.setStrokeColor(WARNING)
        pdf_canvas.setLineWidth(1.3)
        pdf_canvas.setDash(3, 2)

        pdf_canvas.line(
            threshold_x,
            bar_y - 3,
            threshold_x,
            bar_y + bar_height + 4
        )

        pdf_canvas.setDash()

        pdf_canvas.setFillColor(WARNING)
        pdf_canvas.setFont("Helvetica", 7.5)

        threshold_label = (
            f"Threshold: {self.threshold * 100:g}%"
        )

        label_width = pdf_canvas.stringWidth(
            threshold_label,
            "Helvetica",
            7.5
        )

        label_x = max(
            label_width / 2,
            min(
                threshold_x,
                self.width - label_width / 2
            )
        )

        pdf_canvas.drawCentredString(
            label_x,
            36,
            threshold_label
        )

        # Display the probability scale
        pdf_canvas.setFillColor(MUTED)
        pdf_canvas.setFont("Helvetica", 7.5)

        pdf_canvas.drawString(
            bar_x,
            5,
            "0%"
        )

        pdf_canvas.drawRightString(
            self.width - 2,
            5,
            "100%"
        )

        pdf_canvas.restoreState()


class ProbabilityComparisonChart(Flowable):

    def __init__(
        self,
        original_probability,
        simulated_probability,
        threshold=0.40
    ):
        super().__init__()

        self.original_probability = validate_probability(
            original_probability
        )

        self.simulated_probability = validate_probability(
            simulated_probability
        )

        self.threshold = validate_probability(threshold)

        self.width = 0
        self.height = 170

    def wrap(self, available_width, available_height):
        self.width = available_width
        return self.width, self.height

    def draw(self):
        # Compare original and simulated model probabilities
        pdf_canvas = self.canv

        pdf_canvas.saveState()

        chart_x = 42
        chart_y = 27
        chart_width = self.width - 62
        chart_height = 112

        # Draw the percentage scale and grid lines
        for percentage in (0, 25, 50, 75, 100):
            y = chart_y + chart_height * percentage / 100

            pdf_canvas.setStrokeColor(BORDER)
            pdf_canvas.setLineWidth(0.5)

            pdf_canvas.line(
                chart_x,
                y,
                chart_x + chart_width,
                y
            )

            pdf_canvas.setFillColor(MUTED)
            pdf_canvas.setFont("Helvetica", 8)

            pdf_canvas.drawRightString(
                chart_x - 7,
                y - 3,
                f"{percentage}%"
            )

        probabilities = [
            self.original_probability,
            self.simulated_probability
        ]

        labels = [
            "Original",
            "Simulated"
        ]

        bar_width = min(
            80,
            chart_width * 0.23
        )

        # Draw the original and simulated probability bars
        for index, probability in enumerate(probabilities):
            center_x = (
                chart_x
                + chart_width * (index + 0.5) / 2
            )

            bar_height = probability * chart_height

            classification = (
                "Positive"
                if probability >= self.threshold
                else "Negative"
            )

            pdf_canvas.setFillColor(
                get_classification_color(classification)
            )

            pdf_canvas.rect(
                center_x - bar_width / 2,
                chart_y,
                bar_width,
                bar_height,
                stroke=0,
                fill=1
            )

            pdf_canvas.setFillColor(TEXT)
            pdf_canvas.setFont("Helvetica-Bold", 9)

            pdf_canvas.drawCentredString(
                center_x,
                chart_y + bar_height + 7,
                format_probability(probability)
            )

            pdf_canvas.setFillColor(MUTED)
            pdf_canvas.setFont("Helvetica", 9)

            pdf_canvas.drawCentredString(
                center_x,
                10,
                labels[index]
            )

        # Draw the classification threshold line
        threshold_y = (
            chart_y + chart_height * self.threshold
        )

        pdf_canvas.setStrokeColor(WARNING)
        pdf_canvas.setLineWidth(1.2)
        pdf_canvas.setDash(4, 3)

        pdf_canvas.line(
            chart_x,
            threshold_y,
            chart_x + chart_width,
            threshold_y
        )

        pdf_canvas.setDash()

        pdf_canvas.setFillColor(WARNING)
        pdf_canvas.setFont("Helvetica", 8)

        pdf_canvas.drawRightString(
            self.width - 4,
            160,
            f"Classification threshold: "
            f"{self.threshold * 100:g}%"
        )

        pdf_canvas.restoreState()


class ClassificationDistributionChart(Flowable):

    def __init__(self, positive_count, negative_count):
        super().__init__()

        self.positive_count = int(positive_count)
        self.negative_count = int(negative_count)

        if self.positive_count < 0 or self.negative_count < 0:
            raise ValueError(
                "Classification counts cannot be negative."
            )

        self.total = (
            self.positive_count + self.negative_count
        )

        self.width = 0
        self.height = 140

    def wrap(self, available_width, available_height):
        self.width = available_width
        return self.width, self.height

    def draw(self):
        # Draw a compact classification distribution chart
        pdf_canvas = self.canv

        pdf_canvas.saveState()

        if self.total == 0:
            pdf_canvas.setFont("Helvetica", 10)
            pdf_canvas.setFillColor(MUTED)

            pdf_canvas.drawCentredString(
                self.width / 2,
                70,
                "No prediction records available."
            )

            pdf_canvas.restoreState()
            return

        positive_rate = self.positive_count / self.total
        negative_rate = self.negative_count / self.total

        center_x = self.width * 0.27
        center_y = 70
        radius = 52

        bounds = (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        )

        start_angle = 90

        # Draw the distribution segments
        for rate, color in (
            (positive_rate, POSITIVE),
            (negative_rate, NEGATIVE)
        ):
            if rate <= 0:
                continue

            extent = rate * 360

            pdf_canvas.setFillColor(color)

            pdf_canvas.wedge(
                *bounds,
                start_angle,
                extent,
                stroke=0,
                fill=1
            )

            start_angle += extent

        # Create the center of the donut chart
        pdf_canvas.setFillColor(WHITE)

        pdf_canvas.circle(
            center_x,
            center_y,
            33,
            stroke=0,
            fill=1
        )

        pdf_canvas.setFillColor(TEXT)
        pdf_canvas.setFont("Helvetica-Bold", 17)

        pdf_canvas.drawCentredString(
            center_x,
            center_y,
            f"{self.total:,}"
        )

        pdf_canvas.setFillColor(MUTED)
        pdf_canvas.setFont("Helvetica", 8)

        pdf_canvas.drawCentredString(
            center_x,
            center_y - 14,
            "Predictions"
        )

        legend_x = self.width * 0.53

        items = [
            (
                "Positive",
                self.positive_count,
                positive_rate,
                POSITIVE
            ),
            (
                "Negative",
                self.negative_count,
                negative_rate,
                NEGATIVE
            )
        ]

        # Display classification counts and percentages
        for index, item in enumerate(items):
            label, count, rate, color = item
            y = 94 - index * 43

            pdf_canvas.setFillColor(color)

            pdf_canvas.roundRect(
                legend_x,
                y,
                10,
                10,
                2,
                stroke=0,
                fill=1
            )

            pdf_canvas.setFillColor(TEXT)
            pdf_canvas.setFont("Helvetica-Bold", 10)

            pdf_canvas.drawString(
                legend_x + 17,
                y + 1,
                label
            )

            pdf_canvas.setFillColor(MUTED)
            pdf_canvas.setFont("Helvetica", 9)

            pdf_canvas.drawString(
                legend_x + 17,
                y - 15,
                f"{count:,} records ({rate * 100:.2f}%)"
            )

        pdf_canvas.restoreState()


def create_data_table(
    headers,
    rows,
    column_widths=None,
    font_size=8.5
):
    # Create a compact table with automatic row wrapping
    if not headers:
        raise ValueError("Table headers cannot be empty.")

    if column_widths is not None:
        if len(column_widths) != len(headers):
            raise ValueError(
                "Column widths must match the header length."
            )

    cell_style = ParagraphStyle(
        name="DynamicTableCell",
        parent=STYLES["table_cell"],
        fontSize=font_size,
        leading=font_size + 2.5
    )

    header_style = ParagraphStyle(
        name="DynamicTableHeader",
        parent=STYLES["table_header"],
        fontSize=font_size,
        leading=font_size + 2.5
    )

    # Define styles for classification and validation states
    status_colors = {
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "warning": WARNING,
        "invalid": POSITIVE,
        "modified": WARNING,
        "clean": NEGATIVE
    }

    status_styles = {
        status: ParagraphStyle(
            name=f"Status_{status}",
            parent=cell_style,
            fontName="Helvetica-Bold",
            textColor=color
        )
        for status, color in status_colors.items()
    }

    table_data = [
        [
            Paragraph(
                safe_text(header),
                header_style
            )
            for header in headers
        ]
    ]

    # Convert table values into formatted paragraphs
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(
                "Every table row must match the header length."
            )

        formatted_row = []

        for value in row:
            status = str(value).strip().lower()

            style = status_styles.get(
                status,
                cell_style
            )

            formatted_row.append(
                Paragraph(
                    safe_text(value),
                    style
                )
            )

        table_data.append(formatted_row)

    table = Table(
        table_data,
        colWidths=column_widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    # Apply compact spacing and alternating backgrounds
    table_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),

        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),

        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),

        ("LINEBELOW", (0, 0), (-1, 0), 0.7, PRIMARY),

        ("BOX", (0, 0), (-1, -1), 0.5, BORDER)
    ]

    if len(table_data) > 1:
        table_commands.extend([
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                WHITE,
                BACKGROUND
            ]),

            ("LINEBELOW", (0, 1), (-1, -1), 0.4, BORDER)
        ])

    table.setStyle(
        TableStyle(table_commands)
    )

    return table


def create_metadata_table(items, available_width):
    # Arrange metadata in a compact two-column layout
    label_style = ParagraphStyle(
        name="MetadataLabel",
        parent=STYLES["caption"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        spaceAfter=2
    )

    value_style = ParagraphStyle(
        name="MetadataValue",
        parent=STYLES["body"],
        fontSize=9,
        leading=12,
        spaceAfter=0
    )

    items = list(items)
    rows = []

    for start in range(0, len(items), 2):
        current_items = items[start:start + 2]
        row = []

        for label, value in current_items:
            row.append([
                Paragraph(
                    safe_text(label),
                    label_style
                ),

                Paragraph(
                    safe_text(value),
                    value_style
                )
            ])

        while len(row) < 2:
            row.append("")

        rows.append(row)

    if not rows:
        return Spacer(1, 0)

    table = Table(
        rows,
        colWidths=[
            available_width / 2,
            available_width / 2
        ],
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BACKGROUND),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),

            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),

            ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),

            ("BOX", (0, 0), (-1, -1), 0.5, BORDER)
        ])
    )

    return table


def create_summary_cards(
    cards,
    available_width,
    columns=3
):
    # Build compact cards with adaptive and aligned heights
    if not cards:
        return Spacer(1, 0)

    if columns < 1:
        raise ValueError("Columns must be at least one.")

    gap = 8
    padding = 7

    card_width = (
        available_width - gap * (columns - 1)
    ) / columns

    content_width = card_width - padding * 2

    if content_width <= 0:
        raise ValueError(
            "Too many summary card columns for the page width."
        )

    grid_rows = []
    grid_widths = []

    # Calculate the card and gap widths
    for index in range(columns):
        grid_widths.append(card_width)

        if index < columns - 1:
            grid_widths.append(gap)

    for start in range(0, len(cards), columns):
        current_cards = cards[start:start + columns]

        prepared_cards = []
        required_heights = []

        for card in current_cards:
            value_color = card.get(
                "color",
                TEXT
            )

            background = card.get(
                "background",
                BACKGROUND
            )

            value_style = ParagraphStyle(
                name="DynamicCardValue",
                parent=STYLES["card_value"],
                textColor=value_color
            )

            content = [
                Paragraph(
                    safe_text(card["label"]),
                    STYLES["card_label"]
                ),

                Spacer(1, 3),

                Paragraph(
                    safe_text(card["value"]),
                    value_style
                )
            ]

            description = card.get("description")

            if description:
                content.extend([
                    Spacer(1, 3),

                    Paragraph(
                        safe_text(description),
                        STYLES["card_description"]
                    )
                ])

            # Measure the natural card height
            measurement_table = Table(
                [[content]],
                colWidths=[card_width]
            )

            measurement_table.setStyle(
                TableStyle([
                    ("LEFTPADDING", (0, 0), (-1, -1), padding),
                    ("RIGHTPADDING", (0, 0), (-1, -1), padding),

                    ("TOPPADDING", (0, 0), (-1, -1), padding),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), padding)
                ])
            )

            _, required_height = measurement_table.wrap(
                card_width,
                10000
            )

            required_heights.append(required_height)

            prepared_cards.append({
                "content": content,
                "color": value_color,
                "background": background
            })

        # Keep every card in the current row equally tall
        row_height = max(
            60,
            max(required_heights)
        )

        grid_row = []

        for index in range(columns):
            if index < len(prepared_cards):
                card = prepared_cards[index]

                card_table = Table(
                    [[card["content"]]],
                    colWidths=[card_width],
                    rowHeights=[row_height]
                )

                card_table.setStyle(
                    TableStyle([
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, -1),
                            card["background"]
                        ),

                        (
                            "BOX",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            BORDER
                        ),

                        (
                            "LINEABOVE",
                            (0, 0),
                            (-1, 0),
                            2,
                            card["color"]
                        ),

                        ("VALIGN", (0, 0), (-1, -1), "TOP"),

                        ("LEFTPADDING", (0, 0), (-1, -1), padding),
                        ("RIGHTPADDING", (0, 0), (-1, -1), padding),

                        ("TOPPADDING", (0, 0), (-1, -1), padding),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), padding)
                    ])
                )

                grid_row.append(card_table)

            else:
                grid_row.append("")

            if index < columns - 1:
                grid_row.append("")

        grid_rows.append(grid_row)

    # Combine the cards into an aligned grid
    grid = Table(
        grid_rows,
        colWidths=grid_widths,
        hAlign="LEFT"
    )

    grid.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),

            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3)
        ])
    )

    return grid


def create_measurement_table(
    measurements,
    available_width
):
    # Convert model inputs into a readable measurement table
    rows = []

    for feature, information in MEASUREMENT_INFORMATION.items():
        label, unit = information

        value = format_measurement(
            feature,
            measurements.get(feature)
        )

        rows.append([
            label,
            value,
            unit
        ])

    return create_data_table(
        headers=[
            "Measurement",
            "Value",
            "Unit"
        ],

        rows=rows,

        column_widths=[
            available_width * 0.50,
            available_width * 0.25,
            available_width * 0.25
        ]
    )


def create_disclaimer(available_width):
    # Create a compact educational-use notice
    content = [
        Paragraph(
            "Educational Use Only",
            STYLES["disclaimer_title"]
        ),

        Spacer(1, 3),

        Paragraph(
            safe_text(DISCLAIMER),
            STYLES["disclaimer"]
        )
    ]

    table = Table(
        [[content]],
        colWidths=[available_width],
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BACKGROUND),

            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),

            ("LINEBEFORE", (0, 0), (0, -1), 2.5, PRIMARY),

            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),

            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
        ])
    )

    return table


class PDFReportBuilder:

    def __init__(
        self,
        report_title,
        subtitle=None,
        orientation="portrait"
    ):
        # Initialize the shared document configuration
        if orientation not in ("portrait", "landscape"):
            raise ValueError("Invalid report orientation.")

        self.report_title = report_title
        self.subtitle = subtitle
        self.orientation = orientation

        self.page_size = (
            landscape(A4)
            if orientation == "landscape"
            else A4
        )

        self.available_width = (
            self.page_size[0]
            - LEFT_MARGIN
            - RIGHT_MARGIN
            - 12
        )

        self.story = []
        self.generated_at = get_current_timestamp()

        self.add_title(report_title)

        if subtitle:
            self.add_subtitle(subtitle)

    def add_title(self, title):
        self.story.append(
            Paragraph(
                safe_text(title),
                STYLES["title"]
            )
        )

    def add_subtitle(self, subtitle):
        self.story.append(
            Paragraph(
                safe_text(subtitle),
                STYLES["subtitle"]
            )
        )

    def add_section(self, title):
        self.story.append(
            Paragraph(
                safe_text(title),
                STYLES["section"]
            )
        )

    def add_text(self, text):
        self.story.append(
            Paragraph(
                safe_text(text),
                STYLES["body"]
            )
        )

    def add_caption(self, text):
        self.story.append(
            Paragraph(
                safe_text(text),
                STYLES["caption"]
            )
        )

    def add_space(self, height=2):
        self.story.append(
            Spacer(1, height)
        )

    def add_metadata(self, items):
        # Add report metadata and generation timestamp
        metadata = list(items)

        metadata.append((
            "Generated At",
            format_timestamp(self.generated_at)
        ))

        self.story.append(
            create_metadata_table(
                metadata,
                self.available_width
            )
        )

        self.add_space()

    def add_cards(self, cards, columns=3):
        self.story.append(
            create_summary_cards(
                cards,
                self.available_width,
                columns
            )
        )

        self.add_space()

    def add_table(
        self,
        headers,
        rows,
        column_widths=None,
        font_size=8.5
    ):
        # Add a paginated report table
        if not headers:
            raise ValueError("Table headers cannot be empty.")

        if column_widths is None:
            column_widths = [
                self.available_width / len(headers)
            ] * len(headers)

        if sum(column_widths) > self.available_width + 0.1:
            raise ValueError(
                "Table width exceeds the available page width."
            )

        self.story.append(
            create_data_table(
                headers,
                rows,
                column_widths,
                font_size
            )
        )

        self.add_space()

    def add_measurements(self, measurements):
        self.story.append(
            create_measurement_table(
                measurements,
                self.available_width
            )
        )

        self.add_space()

    def add_probability_bar(
        self,
        probability,
        threshold=0.40,
        classification=None
    ):
        self.story.append(
            ProbabilityBar(
                probability,
                threshold,
                classification
            )
        )

        self.add_space()

    def add_probability_comparison(
        self,
        original_probability,
        simulated_probability,
        threshold=0.40
    ):
        self.story.append(
            ProbabilityComparisonChart(
                original_probability,
                simulated_probability,
                threshold
            )
        )

        self.add_space()

    def add_classification_distribution(
        self,
        positive_count,
        negative_count
    ):
        self.story.append(
            ClassificationDistributionChart(
                positive_count,
                negative_count
            )
        )

        self.add_space()

    def add_disclaimer(self):
        self.add_space(3)

        self.story.append(
            create_disclaimer(
                self.available_width
            )
        )

    def add_page_break(self):
        self.story.append(
            PageBreak()
        )

    def add_element(self, element):
        self.story.append(element)

    def draw_page(self, pdf_canvas, document):
        # Apply the same header and footer to every page
        draw_page_header(
            pdf_canvas,
            document,
            self.report_title
        )

        draw_page_footer(
            pdf_canvas,
            document
        )

    def build(self):
        # Generate the complete PDF as in-memory bytes
        with BytesIO() as buffer:
            document = SimpleDocTemplate(
                buffer,

                pagesize=self.page_size,

                leftMargin=LEFT_MARGIN,
                rightMargin=RIGHT_MARGIN,

                topMargin=TOP_MARGIN,
                bottomMargin=BOTTOM_MARGIN,

                title=f"{PROJECT_NAME} - {self.report_title}",

                author=PROJECT_NAME,
                subject=PROJECT_SUBTITLE,

                allowSplitting=True
            )

            document.build(
                list(self.story),

                onFirstPage=self.draw_page,
                onLaterPages=self.draw_page,

                canvasmaker=NumberedCanvas
            )

            return buffer.getvalue()


def generate_individual_report(
    measurements,
    probability,
    classification,
    threshold=0.40,
    prediction_id=None,
    prediction_date=None,
    model_name="Random Forest",
    model_version="v1.0",
    warnings=None,
    source="individual",
    saved_history=False
):
    # Validate the prediction information
    probability = validate_probability(probability)
    threshold = validate_probability(threshold)

    classification = normalize_classification(
        classification
    )

    expected_classification = (
        "Positive"
        if probability >= threshold
        else "Negative"
    )

    if classification != expected_classification:
        raise ValueError(
            "Classification does not match the probability "
            "and decision threshold."
        )

    # Ensure all required measurements are available
    missing_features = [
        feature
        for feature in MEASUREMENT_INFORMATION
        if feature not in measurements
        or measurements[feature] is None
    ]

    if missing_features:
        raise ValueError(
            "Missing measurements: "
            + ", ".join(missing_features)
        )

    if source not in ("individual", "batch"):
        raise ValueError("Invalid prediction source.")

    # Initialize the individual prediction report
    report = PDFReportBuilder(
        report_title="Individual Prediction Report",
        subtitle=(
            "Personalized machine learning prediction summary "
            "based on the submitted health measurements."
        ),
        orientation="portrait"
    )

    classification_color = get_classification_color(
        classification
    )

    classification_background = (
        get_classification_background(classification)
    )

    # Prepare the prediction identifier and database status
    prediction_reference = (
        f"#{prediction_id}"
        if prediction_id is not None
        else "Not saved"
    )

    database_status = (
        "Saved to Prediction History"
        if prediction_id is not None
        else "Not saved to Prediction History"
    )

    source_label = (
        "Batch Prediction"
        if source == "batch"
        else "Individual Prediction"
    )

    # Add report and prediction metadata
    metadata = [
        ("Prediction ID", prediction_reference),
        ("Prediction Source", source_label),
        (
            "Prediction Date",
            format_timestamp(prediction_date)
            if prediction_date is not None
            else "Not available"
        ),
        ("Database Status", database_status),
        ("Model", model_name),
        ("Model Version", model_version)
    ]

    report.add_metadata(metadata)

    # Display the classification and probability summary
    report.add_section("Prediction Summary")

    report.add_cards(
        [
            {
                "label": "Classification",
                "value": classification,
                "description": "Model-predicted class",
                "color": classification_color,
                "background": classification_background
            },
            {
                "label": "Positive-Class Probability",
                "value": format_probability(probability),
                "description": (
                    f"Decision threshold: "
                    f"{threshold * 100:g}%"
                ),
                "color": classification_color,
                "background": classification_background
            }
        ],
        columns=2
    )

    # Visualize the predicted probability and threshold
    report.add_probability_bar(
        probability=probability,
        threshold=threshold,
        classification=classification
    )

    # Explain how the model determined the classification
    if classification == "Positive":
        interpretation = (
            f"The model-estimated positive-class probability "
            f"of {format_probability(probability)} meets or "
            f"exceeds the classification threshold of "
            f"{threshold * 100:g}%, resulting in a positive "
            f"classification."
        )

    else:
        interpretation = (
            f"The model-estimated positive-class probability "
            f"of {format_probability(probability)} is below "
            f"the classification threshold of "
            f"{threshold * 100:g}%, resulting in a negative "
            f"classification."
        )

    report.add_section("Model Interpretation")
    report.add_text(interpretation)

    report.add_caption(
        "The reported probability is a model output, not a "
        "clinically calibrated estimate of diabetes risk."
    )

    # Display the submitted health measurements
    report.add_section("Submitted Measurements")
    report.add_measurements(measurements)

    # Include unusual measurement warnings when available
    if warnings:
        report.add_section("Measurement Warnings")

        warning_rows = []

        for feature, message in warnings.items():
            label = MEASUREMENT_INFORMATION.get(
                feature,
                (feature, "")
            )[0]

            warning_rows.append([
                label,
                str(message)
            ])

        report.add_table(
            headers=[
                "Measurement",
                "Validation Warning"
            ],
            rows=warning_rows,
            column_widths=[
                report.available_width * 0.30,
                report.available_width * 0.70
            ],
            font_size=8
        )

        report.add_caption(
            "Unusual measurements were accepted by the "
            "application but should be verified before "
            "interpreting the prediction."
        )

    # Include the educational disclaimer
    if saved_history:
        report.add_caption("Original measurement warnings were not stored in history.")

    report.add_disclaimer()

    return report.build()

def generate_what_if_report(
    original_measurements, simulated_measurements,
    original_result, simulated_result, threshold=0.40,
    source="Manual Measurements", model_name="Random Forest",
    model_version="v1.0", baseline_warnings=None,
    simulation_warnings=None
):
    # Validate the existing simulation results without rerunning the model.
    threshold = validate_probability(threshold)
    probabilities = []
    classifications = []
    for result in (original_result, simulated_result):
        probability = validate_probability(result["probability"])
        classification = normalize_classification(result["classification"])
        if classification != ("Positive" if probability >= threshold else "Negative"):
            raise ValueError("Classification does not match the threshold.")
        probabilities.append(probability)
        classifications.append(classification)

    rows = []
    modified = 0
    for feature, (label, unit) in MEASUREMENT_INFORMATION.items():
        original = float(original_measurements[feature])
        simulated = float(simulated_measurements[feature])
        if not isfinite(original) or not isfinite(simulated):
            raise ValueError("Measurements must be finite.")
        difference = simulated - original
        changed = abs(difference) > 1e-8
        modified += int(changed)
        decimals = 3 if feature == "DiabetesPedigreeFunction" else 1 if feature == "BMI" else 0
        rows.append([label, format_measurement(feature, original),
                     format_measurement(feature, simulated),
                     f"{difference:+.{decimals}f}" if changed else "—",
                     unit, "Modified" if changed else "Unchanged"])

    report = PDFReportBuilder("What-If Simulation Report",
        "Comparative analysis of baseline and hypothetical measurements.")
    report.add_metadata([
        ("Baseline Source", source), ("Model", model_name),
        ("Model Version", model_version), ("Decision Threshold", format_probability(threshold)),
        ("Modified Measurements", f"{modified} of 8")])
    report.add_section("Simulation Summary")
    change = (probabilities[1] - probabilities[0]) * 100
    cards = []
    for label, probability, classification in zip(
        ("Original Prediction", "Simulated Prediction"), probabilities, classifications
    ):
        cards.append(dict(label=label, value=format_probability(probability),
            description=classification + " classification", color=get_classification_color(classification),
            background=get_classification_background(classification)))
    cards.append(dict(label="Probability Change", value=f"{change:+.2f} pp" if abs(change)>1e-8 else "0.00 pp",
        description="Percentage points", color=NEUTRAL, background=BACKGROUND))
    report.add_cards(cards)
    report.add_section("Probability Comparison")
    chart = ProbabilityComparisonChart(*probabilities, threshold)
    # Scale the comparison chart only for this report.
    chart.height = 125
    original_draw = chart.draw
    def draw_compact_chart():
        chart.canv.saveState()
        chart.canv.scale(1, 125 / 170)
        original_draw()
        chart.canv.restoreState()
    chart.draw = draw_compact_chart
    report.add_element(chart)
    report.add_section("Measurement Comparison")
    report.add_table(["Measurement", "Original", "Simulated", "Change", "Unit", "Status"], rows,
        [report.available_width * fraction for fraction in (.31,.14,.14,.12,.13,.16)], font_size=8)
    report.add_section("Simulation Interpretation")
    direction = "increased" if change > 1e-8 else "decreased" if change < -1e-8 else None
    explanation = (f"Model probability {direction} by {abs(change):.2f} percentage points." if direction else "Model probability remained unchanged.")
    report.add_text(f"{explanation} Classification: {classifications[0]} to {classifications[1]}.")
    report.add_caption("Hypothetical model outputs do not establish causal effects or changes in actual diabetes risk. Simulation results are not saved to history.")
    warning_rows = []
    for stage, warnings in (("Baseline", baseline_warnings), ("Simulated", simulation_warnings)):
        for feature, message in (warnings or {}).items():
            warning_rows.append([stage, MEASUREMENT_INFORMATION.get(feature, (feature, ""))[0], str(message)])
    if warning_rows:
        report.add_section("Measurement Warnings")
        report.add_table(["Source", "Measurement", "Warning"], warning_rows,
            [report.available_width * f for f in (.16,.29,.55)], font_size=8)
    report.add_disclaimer()
    return report.build()


class BatchProbabilityHistogram(Flowable):
    def __init__(self, probabilities, threshold):
        super().__init__()
        self.counts = [0] * 10
        for value in probabilities:
            self.counts[min(9, int(validate_probability(value) * 10))] += 1
        self.threshold = validate_probability(threshold)
        self.height = 108

    def wrap(self, available_width, available_height):
        self.width = available_width
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        x, y, h = 24, 20, 66
        w = self.width - 62
        peak = max(1, max(self.counts))
        c.setFont('Helvetica', 8)
        c.setFillColor(MUTED)
        c.drawString(x, 98, 'Record counts')
        for i, count in enumerate(self.counts):
            bh = h * count / peak
            bx = x + i * w / 10
            c.setFillColor(PRIMARY)
            c.rect(bx + 3, y, w / 10 - 6, bh, stroke=0, fill=1)
            c.setFillColor(TEXT)
            c.drawCentredString(bx + w / 20, y + bh + 4, str(count))
        c.setStrokeColor(WARNING)
        c.setDash(3, 2)
        tx = x + w * self.threshold
        c.line(tx, y, tx, y + h + 12)
        c.setDash()
        c.setFillColor(MUTED)
        for i in range(11):
            c.drawCentredString(x + i * w / 10, 14, f'{i*10}%')
        c.restoreState()


def generate_batch_report(result, validation, batch_id=None, filename=None, saved_at=None, model_version="v1.0"):
    saved_history = validation is None
    if saved_history and not batch_id:
        raise ValueError("A saved batch ID is required.")
    # Build the executive summary using the shared report foundation.
    report = PDFReportBuilder('Saved Batch Prediction Report' if saved_history else 'Batch Prediction Report', 'Saved prediction analytics and records.' if saved_history else 'Batch processing, prediction analytics and validation results.', 'landscape')
    metadata = [
        ('Saved On', format_timestamp(saved_at)) if saved_history else ('Source File', filename or 'Not specified'),
        ('Batch ID', str(batch_id) if batch_id else 'Not saved'),
        ('Database Status', 'Saved' if batch_id else 'Not saved'),
        ('Model', result.model_name), ('Model Version', model_version),
        ('Decision Threshold', format_probability(result.threshold)),
    ]
    metadata.append(("Generated At", format_timestamp(report.generated_at)))
    cells = [[Paragraph(safe_text(label), STYLES["caption"]), Paragraph(safe_text(value), STYLES["body"])] for label, value in metadata]
    cells.append([Paragraph("", STYLES["caption"]), Paragraph("", STYLES["body"])])
    table = Table([cells[i] + cells[i+1] for i in range(0, len(cells), 2)], colWidths=[report.available_width*f for f in (.15,.35,.15,.35)])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), BACKGROUND), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("GRID", (0,0), (-1,-1), .4, BORDER)]))
    table.setStyle(TableStyle([("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
    report.add_element(table)
    report.add_section('Saved Batch Summary' if saved_history else 'Processing Summary')
    report.add_cards([
        {'label': label, 'value': f'{count:,}' if count is not None else 'N/A', 'color': color}
        for label, count, color in [
            ('Uploaded', None if saved_history else validation.total_rows, TEXT),
            ('Processed', result.total_processed, PRIMARY),
            ('Excluded', None if saved_history else result.total_excluded, POSITIVE),
            ('Positive', result.positive_count, POSITIVE),
            ('Negative', result.negative_count, NEGATIVE),
            ('Warning Records', None if saved_history else result.total_warnings, WARNING),
        ]
    ], columns=6)
    if saved_history:
        report.add_caption(f'Average positive-class probability: {format_probability(result.average_probability)} | Original upload totals, excluded records and validation warnings are not stored in history.')
    else:
        report.add_caption(f'Acceptance: {validation.success_rate:.2f}% | Average positive-class probability: {format_probability(result.average_probability)} | Clean records: {validation.clean_rows:,}')
    # Use the landscape width for both charts on the summary page.
    report.add_section('Visual Analytics')
    width = report.available_width
    donut = ClassificationDistributionChart(result.positive_count, result.negative_count)
    donut.height = 108
    original_draw = donut.draw
    def draw_compact_donut():
        donut.canv.saveState()
        donut.canv.scale(1, 108 / 140)
        original_draw()
        donut.canv.restoreState()
    donut.draw = draw_compact_donut
    charts = Table([
        [Paragraph('Classification Distribution', STYLES['body']),
         Paragraph('Probability Distribution', STYLES['body'])],
        [donut,
         BatchProbabilityHistogram(result.results['Probability'], result.threshold)]
    ], colWidths=[width * .43, width * .57])
    charts.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    report.add_element(charts)
    report.add_caption(f'Dashed line: classification threshold {format_probability(result.threshold)}. Each bar represents a 10-percentage-point interval; the final interval includes 100%.')
    disclaimer = create_disclaimer(report.available_width)
    disclaimer.setStyle(TableStyle([("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
    report.add_element(disclaimer)
    report.add_page_break()
    report.add_section('Accepted Prediction Records')
    report.add_caption('Measurements use the units shown in the column headers. File rows refer to the original uploaded file.')
    features = list(MEASUREMENT_INFORMATION)
    warning_rows = set(validation.warnings['File Row']) if not saved_history and not validation.warnings.empty else set()
    headers = ['File Row', 'Pregnancies', 'Glucose (mg/dL)', 'BP (mmHg)', 'Skin (mm)', 'Insulin (µU/mL)', 'BMI (kg/m²)', 'DPF', 'Age (years)', 'Class', 'Probability', 'Validation']
    rows = ([int(row['File Row']), *[format_measurement(f, row[f]) for f in features], row['Classification'], format_probability(row['Probability']), 'N/A' if saved_history else ('Warning' if row['File Row'] in warning_rows else 'Clean')] for _, row in result.results.iterrows())
    report.add_table(headers, rows, font_size=7.5)
    # Keep untrusted uploaded values bounded in the printable report.
    def printable(value):
        value = str(value)
        return value if len(value) <= 500 else value[:500] + ' [truncated; see Excel]'
    if saved_history:
        return report.build()
    if validation.invalid_data.empty and validation.errors.empty and validation.warnings.empty:
        return report.build()
    for title, frame, columns in [
        ('Excluded Records', validation.invalid_data, ['File Row', 'Error Count', 'Errors']),
        ('Validation Errors', validation.errors, ['File Row', 'Field', 'Entered Value', 'Issue']),
        ('Validation Warnings', validation.warnings, ['File Row', 'Field', 'Entered Value', 'Warning']),
    ]:
        report.add_section(title)
        if frame.empty:
            report.add_text('No records in this section.')
            continue
        widths = ([.12,.13,.75] if len(columns)==3 else [.10,.19,.23,.48])
        report.add_table(columns, ([printable(row[col]) for col in columns] for _, row in frame.iterrows()), [report.available_width * w for w in widths], font_size=8)
    report.add_caption('Long uploaded text may be shortened in this PDF. The Excel export contains the full excluded measurements and validation details.')
    return report.build()


# Reconstruct a report from saved records without inventing validation data.
def generate_saved_batch_report(records, batch_id):
    from types import SimpleNamespace
    import pandas as pd

    records = list(records)
    if not records:
        raise ValueError("No saved batch records available.")
    first = records[0]
    mapping = {
        "Pregnancies": "pregnancies", "Glucose": "glucose",
        "BloodPressure": "blood_pressure", "SkinThickness": "skin_thickness",
        "Insulin": "insulin", "BMI": "bmi",
        "DiabetesPedigreeFunction": "diabetes_pedigree_function", "Age": "age",
    }
    rows = []
    threshold = validate_probability(first["threshold"])
    for record in records:
        if record["source"] != "batch" or str(record["batch_id"]) != str(batch_id):
            raise ValueError("Record does not belong to the selected batch.")
        if any(record[key] != first[key] for key in ("model_name", "model_version", "threshold")):
            raise ValueError("Inconsistent saved batch model information.")
        probability = validate_probability(record["probability"])
        classification = normalize_classification(record["classification"])
        if classification != ("Positive" if probability >= threshold else "Negative"):
            raise ValueError("Inconsistent saved classification.")
        rows.append({"File Row": int(record["batch_row_number"]),
                     **{feature: float(record[column]) for feature, column in mapping.items()},
                     "Probability": probability, "Classification": classification})
    frame = pd.DataFrame(rows).sort_values("File Row")
    positive = int((frame["Classification"] == "Positive").sum())
    result = SimpleNamespace(results=frame, threshold=threshold, model_name=first["model_name"],
                             total_processed=len(frame), positive_count=positive,
                             negative_count=len(frame)-positive, average_probability=float(frame["Probability"].mean()))
    return generate_batch_report(result, None, batch_id=batch_id,
                                 saved_at=min(record["created_at"] for record in records),
                                 model_version=first["model_version"])
