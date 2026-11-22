
from datetime import date, datetime, time, timedelta, timezone

from psycopg.rows import dict_row

from database.connection import get_connection


# Match the database columns with the prediction form fields.
FEATURE_COLUMNS = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree_function",
    "Age": "age",
}


# Save a completed prediction and return its database ID.
def save_prediction(
    measurements,
    probability,
    classification,
    threshold=0.40,
    model_name="Random Forest",
    model_version="v1.0",
):
    if classification not in ("Positive", "Negative"):
        raise ValueError("Classification must be Positive or Negative.")

    probability = float(probability)
    threshold = float(threshold)

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    expected_classification = (
        "Positive" if probability >= threshold else "Negative"
    )

    if classification != expected_classification:
        raise ValueError("Classification does not match the probability.")

    missing = [
        name for name in FEATURE_COLUMNS if name not in measurements
    ]

    if missing:
        raise ValueError(f"Missing measurements: {', '.join(missing)}")

    columns = list(FEATURE_COLUMNS.values())

    values = [
        measurements[name]
        for name in FEATURE_COLUMNS
    ]

    columns.extend([
        "probability",
        "classification",
        "threshold",
        "model_name",
        "model_version",
    ])

    values.extend([
        probability,
        classification,
        threshold,
        model_name,
        model_version,
    ])

    column_sql = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(values))

    query = f"""
        INSERT INTO public.predictions ({column_sql})
        VALUES ({placeholders})
        RETURNING id;
    """

    # Insert the prediction using parameterized SQL.
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, values)
            prediction_id = cursor.fetchone()[0]

    return prediction_id


# Retrieve saved predictions with optional filters and pagination.
def get_all_predictions(
    classification=None,
    search_id=None,
    start_date=None,
    end_date=None,
    limit=50,
    offset=0,
):
    limit = int(limit)
    offset = int(offset)

    if not 1 <= limit <= 500:
        raise ValueError("Limit must be between 1 and 500.")

    if offset < 0:
        raise ValueError("Offset cannot be negative.")

    conditions = []
    parameters = []

    if classification is not None:
        if classification not in ("Positive", "Negative"):
            raise ValueError("Invalid classification filter.")

        conditions.append("classification = %s")
        parameters.append(classification)

    if search_id is not None:
        conditions.append("id = %s")
        parameters.append(int(search_id))

    if start_date is not None:
        conditions.append("created_at >= %s")
        parameters.append(
            datetime.combine(
                start_date,
                time.min,
                tzinfo=timezone.utc,
            )
        )

    if end_date is not None:
        conditions.append("created_at < %s")
        parameters.append(
            datetime.combine(
                end_date + timedelta(days=1),
                time.min,
                tzinfo=timezone.utc,
            )
        )

    where_sql = (
        "WHERE " + " AND ".join(conditions)
        if conditions
        else ""
    )

    query = f"""
        SELECT *
        FROM public.predictions
        {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT %s OFFSET %s;
    """

    parameters.extend([limit, offset])

    # Return rows as dictionaries for easy use in Streamlit.
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchall()


# Retrieve one saved prediction by its unique ID.
def get_prediction_by_id(prediction_id):
    query = """
        SELECT *
        FROM public.predictions
        WHERE id = %s;
    """

    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, (int(prediction_id),))
            return cursor.fetchone()


# Count predictions, optionally filtered by classification.
def get_prediction_count(classification=None):
    query = "SELECT COUNT(*) FROM public.predictions"
    parameters = []

    if classification is not None:
        if classification not in ("Positive", "Negative"):
            raise ValueError("Invalid classification filter.")

        query += " WHERE classification = %s"
        parameters.append(classification)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchone()[0]
