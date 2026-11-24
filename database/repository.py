
from datetime import datetime, time, timedelta, timezone
from uuid import UUID

from psycopg.rows import dict_row

from database.connection import get_connection


# Match database columns with prediction form fields.
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


PREDICTION_COLUMNS = [
    *FEATURE_COLUMNS.values(),
    "probability",
    "classification",
    "threshold",
    "model_name",
    "model_version",
    "source",
    "batch_id",
    "batch_row_number",
]


# Validate and normalize a batch UUID.
def _normalize_batch_id(batch_id):
    try:
        return UUID(str(batch_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid batch ID.") from None


# Validate and prepare prediction values for database insertion.
def _prepare_prediction_values(
    measurements,
    probability,
    classification,
    threshold=0.40,
    model_name="Random Forest",
    model_version="v1.0",
    source="individual",
    batch_id=None,
    batch_row_number=None,
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
        name for name in FEATURE_COLUMNS
        if name not in measurements
    ]

    if missing:
        raise ValueError(f"Missing measurements: {', '.join(missing)}")

    if source not in ("individual", "batch"):
        raise ValueError("Invalid prediction source.")

    if source == "individual":
        if batch_id is not None or batch_row_number is not None:
            raise ValueError(
                "Individual predictions cannot contain batch information."
            )

    if source == "batch":
        if batch_id is None or batch_row_number is None:
            raise ValueError(
                "Batch predictions require a batch ID and row number."
            )

        batch_id = _normalize_batch_id(batch_id)

        if isinstance(batch_row_number, bool):
            raise ValueError("Invalid batch row number.")

        row_number = int(batch_row_number)

        if row_number != float(batch_row_number) or row_number < 2:
            raise ValueError(
                "Batch row number must be a whole number greater than 1."
            )

        batch_row_number = row_number

    values = [
        measurements[name]
        for name in FEATURE_COLUMNS
    ]

    values.extend([
        probability,
        classification,
        threshold,
        model_name,
        model_version,
        source,
        batch_id,
        batch_row_number,
    ])

    return values


# Build reusable filters for prediction retrieval and counting.
def _build_filters(
    classification=None,
    search_id=None,
    start_date=None,
    end_date=None,
    source=None,
    batch_id=None,
):
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

    if start_date is not None and end_date is not None:
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date.")

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

    if source is not None:
        if source not in ("individual", "batch"):
            raise ValueError("Invalid prediction source filter.")

        conditions.append("source = %s")
        parameters.append(source)

    if batch_id is not None:
        if source == "individual":
            raise ValueError(
                "A batch ID cannot be used with the individual source."
            )

        conditions.append("batch_id = %s")
        parameters.append(_normalize_batch_id(batch_id))

    where_sql = (
        "WHERE " + " AND ".join(conditions)
        if conditions
        else ""
    )

    return where_sql, parameters


# Save an individual prediction and return its database ID.
def save_prediction(
    measurements,
    probability,
    classification,
    threshold=0.40,
    model_name="Random Forest",
    model_version="v1.0",
):
    values = _prepare_prediction_values(
        measurements=measurements,
        probability=probability,
        classification=classification,
        threshold=threshold,
        model_name=model_name,
        model_version=model_version,
        source="individual",
    )

    column_sql = ", ".join(PREDICTION_COLUMNS)
    placeholders = ", ".join(["%s"] * len(values))

    query = f"""
        INSERT INTO public.predictions ({column_sql})
        VALUES ({placeholders})
        RETURNING id;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, values)
            prediction_id = cursor.fetchone()[0]

    return prediction_id


# Save an entire batch in a single database transaction.
def save_batch_predictions(
    predictions,
    batch_id,
    threshold=0.40,
    model_name="Random Forest",
    model_version="v1.0",
):
    predictions = list(predictions)

    if not predictions:
        raise ValueError("No predictions available to save.")

    batch_id = _normalize_batch_id(batch_id)

    prepared_rows = []
    row_numbers = set()

    for prediction in predictions:
        values = _prepare_prediction_values(
            measurements=prediction["measurements"],
            probability=prediction["probability"],
            classification=prediction["classification"],
            threshold=threshold,
            model_name=model_name,
            model_version=model_version,
            source="batch",
            batch_id=batch_id,
            batch_row_number=prediction["batch_row_number"],
        )

        row_number = values[-1]

        if row_number in row_numbers:
            raise ValueError(
                f"Duplicate row number in batch: {row_number}."
            )

        row_numbers.add(row_number)
        prepared_rows.append(values)

    column_sql = ", ".join(PREDICTION_COLUMNS)
    placeholders = ", ".join(["%s"] * len(PREDICTION_COLUMNS))

    query = f"""
        INSERT INTO public.predictions ({column_sql})
        VALUES ({placeholders})
        ON CONFLICT (batch_id, batch_row_number)
        WHERE source = 'batch'
        DO NOTHING
        RETURNING id;
    """

    saved_ids = []

    with get_connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                for values in prepared_rows:
                    cursor.execute(query, values)
                    result = cursor.fetchone()

                    if result is not None:
                        saved_ids.append(result[0])

                if saved_ids and len(saved_ids) != len(prepared_rows):
                    raise ValueError(
                        "Batch contains previously saved rows. "
                        "No new records were saved."
                    )

    return {
        "batch_id": str(batch_id),
        "total_records": len(prepared_rows),
        "saved_count": len(saved_ids),
        "skipped_count": len(prepared_rows) - len(saved_ids),
        "prediction_ids": saved_ids,
        "already_saved": len(saved_ids) == 0,
    }


# Retrieve predictions with optional filters and pagination.
def get_all_predictions(
    classification=None,
    search_id=None,
    start_date=None,
    end_date=None,
    limit=50,
    offset=0,
    source=None,
    batch_id=None,
):
    limit = int(limit)
    offset = int(offset)

    if not 1 <= limit <= 500:
        raise ValueError("Limit must be between 1 and 500.")

    if offset < 0:
        raise ValueError("Offset cannot be negative.")

    where_sql, parameters = _build_filters(
        classification=classification,
        search_id=search_id,
        start_date=start_date,
        end_date=end_date,
        source=source,
        batch_id=batch_id,
    )

    query = f"""
        SELECT *
        FROM public.predictions
        {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT %s OFFSET %s;
    """

    parameters.extend([limit, offset])

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


# Count predictions using the same filters as prediction retrieval.
def get_prediction_count(
    classification=None,
    search_id=None,
    start_date=None,
    end_date=None,
    source=None,
    batch_id=None,
):
    where_sql, parameters = _build_filters(
        classification=classification,
        search_id=search_id,
        start_date=start_date,
        end_date=end_date,
        source=source,
        batch_id=batch_id,
    )

    query = f"""
        SELECT COUNT(*)
        FROM public.predictions
        {where_sql};
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchone()[0]


# Retrieve all predictions belonging to a specific batch.
def get_batch_predictions(batch_id):
    batch_id = _normalize_batch_id(batch_id)

    query = """
        SELECT *
        FROM public.predictions
        WHERE source = 'batch'
          AND batch_id = %s
        ORDER BY batch_row_number ASC;
    """

    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, (batch_id,))
            return cursor.fetchall()


# Retrieve saved batches with their summary statistics.
def get_available_batches():
    query = """
        SELECT
            batch_id,
            MIN(created_at) AS created_at,
            COUNT(*) AS total_predictions,
            COUNT(*) FILTER (
                WHERE classification = 'Positive'
            ) AS positive_count,
            COUNT(*) FILTER (
                WHERE classification = 'Negative'
            ) AS negative_count,
            AVG(probability) AS average_probability
        FROM public.predictions
        WHERE source = 'batch'
          AND batch_id IS NOT NULL
        GROUP BY batch_id
        ORDER BY MIN(created_at) DESC;
    """

    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query)
            return cursor.fetchall()
