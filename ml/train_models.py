
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.preprocessing import (
    load_dataset,
    create_model_pipeline,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


# Define the model configuration
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42
}

CLASSIFICATION_THRESHOLD = 0.40

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


# Train the selected Random Forest pipeline
def train_model():
    # Load the original diabetes dataset
    df = load_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Reproduce the original stratified training split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Initialize Random Forest using the selected settings
    classifier = RandomForestClassifier(**RANDOM_FOREST_PARAMS)

    # Combine preprocessing and classification
    pipeline = create_model_pipeline(
        model=classifier,
        scale=False
    )

    # Train using only the training records
    pipeline.fit(X_train, y_train)

    return pipeline


# Save the trained pipeline and its configuration
def save_model(pipeline):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "random_forest_pipeline.joblib"
    config_path = MODELS_DIR / "model_config.joblib"

    # Save the complete preprocessing and classification pipeline
    joblib.dump(pipeline, model_path)

    # Save the prediction settings needed by the application
    config = {
        "model_name": "Random Forest",
        "threshold": CLASSIFICATION_THRESHOLD,
        "feature_names": FEATURE_COLUMNS,
        "positive_class": 1
    }

    joblib.dump(config, config_path)

    print(f"Model saved: {model_path}")
    print(f"Configuration saved: {config_path}")


# Run training when this file is executed directly
if __name__ == "__main__":
    trained_pipeline = train_model()
    save_model(trained_pipeline)

    print("Random Forest training completed successfully.")
