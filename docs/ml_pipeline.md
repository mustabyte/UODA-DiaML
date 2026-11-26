# Machine Learning Pipeline — UODA DiaML

## 1. Overview

UODA DiaML uses a supervised machine learning classification pipeline to predict diabetes outcomes based on eight health-related measurements.

The pipeline integrates dataset preprocessing, model training, performance evaluation, model persistence, and prediction functionality.

## 2. Dataset Description

The project uses the Pima Indians Diabetes Dataset.

- Total records: 768
- Input features: 8
- Target variable: Outcome
- Negative records: 500
- Positive records: 268
- Problem type: Binary classification

### Input Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skinfold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes hereditary indicator |
| Age | Age in years |

### Target Variable

- 0: Negative classification
- 1: Positive classification

## 3. Data Preprocessing

The preprocessing pipeline prepares the dataset for machine learning by handling invalid measurements and maintaining consistent feature ordering.

Zero values are treated as missing measurements in the following features:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

Missing measurements are handled using median imputation.

Preprocessing is integrated into the machine learning pipeline to ensure consistency between training and prediction.

## 4. Dataset Splitting

The dataset is divided into training and testing subsets.

| Subset | Records |
|---|---:|
| Training | 614 |
| Testing | 154 |
| Total | 768 |

The testing subset is used to evaluate model performance on unseen records.

## 5. Model Training and Comparison

Three classification algorithms were evaluated during model development.

| Model | Cross-Validation F1 Score |
|---|---:|
| Logistic Regression | 0.6823 |
| Support Vector Machine | 0.6902 |
| Random Forest | 0.6980 |

Random Forest was selected because it achieved the highest cross-validation F1 score among the evaluated models.

## 6. Random Forest Configuration

The selected Random Forest classifier uses the following hyperparameters:

| Parameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 5 |
| min_samples_split | 5 |
| min_samples_leaf | 1 |
| class_weight | balanced |
| random_state | 42 |

Balanced class weights help address the unequal distribution of positive and negative records.

## 7. Classification Threshold

The application uses a classification threshold of 0.40.

- Probability greater than or equal to 0.40: Positive classification
- Probability below 0.40: Negative classification

The selected threshold prioritizes recall, reducing the number of missed positive cases while accepting a higher number of false-positive predictions.

## 8. Model Evaluation

The selected model achieved the following results on the testing dataset.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 72.73% |
| Precision | 57.32% |
| Recall | 87.04% |
| F1 Score | 69.12% |
| ROC-AUC | 82.20% |

### Confusion Matrix

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 65 | 35 |
| Positive | 7 | 47 |

- True Negatives: 65
- False Positives: 35
- False Negatives: 7
- True Positives: 47

The model demonstrates relatively high recall, although its precision indicates that false-positive predictions remain a limitation.

### Evaluation Techniques

The evaluation process includes:

- Accuracy analysis
- Precision analysis
- Recall analysis
- F1 score calculation
- Confusion matrix analysis
- ROC curve analysis
- ROC-AUC calculation

## 9. Feature Importance

Random Forest provides feature importance scores that indicate the relative contribution of individual features to the model's predictions.

Feature importance analysis helps improve model interpretability.

However, these scores represent model-specific relationships and should not be interpreted as evidence of medical causation.

## 10. Model Persistence

The trained pipeline and model configuration are saved using Joblib.

| File | Purpose |
|---|---|
| models/random_forest_pipeline.joblib | Stores the trained prediction pipeline |
| models/model_config.joblib | Stores model configuration and classification settings |

Model persistence allows the application to generate predictions without retraining the classifier every time it starts.

## 11. ML Module Structure

| File | Responsibility |
|---|---|
| ml/preprocessing.py | Dataset preprocessing and feature preparation |
| ml/train_models.py | Model training and comparison |
| ml/evaluate_models.py | Model evaluation and performance metrics |
| ml/predict.py | Model loading and individual prediction |
| ml/batch_predict.py | Batch prediction processing |

## 12. Development Notebooks

The machine learning development process is organized into three Jupyter notebooks.

| Notebook | Purpose |
|---|---|
| 01_dataset_exploration.ipynb | Dataset exploration and initial analysis |
| 02_data_preprocessing.ipynb | Data cleaning and preprocessing |
| 03_model_training.ipynb | Model training, comparison, and evaluation |

## 13. Automated Testing

Automated tests are used to verify the reliability of the machine learning pipeline.

The testing process covers:

- Saved model loading
- Model configuration validation
- Missing-value imputation
- Feature ordering
- Input validation
- Prediction generation
- Classification threshold behavior
- Evaluation metric calculation
- Confusion matrix generation
- ROC curve generation

Tests can be executed from the project root using:

```bash
python -m pytest tests/test_ml_pipeline.py tests/test_model_evaluation.py -v
```

## 14. Model Limitations

- The dataset contains a relatively small number of records.
- The dataset represents a specific population.
- Predictions may not generalize to other populations.
- False-positive and false-negative predictions remain possible.
- Model probabilities are not clinically validated risk estimates.
- The system is intended for educational purposes rather than medical diagnosis.

## 15. Future Improvements

Potential improvements include:

- Training with larger and more diverse datasets.
- Evaluating additional classification algorithms.
- Improving probability calibration.
- Implementing advanced model explainability techniques.
- Performing external dataset validation.

## 16. Conclusion

The UODA DiaML machine learning pipeline provides a structured approach to diabetes classification through consistent preprocessing, model evaluation, automated testing, and reusable prediction components.

The trained Random Forest pipeline serves as the machine learning foundation of the UODA DiaML application.

---

**Project:** UODA DiaML

**Group:** Group 6

**Technical Documentation:** Mehedi Hasan Rafid