"""
ml/01_train_readmission_model.py

Train an XGBoost model to predict 30-day patient readmission.

Input:  gold.patient_features (built by 03_gold_patient_features.py)
Output: - Trained model logged to MLflow
        - Evaluation metrics: AUC-ROC, precision, recall, F1
        - Model saved locally for SHAP analysis

Target variable: readmitted_30d (binary: True/False)

Features used:
  Demographics:  age, gender (encoded)
  Clinical:      active_condition_count, unique_diagnosis_count
  Utilisation:   total_encounters, inpatient_count, emergency_count, avg_los_days
  Medications:   total_medication_count, active_medication_count, chronic_medication_count
  Labs:          hba1c, bmi, systolic_bp, diastolic_bp, cholesterol, egfr, blood_glucose
  Procedures:    total_procedure_count, surgical_procedure_count, has_dialysis
  Risk flags:    is_diabetic, is_obese, is_hypertensive, has_ckd, is_elderly
"""

# COMMAND ----------

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import xgboost as xgb

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

GOLD_BASE   = "./spark-warehouse/gold"
MODEL_PATH  = "./ml/models/readmission_xgb"
MLFLOW_EXPERIMENT = "healthcare-readmission-prediction"

# COMMAND ----------
# Load feature table

df_spark = spark.read.format("delta").load(f"{GOLD_BASE}/patient_features")
df = df_spark.toPandas()

print(f"Total patients: {len(df)}")
print(f"Readmission rate: {df['readmitted_30d'].mean():.2%}")

# COMMAND ----------
# Feature engineering

FEATURE_COLS = [
    "age",
    "active_condition_count",
    "unique_diagnosis_count",
    "total_encounters",
    "inpatient_count",
    "emergency_count",
    "avg_los_days",
    "total_medication_count",
    "active_medication_count",
    "chronic_medication_count",
    "hba1c",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "egfr",
    "blood_glucose",
    "total_procedure_count",
    "surgical_procedure_count",
    # Boolean features — cast to int for XGBoost
    "has_dialysis",
    "is_diabetic",
    "is_obese",
    "is_hypertensive",
    "has_ckd",
    "is_elderly",
    # Gender encoded
    "gender_encoded",
]

TARGET = "readmitted_30d"

# Encode gender
le = LabelEncoder()
df["gender_encoded"] = le.fit_transform(df["gender"].fillna("unknown"))

# Cast booleans to int
bool_cols = ["has_dialysis", "is_diabetic", "is_obese", "is_hypertensive", "has_ckd", "is_elderly"]
for col in bool_cols:
    df[col] = df[col].fillna(False).astype(int)

# Fill remaining nulls with median (lab values may be missing for some patients)
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

X = df[FEATURE_COLS]
y = df[TARGET].astype(int)

# COMMAND ----------
# Train / test split — stratified to preserve readmission rate in both sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} patients | Test: {len(X_test)} patients")
print(f"Train readmission rate: {y_train.mean():.2%}")
print(f"Test readmission rate:  {y_test.mean():.2%}")

# COMMAND ----------
# Train XGBoost with MLflow tracking

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Class imbalance — readmissions are typically ~15-20% of patients
# scale_pos_weight compensates: majority_count / minority_count
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

params = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": 42,
}

with mlflow.start_run(run_name="xgb_readmission_v1"):

    mlflow.log_params(params)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("features", FEATURE_COLS)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc    = roc_auc_score(y_test, y_pred_proba)
    prec   = precision_score(y_test, y_pred)
    rec    = recall_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)

    mlflow.log_metric("auc_roc",   auc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1",        f1)

    print(f"\n=== Model Evaluation ===")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Log model to MLflow
    mlflow.xgboost.log_model(model, "xgb_readmission_model")

    # Save locally for SHAP analysis
    import os
    os.makedirs("./ml/models", exist_ok=True)
    model.save_model(f"{MODEL_PATH}.json")
    print(f"\nModel saved to {MODEL_PATH}.json")

# COMMAND ----------
# Save predictions + scores back to Gold for Power BI

df_test_results = X_test.copy()
df_test_results["patient_id"] = df.loc[X_test.index, "patient_id"].values
df_test_results["actual_readmitted"] = y_test.values
df_test_results["readmission_risk_score"] = y_pred_proba
df_test_results["predicted_readmitted"] = y_pred

df_scores_spark = spark.createDataFrame(
    df_test_results[["patient_id", "readmission_risk_score", "predicted_readmitted", "actual_readmitted"]]
)

(
    df_scores_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/readmission_scores")
)

print(f"Readmission scores written to {GOLD_BASE}/readmission_scores")
