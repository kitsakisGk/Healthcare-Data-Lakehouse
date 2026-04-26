"""
ml/02_explain_shap.py

SHAP explainability for the XGBoost readmission model.

What this does:
  - Loads the trained XGBoost model
  - Computes SHAP values for the test set
  - Produces: feature importance plot, beeswarm plot, waterfall for top-risk patients
  - Saves a Gold table: patient_id + top 3 risk factors per patient (for Power BI)

SHAP (SHapley Additive exPlanations) tells you WHY the model made each prediction.
For example: "this patient is high risk because of high HbA1c, frequent inpatient
admissions, and active dialysis."

That explainability is critical for healthcare — clinicians won't trust a black box.

Depends on: ml/01_train_readmission_model.py having run first.
"""

# COMMAND ----------

import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

GOLD_BASE  = "./spark-warehouse/gold"
MODEL_PATH = "./ml/models/readmission_xgb.json"
PLOTS_PATH = "./docs/shap_plots"

os.makedirs(PLOTS_PATH, exist_ok=True)

FEATURE_COLS = [
    "age", "active_condition_count", "unique_diagnosis_count",
    "total_encounters", "inpatient_count", "emergency_count", "avg_los_days",
    "total_medication_count", "active_medication_count", "chronic_medication_count",
    "hba1c", "bmi", "systolic_bp", "diastolic_bp", "cholesterol", "egfr", "blood_glucose",
    "total_procedure_count", "surgical_procedure_count",
    "has_dialysis", "is_diabetic", "is_obese", "is_hypertensive", "has_ckd", "is_elderly",
    "gender_encoded",
]

# COMMAND ----------
# Load model and feature data

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print("Model loaded.")

df_spark = spark.read.format("delta").load(f"{GOLD_BASE}/patient_features")
df = df_spark.toPandas()

# Reproduce the same feature prep as training
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["gender_encoded"] = le.fit_transform(df["gender"].fillna("unknown"))

bool_cols = ["has_dialysis", "is_diabetic", "is_obese", "is_hypertensive", "has_ckd", "is_elderly"]
for col in bool_cols:
    df[col] = df[col].fillna(False).astype(int)

df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
X = df[FEATURE_COLS]

# COMMAND ----------
# Compute SHAP values

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print(f"SHAP values shape: {shap_values.shape}")

# COMMAND ----------
# Plot 1 — Global feature importance (mean |SHAP|)

shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Feature Importance — Mean |SHAP Value|")
plt.tight_layout()
plt.savefig(f"{PLOTS_PATH}/feature_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_PATH}/feature_importance_bar.png")

# COMMAND ----------
# Plot 2 — Beeswarm (shows direction + magnitude of each feature's impact)

shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Beeswarm — Feature Impact on Readmission Risk")
plt.tight_layout()
plt.savefig(f"{PLOTS_PATH}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_PATH}/shap_beeswarm.png")

# COMMAND ----------
# Plot 3 — Waterfall for the highest-risk patient

risk_scores = model.predict_proba(X)[:, 1]
highest_risk_idx = np.argmax(risk_scores)

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[highest_risk_idx],
        base_values=explainer.expected_value,
        data=X.iloc[highest_risk_idx],
        feature_names=FEATURE_COLS,
    ),
    show=False,
)
plt.title(f"Highest Risk Patient — Risk Score: {risk_scores[highest_risk_idx]:.2%}")
plt.tight_layout()
plt.savefig(f"{PLOTS_PATH}/waterfall_highest_risk.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_PATH}/waterfall_highest_risk.png")

# COMMAND ----------
# Build per-patient top risk factors table for Power BI
# For each patient: their top 3 features driving readmission risk

shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
shap_df["patient_id"] = df["patient_id"].values
shap_df["readmission_risk_score"] = risk_scores

def top_n_factors(row, n=3):
    """Return the top N features by absolute SHAP value as a comma-separated string."""
    feature_shap = {col: abs(row[col]) for col in FEATURE_COLS}
    top = sorted(feature_shap, key=feature_shap.get, reverse=True)[:n]
    return ", ".join(top)

shap_df["top_risk_factors"] = shap_df.apply(top_n_factors, axis=1)

df_risk_factors = shap_df[["patient_id", "readmission_risk_score", "top_risk_factors"]]

# COMMAND ----------
# Write to Gold

schema = StructType([
    StructField("patient_id", StringType()),
    StructField("readmission_risk_score", DoubleType()),
    StructField("top_risk_factors", StringType()),
])

df_risk_spark = spark.createDataFrame(df_risk_factors, schema=schema)

(
    df_risk_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/patient_risk_factors")
)

print(f"Written to {GOLD_BASE}/patient_risk_factors")

# Preview top 10 highest risk patients
df_risk_factors.sort_values("readmission_risk_score", ascending=False).head(10)
