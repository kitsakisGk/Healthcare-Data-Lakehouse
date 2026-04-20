"""
transformation/gold/03_gold_patient_features.py

Silver → Gold: Build the ML-ready patient feature table.

Enriches the patient summary with:
  - Medication features: total count, active count, chronic count
  - Lab features: latest HbA1c, BMI, systolic BP, diastolic BP, cholesterol
  - Procedure features: total count, surgical count, dialysis flag
  - Derived risk flags: diabetic, hypertensive, obese, chronic_kidney_disease

This is the final input table for the XGBoost readmission model.
One row per patient, all numeric/boolean features, no PII.

Depends on:
  - gold.patient_summary (01_gold_patient_summary.py)
  - silver.medication
  - silver.observation
  - silver.procedure
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

SILVER_BASE = "./spark-warehouse/silver"
GOLD_BASE   = "./spark-warehouse/gold"

# COMMAND ----------
# Load tables

df_summary    = spark.read.format("delta").load(f"{GOLD_BASE}/patient_summary")
df_medication = spark.read.format("delta").load(f"{SILVER_BASE}/medication")
df_observation = spark.read.format("delta").load(f"{SILVER_BASE}/observation")
df_procedure  = spark.read.format("delta").load(f"{SILVER_BASE}/procedure")

# COMMAND ----------
# Medication features

df_med_features = (
    df_medication
    .groupBy("patient_id")
    .agg(
        F.countDistinct("medication_id").alias("total_medication_count"),
        F.sum(F.when(F.col("status") == "active", 1).otherwise(0)).alias("active_medication_count"),
        F.sum(F.when(F.col("is_chronic"), 1).otherwise(0)).alias("chronic_medication_count"),
        F.countDistinct("rxnorm_code").alias("unique_medication_count"),
    )
)

# COMMAND ----------
# Lab / vital sign features
# We want the LATEST value per patient for each key LOINC code
#
# Key LOINC codes:
#   4548-4  = HbA1c (%)              — diabetes marker
#   39156-5 = BMI (kg/m2)            — obesity marker
#   8480-6  = Systolic BP (mmHg)
#   8462-4  = Diastolic BP (mmHg)
#   2093-3  = Total cholesterol
#   33914-3 = eGFR                   — kidney function
#   2345-7  = Blood glucose

KEY_LOINC = {
    "4548-4":  "hba1c",
    "39156-5": "bmi",
    "8480-6":  "systolic_bp",
    "8462-4":  "diastolic_bp",
    "2093-3":  "cholesterol",
    "33914-3": "egfr",
    "2345-7":  "blood_glucose",
}

w_latest = Window.partitionBy("patient_id", "loinc_code").orderBy(F.col("effective_datetime").desc())

df_obs_latest = (
    df_observation
    .filter(F.col("loinc_code").isin(list(KEY_LOINC.keys())))
    .filter(F.col("value_quantity").isNotNull())
    .withColumn("row_num", F.row_number().over(w_latest))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
)

# Pivot so each LOINC becomes a column
df_obs_pivot = (
    df_obs_latest
    .groupBy("patient_id")
    .pivot("loinc_code", list(KEY_LOINC.keys()))
    .agg(F.first("value_quantity"))
)

# Rename columns from LOINC codes to readable names
for loinc_code, col_name in KEY_LOINC.items():
    if loinc_code in df_obs_pivot.columns:
        df_obs_pivot = df_obs_pivot.withColumnRenamed(loinc_code, col_name)

# COMMAND ----------
# Procedure features

df_proc_features = (
    df_procedure
    .groupBy("patient_id")
    .agg(
        F.count("*").alias("total_procedure_count"),
        # Surgical procedures — SNOMED codes for surgery-related concepts
        F.sum(
            F.when(
                F.lower(F.col("snomed_display")).rlike("surgery|surgical|operation|excision|repair|replacement"),
                1
            ).otherwise(0)
        ).alias("surgical_procedure_count"),
        # Dialysis flag — strong readmission risk indicator
        F.max(
            F.when(
                F.lower(F.col("snomed_display")).rlike("dialysis|hemodialysis|haemodialysis"),
                True
            ).otherwise(False)
        ).alias("has_dialysis"),
    )
)

# COMMAND ----------
# Join everything onto the patient summary

df_features = (
    df_summary.select(
        "patient_id", "age", "gender", "country",
        "active_condition_count", "unique_diagnosis_count",
        "total_encounters", "inpatient_count", "emergency_count",
        "avg_los_days", "readmitted_30d",
    )
    .join(df_med_features, on="patient_id", how="left")
    .join(df_obs_pivot,    on="patient_id", how="left")
    .join(df_proc_features, on="patient_id", how="left")
    .fillna({
        "total_medication_count": 0,
        "active_medication_count": 0,
        "chronic_medication_count": 0,
        "unique_medication_count": 0,
        "total_procedure_count": 0,
        "surgical_procedure_count": 0,
        "has_dialysis": False,
    })
)

# COMMAND ----------
# Derived risk flags — useful both as features and for Power BI segmentation

df_features = (
    df_features
    .withColumn("is_diabetic",       F.col("hba1c") >= 6.5)
    .withColumn("is_obese",          F.col("bmi") >= 30.0)
    .withColumn("is_hypertensive",   F.col("systolic_bp") >= 140)
    .withColumn("has_ckd",           F.col("egfr") < 60)        # Chronic Kidney Disease threshold
    .withColumn("is_elderly",        F.col("age") >= 65)
    .withColumn("gold_timestamp",    F.current_timestamp())
)

# COMMAND ----------

print(f"Gold patient_features rows: {df_features.count()}")
print(f"Readmission rate: {df_features.filter(F.col('readmitted_30d')).count() / df_features.count():.2%}")
print(f"Diabetic patients: {df_features.filter(F.col('is_diabetic')).count()}")
print(f"Obese patients: {df_features.filter(F.col('is_obese')).count()}")
print(f"Dialysis patients: {df_features.filter(F.col('has_dialysis')).count()}")

# COMMAND ----------

(
    df_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/patient_features")
)

print(f"Written to {GOLD_BASE}/patient_features")

df_features.select(
    "patient_id", "age", "gender", "active_condition_count",
    "hba1c", "bmi", "systolic_bp", "total_medication_count",
    "is_diabetic", "is_obese", "readmitted_30d"
).show(10)
