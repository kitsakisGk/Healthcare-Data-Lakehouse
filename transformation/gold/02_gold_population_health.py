"""
transformation/gold/02_gold_population_health.py

Silver → Gold: Population health summary tables for Power BI reporting.

Produces:
  - condition_prevalence: top conditions by patient count and percentage
  - encounter_volume_monthly: monthly encounter counts by class
  - readmission_by_condition: 30-day readmission rate per diagnosis group
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

SILVER_BASE = "./spark-warehouse/silver"
GOLD_BASE   = "./spark-warehouse/gold"

# COMMAND ----------
# 1. Condition Prevalence

df_condition = spark.read.format("delta").load(f"{SILVER_BASE}/condition")
total_patients = spark.read.format("delta").load(f"{GOLD_BASE}/patient_summary").count()

df_condition_prev = (
    df_condition
    .filter(F.col("snomed_display").isNotNull())
    .groupBy("snomed_code", "snomed_display")
    .agg(F.countDistinct("patient_id").alias("patient_count"))
    .withColumn("prevalence_pct", F.round(F.col("patient_count") / total_patients * 100, 2))
    .orderBy(F.col("patient_count").desc())
)

(
    df_condition_prev.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/condition_prevalence")
)
print("Written: condition_prevalence")
df_condition_prev.show(20, truncate=False)

# COMMAND ----------
# 2. Monthly Encounter Volume

df_encounter = spark.read.format("delta").load(f"{SILVER_BASE}/encounter")

df_enc_volume = (
    df_encounter
    .withColumn("year_month", F.date_format("start_datetime", "yyyy-MM"))
    .groupBy("year_month", "class_code")
    .agg(F.count("*").alias("encounter_count"))
    .orderBy("year_month", "class_code")
)

(
    df_enc_volume.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/encounter_volume_monthly")
)
print("Written: encounter_volume_monthly")

# COMMAND ----------
# 3. Readmission Rate by Condition

df_patient_summary = spark.read.format("delta").load(f"{GOLD_BASE}/patient_summary")

df_readmit_condition = (
    df_condition
    .filter(F.col("clinical_status") == "active")
    .join(df_patient_summary.select("patient_id", "readmitted_30d"), on="patient_id", how="inner")
    .groupBy("snomed_code", "snomed_display")
    .agg(
        F.countDistinct("patient_id").alias("patient_count"),
        F.sum(F.when(F.col("readmitted_30d"), 1).otherwise(0)).alias("readmitted_count"),
    )
    .withColumn("readmission_rate_pct", F.round(F.col("readmitted_count") / F.col("patient_count") * 100, 2))
    .filter(F.col("patient_count") >= 10)
    .orderBy(F.col("readmission_rate_pct").desc())
)

(
    df_readmit_condition.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/readmission_by_condition")
)
print("Written: readmission_by_condition")
df_readmit_condition.show(20, truncate=False)
