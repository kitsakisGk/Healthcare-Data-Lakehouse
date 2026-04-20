"""
transformation/gold/01_gold_patient_summary.py

Silver → Gold: Build the patient summary table.

One row per patient with all features needed for the ML readmission model
and for Power BI reporting.

Columns produced:
  - Demographics: age, gender, country
  - Clinical: chronic_condition_count, active_condition_count, condition_list
  - Utilisation: total_encounters, inpatient_count, emergency_count, avg_los_days
  - Readmission flag: readmitted_30d (target variable for ML)
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

SILVER_BASE = "./spark-warehouse/silver"
GOLD_BASE = "./spark-warehouse/gold"

# COMMAND ----------

df_patient = spark.read.format("delta").load(f"{SILVER_BASE}/patient")
df_condition = spark.read.format("delta").load(f"{SILVER_BASE}/condition")
df_encounter = spark.read.format("delta").load(f"{SILVER_BASE}/encounter")

# COMMAND ----------
# Demographics

df_patient_base = df_patient.select(
    "patient_id", "birth_date", "gender", "country", "deceased", "deceased_date", "race", "ethnicity"
).withColumn(
    "age",
    F.floor(F.months_between(F.current_date(), F.to_date("birth_date", "yyyy-MM-dd")) / 12).cast("int"),
)

# COMMAND ----------
# Condition aggregations

df_conditions_agg = (
    df_condition
    .groupBy("patient_id")
    .agg(
        F.count("*").alias("total_condition_count"),
        F.sum(F.when(F.col("clinical_status") == "active", 1).otherwise(0)).alias("active_condition_count"),
        F.collect_set("snomed_display").alias("condition_list"),
        F.countDistinct("snomed_code").alias("unique_diagnosis_count"),
    )
)

# COMMAND ----------
# Encounter aggregations

df_encounters_agg = (
    df_encounter
    .withColumn(
        "los_hours",
        (F.unix_timestamp("end_datetime") - F.unix_timestamp("start_datetime")) / 3600,
    )
    .groupBy("patient_id")
    .agg(
        F.count("*").alias("total_encounters"),
        F.sum(F.when(F.col("class_code") == "IMP", 1).otherwise(0)).alias("inpatient_count"),
        F.sum(F.when(F.col("class_code") == "EMER", 1).otherwise(0)).alias("emergency_count"),
        F.sum(F.when(F.col("class_code") == "AMB", 1).otherwise(0)).alias("outpatient_count"),
        F.avg("los_hours").alias("avg_los_hours"),
        F.max("start_datetime").alias("last_encounter_date"),
        F.min("start_datetime").alias("first_encounter_date"),
    )
    .withColumn("avg_los_days", F.round(F.col("avg_los_hours") / 24, 2))
    .drop("avg_los_hours")
)

# COMMAND ----------
# 30-day readmission flag

w_patient_time = Window.partitionBy("patient_id").orderBy("start_datetime")

df_inpatient = (
    df_encounter
    .filter(F.col("class_code") == "IMP")
    .withColumn("next_admit", F.lead("start_datetime").over(w_patient_time))
    .withColumn("days_to_readmit", F.datediff(F.col("next_admit"), F.col("end_datetime")))
    .withColumn("readmitted_30d_flag", F.col("days_to_readmit") <= 30)
)

df_readmission = (
    df_inpatient
    .groupBy("patient_id")
    .agg(
        F.max("readmitted_30d_flag").alias("readmitted_30d"),
        F.min(F.when(F.col("readmitted_30d_flag"), F.col("days_to_readmit"))).alias("min_readmit_days"),
    )
)

# COMMAND ----------
# Join everything

df_gold = (
    df_patient_base
    .join(df_conditions_agg, on="patient_id", how="left")
    .join(df_encounters_agg, on="patient_id", how="left")
    .join(df_readmission, on="patient_id", how="left")
    .fillna({
        "total_condition_count": 0,
        "active_condition_count": 0,
        "unique_diagnosis_count": 0,
        "total_encounters": 0,
        "inpatient_count": 0,
        "emergency_count": 0,
        "outpatient_count": 0,
        "readmitted_30d": False,
    })
    .withColumn("gold_timestamp", F.current_timestamp())
)

# COMMAND ----------

(
    df_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/patient_summary")
)

print(f"Gold patient_summary written. Rows: {df_gold.count()}")
print(f"Readmission rate: {df_gold.filter(F.col('readmitted_30d')).count() / df_gold.count():.2%}")

df_gold.select(
    "patient_id", "age", "gender", "active_condition_count",
    "total_encounters", "inpatient_count", "readmitted_30d"
).show(10)
