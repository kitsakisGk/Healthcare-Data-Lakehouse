"""
transformation/silver/02_silver_conditions.py

Bronze → Silver: Parse FHIR Condition resources into a clean flat table.

FHIR Condition = a clinical diagnosis (e.g. "Type 2 diabetes", "Hypertension")
Coded using SNOMED CT (the international clinical terminology standard).

FHIR Condition reference:
  https://www.hl7.org/fhir/condition.html
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
import json

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

BRONZE_BASE = "./spark-warehouse/bronze"
SILVER_BASE = "./spark-warehouse/silver"

# COMMAND ----------

df_raw = spark.read.format("delta").load(f"{BRONZE_BASE}/condition")
print(f"Bronze condition rows: {df_raw.count()}")

# COMMAND ----------

condition_schema = StructType([
    StructField("condition_id", StringType()),
    StructField("patient_id", StringType()),
    StructField("encounter_id", StringType()),
    StructField("clinical_status", StringType()),
    StructField("verification_status", StringType()),
    StructField("onset_date", StringType()),
    StructField("abatement_date", StringType()),
    StructField("recorded_date", StringType()),
    StructField("snomed_code", StringType()),
    StructField("snomed_display", StringType()),
    StructField("icd10_code", StringType()),
    StructField("icd10_display", StringType()),
    StructField("category", StringType()),
])


def parse_condition(raw_json: str) -> dict:
    try:
        c = json.loads(raw_json)
    except Exception:
        return {}

    codings = c.get("code", {}).get("coding", [])
    snomed = next((cd for cd in codings if "snomed" in cd.get("system", "").lower()), {})
    icd10 = next((cd for cd in codings if "icd" in cd.get("system", "").lower()), {})

    return {
        "condition_id": c.get("id"),
        "patient_id": c.get("subject", {}).get("reference", "").replace("urn:uuid:", ""),
        "encounter_id": c.get("encounter", {}).get("reference", "").replace("urn:uuid:", ""),
        "clinical_status": c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
        "verification_status": c.get("verificationStatus", {}).get("coding", [{}])[0].get("code"),
        "onset_date": c.get("onsetDateTime"),
        "abatement_date": c.get("abatementDateTime"),
        "recorded_date": c.get("recordedDate"),
        "snomed_code": snomed.get("code"),
        "snomed_display": snomed.get("display"),
        "icd10_code": icd10.get("code"),
        "icd10_display": icd10.get("display"),
        "category": next(
            (cat.get("coding", [{}])[0].get("display") for cat in c.get("category", [])),
            None,
        ),
    }


parse_condition_udf = F.udf(parse_condition, condition_schema)

# COMMAND ----------

df_parsed = (
    df_raw
    .withColumn("parsed", parse_condition_udf(F.col("raw_json")))
    .select(
        F.col("parsed.*"),
        F.col("ingestion_timestamp"),
        F.col("batch_id"),
        F.current_timestamp().alias("silver_timestamp"),
    )
    .filter(F.col("patient_id").isNotNull() & (F.col("patient_id") != ""))
)

print(f"Silver condition rows: {df_parsed.count()}")

# COMMAND ----------

(
    df_parsed.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{SILVER_BASE}/condition")
)

print(f"Written to {SILVER_BASE}/condition")
df_parsed.select("condition_id", "patient_id", "snomed_display", "clinical_status", "onset_date").show(10)
