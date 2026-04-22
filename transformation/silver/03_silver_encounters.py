"""
transformation/silver/03_silver_encounters.py

Bronze → Silver: Parse FHIR Encounter resources.

FHIR Encounter = a hospital visit, outpatient appointment, emergency admission, etc.
Critical for the Gold layer 30-day readmission KPI.

FHIR Encounter reference:
  https://www.hl7.org/fhir/encounter.html
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

df_raw = spark.read.format("delta").load(f"{BRONZE_BASE}/encounter")
print(f"Bronze encounter rows: {df_raw.count()}")

# COMMAND ----------

encounter_schema = StructType([
    StructField("encounter_id", StringType()),
    StructField("patient_id", StringType()),
    StructField("status", StringType()),
    StructField("class_code", StringType()),
    StructField("type_display", StringType()),
    StructField("start_datetime", StringType()),
    StructField("end_datetime", StringType()),
    StructField("service_provider", StringType()),
    StructField("reason_snomed", StringType()),
    StructField("reason_display", StringType()),
    StructField("discharge_disposition", StringType()),
])


def parse_encounter(raw_json: str) -> dict:
    try:
        e = json.loads(raw_json)
    except Exception:
        return {}

    period = e.get("period", {})
    enc_class = e.get("class", {})
    type_coding = e.get("type", [{}])[0].get("coding", [{}])[0] if e.get("type") else {}
    reason_coding = (
        e.get("reasonCode", [{}])[0].get("coding", [{}])[0]
        if e.get("reasonCode") else {}
    )

    return {
        "encounter_id": e.get("id"),
        "patient_id": e.get("subject", {}).get("reference", "").replace("urn:uuid:", ""),
        "status": e.get("status"),
        "class_code": enc_class.get("code") if isinstance(enc_class, dict) else None,
        "type_display": type_coding.get("display"),
        "start_datetime": period.get("start"),
        "end_datetime": period.get("end"),
        "service_provider": e.get("serviceProvider", {}).get("display"),
        "reason_snomed": reason_coding.get("code"),
        "reason_display": reason_coding.get("display"),
        "discharge_disposition": (
            e.get("hospitalization", {}).get("dischargeDisposition", {}).get("coding", [{}])[0].get("display")
        ),
    }


parse_encounter_udf = F.udf(parse_encounter, encounter_schema)

# COMMAND ----------

df_parsed = (
    df_raw
    .withColumn("parsed", parse_encounter_udf(F.col("raw_json")))
    .select(
        F.col("parsed.*"),
        F.col("ingestion_timestamp"),
        F.col("batch_id"),
        F.current_timestamp().alias("silver_timestamp"),
    )
    .filter(F.col("patient_id").isNotNull() & (F.col("patient_id") != ""))
    .withColumn("start_datetime", F.to_timestamp("start_datetime"))
    .withColumn("end_datetime", F.to_timestamp("end_datetime"))
)

print(f"Silver encounter rows: {df_parsed.count()}")

# COMMAND ----------

(
    df_parsed.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{SILVER_BASE}/encounter")
)

print(f"Written to {SILVER_BASE}/encounter")
df_parsed.select("encounter_id", "patient_id", "class_code", "start_datetime", "end_datetime").show(10)
