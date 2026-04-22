"""
transformation/silver/04_silver_medications.py

Bronze → Silver: Parse FHIR MedicationRequest resources.

FHIR MedicationRequest = a prescription or medication order.
Key for ML features: medication count per patient, active medications,
medication classes (e.g. insulin, antihypertensives).

FHIR MedicationRequest reference:
  https://www.hl7.org/fhir/medicationrequest.html
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BooleanType
import json

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

BRONZE_BASE = "./spark-warehouse/bronze"
SILVER_BASE = "./spark-warehouse/silver"

# COMMAND ----------

df_raw = spark.read.format("delta").load(f"{BRONZE_BASE}/medicationrequest")
print(f"Bronze medicationrequest rows: {df_raw.count()}")

# COMMAND ----------

medication_schema = StructType([
    StructField("medication_id", StringType()),
    StructField("patient_id", StringType()),
    StructField("encounter_id", StringType()),
    StructField("status", StringType()),           # active, completed, stopped, cancelled
    StructField("intent", StringType()),           # order, plan, proposal
    StructField("authored_on", StringType()),
    StructField("rxnorm_code", StringType()),      # RxNorm — standard medication coding system
    StructField("rxnorm_display", StringType()),
    StructField("dosage_text", StringType()),
    StructField("route", StringType()),            # oral, intravenous, etc.
    StructField("frequency", StringType()),
    StructField("prescriber_id", StringType()),
    StructField("reason_snomed", StringType()),
    StructField("reason_display", StringType()),
    StructField("is_chronic", BooleanType()),      # derived: still active after 90 days
])


def parse_medication(raw_json: str) -> dict:
    try:
        m = json.loads(raw_json)
    except Exception:
        return {}

    # Medication coding — Synthea uses RxNorm
    codings = m.get("medicationCodeableConcept", {}).get("coding", [])
    rxnorm = next((c for c in codings if "rxnorm" in c.get("system", "").lower()), {})

    # Dosage
    dosage = next(iter(m.get("dosageInstruction", [])), {})
    route = dosage.get("route", {}).get("coding", [{}])[0].get("display")
    timing = dosage.get("timing", {}).get("code", {}).get("text")

    # Reason
    reason_coding = (
        m.get("reasonCode", [{}])[0].get("coding", [{}])[0]
        if m.get("reasonCode") else {}
    )

    authored_on = m.get("authoredOn")

    return {
        "medication_id": m.get("id"),
        "patient_id": m.get("subject", {}).get("reference", "").replace("urn:uuid:", ""),
        "encounter_id": m.get("encounter", {}).get("reference", "").replace("urn:uuid:", ""),
        "status": m.get("status"),
        "intent": m.get("intent"),
        "authored_on": authored_on,
        "rxnorm_code": rxnorm.get("code"),
        "rxnorm_display": rxnorm.get("display"),
        "dosage_text": dosage.get("text"),
        "route": route,
        "frequency": timing,
        "prescriber_id": m.get("requester", {}).get("reference", "").replace("urn:uuid:", ""),
        "reason_snomed": reason_coding.get("code"),
        "reason_display": reason_coding.get("display"),
        "is_chronic": None,  # derived below after parsing
    }


parse_medication_udf = F.udf(parse_medication, medication_schema)

# COMMAND ----------

df_parsed = (
    df_raw
    .withColumn("parsed", parse_medication_udf(F.col("raw_json")))
    .select(
        F.col("parsed.*"),
        F.col("ingestion_timestamp"),
        F.col("batch_id"),
        F.current_timestamp().alias("silver_timestamp"),
    )
    .filter(F.col("patient_id").isNotNull() & (F.col("patient_id") != ""))
    .withColumn("authored_on", F.to_date("authored_on"))
)

# Derive is_chronic: medication authored more than 90 days ago and still active
df_parsed = df_parsed.withColumn(
    "is_chronic",
    (F.col("status") == "active") &
    (F.datediff(F.current_date(), F.col("authored_on")) > 90)
)

print(f"Silver medication rows: {df_parsed.count()}")
print(f"Active medications: {df_parsed.filter(F.col('status') == 'active').count()}")
print(f"Chronic medications: {df_parsed.filter(F.col('is_chronic')).count()}")

# COMMAND ----------

(
    df_parsed.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{SILVER_BASE}/medication")
)

print(f"Written to {SILVER_BASE}/medication")
df_parsed.select("medication_id", "patient_id", "rxnorm_display", "status", "authored_on", "is_chronic").show(10)
