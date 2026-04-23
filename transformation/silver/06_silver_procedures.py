"""
transformation/silver/06_silver_procedures.py

Bronze → Silver: Parse FHIR Procedure resources.

FHIR Procedure = a clinical action performed on a patient.
Examples: surgeries, dialysis sessions, imaging, physiotherapy.
Coded using SNOMED CT.

Useful ML features: procedure count, surgical history, dialysis flag
(dialysis patients have very high readmission risk).

FHIR Procedure reference:
  https://www.hl7.org/fhir/procedure.html
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

df_raw = spark.read.format("delta").load(f"{BRONZE_BASE}/procedure")
print(f"Bronze procedure rows: {df_raw.count()}")

# COMMAND ----------

procedure_schema = StructType([
    StructField("procedure_id", StringType()),
    StructField("patient_id", StringType()),
    StructField("encounter_id", StringType()),
    StructField("status", StringType()),           # completed, in-progress, not-done
    StructField("snomed_code", StringType()),
    StructField("snomed_display", StringType()),
    StructField("performed_start", StringType()),
    StructField("performed_end", StringType()),
    StructField("reason_snomed", StringType()),
    StructField("reason_display", StringType()),
    StructField("body_site", StringType()),
    StructField("outcome", StringType()),
])


def parse_procedure(raw_json: str) -> dict:
    try:
        p = json.loads(raw_json)
    except Exception:
        return {}

    codings = p.get("code", {}).get("coding", [])
    snomed = next((c for c in codings if "snomed" in c.get("system", "").lower()), codings[0] if codings else {})

    # performedPeriod or performedDateTime
    performed = p.get("performedPeriod", {})
    performed_start = performed.get("start") or p.get("performedDateTime")
    performed_end = performed.get("end")

    reason_coding = (
        p.get("reasonCode", [{}])[0].get("coding", [{}])[0]
        if p.get("reasonCode") else {}
    )

    body_site = (
        p.get("bodySite", [{}])[0].get("coding", [{}])[0].get("display")
        if p.get("bodySite") else None
    )

    outcome = p.get("outcome", {}).get("coding", [{}])[0].get("display")

    return {
        "procedure_id": p.get("id"),
        "patient_id": p.get("subject", {}).get("reference", "").replace("urn:uuid:", ""),
        "encounter_id": p.get("encounter", {}).get("reference", "").replace("urn:uuid:", ""),
        "status": p.get("status"),
        "snomed_code": snomed.get("code"),
        "snomed_display": snomed.get("display"),
        "performed_start": performed_start,
        "performed_end": performed_end,
        "reason_snomed": reason_coding.get("code"),
        "reason_display": reason_coding.get("display"),
        "body_site": body_site,
        "outcome": outcome,
    }


parse_procedure_udf = F.udf(parse_procedure, procedure_schema)

# COMMAND ----------

df_parsed = (
    df_raw
    .withColumn("parsed", parse_procedure_udf(F.col("raw_json")))
    .select(
        F.col("parsed.*"),
        F.col("ingestion_timestamp"),
        F.col("batch_id"),
        F.current_timestamp().alias("silver_timestamp"),
    )
    .filter(F.col("patient_id").isNotNull() & (F.col("patient_id") != ""))
    .withColumn("performed_start", F.to_timestamp("performed_start"))
    .withColumn("performed_end", F.to_timestamp("performed_end"))
)

print(f"Silver procedure rows: {df_parsed.count()}")
df_parsed.groupBy("status").count().show()

# COMMAND ----------

(
    df_parsed.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{SILVER_BASE}/procedure")
)

print(f"Written to {SILVER_BASE}/procedure")
df_parsed.select("procedure_id", "patient_id", "snomed_display", "status", "performed_start").show(10)
