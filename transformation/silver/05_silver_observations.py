"""
transformation/silver/05_silver_observations.py

Bronze → Silver: Parse FHIR Observation resources.

FHIR Observation = a clinical measurement or test result.
Examples: blood pressure, blood glucose, BMI, lab results (HbA1c, cholesterol).
Coded using LOINC (the international standard for lab and clinical observations).

This is rich ML feature territory — vital signs and lab values are strong
predictors of readmission risk.

FHIR Observation reference:
  https://www.hl7.org/fhir/observation.html
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import json

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

BRONZE_BASE = "./spark-warehouse/bronze"
SILVER_BASE = "./spark-warehouse/silver"

# COMMAND ----------

df_raw = spark.read.format("delta").load(f"{BRONZE_BASE}/observation")
print(f"Bronze observation rows: {df_raw.count()}")

# COMMAND ----------

observation_schema = StructType([
    StructField("observation_id", StringType()),
    StructField("patient_id", StringType()),
    StructField("encounter_id", StringType()),
    StructField("status", StringType()),
    StructField("category", StringType()),         # vital-signs, laboratory, survey, etc.
    StructField("loinc_code", StringType()),       # LOINC code — international lab standard
    StructField("loinc_display", StringType()),
    StructField("effective_datetime", StringType()),
    StructField("value_quantity", DoubleType()),   # numeric result (e.g. 7.2 for HbA1c)
    StructField("value_unit", StringType()),       # unit (e.g. %, mmHg, kg/m2)
    StructField("value_string", StringType()),     # text result when not numeric
    StructField("value_code", StringType()),       # coded result (e.g. positive/negative)
    StructField("interpretation", StringType()),   # normal, high, low, critical
    StructField("reference_range_low", DoubleType()),
    StructField("reference_range_high", DoubleType()),
])


def parse_observation(raw_json: str) -> dict:
    try:
        o = json.loads(raw_json)
    except Exception:
        return {}

    # LOINC coding
    codings = o.get("code", {}).get("coding", [])
    loinc = next((c for c in codings if "loinc" in c.get("system", "").lower()), codings[0] if codings else {})

    # Category
    category = None
    cats = o.get("category", [])
    if cats:
        cat_codings = cats[0].get("coding", [{}])
        category = cat_codings[0].get("code") if cat_codings else None

    # Value — FHIR uses different value[x] fields depending on the type
    value_qty = o.get("valueQuantity", {})
    value_cc = o.get("valueCodeableConcept", {})
    value_string = o.get("valueString")

    try:
        qty_value = float(value_qty.get("value")) if value_qty.get("value") is not None else None
    except (TypeError, ValueError):
        qty_value = None

    # Interpretation
    interp_codings = o.get("interpretation", [{}])[0].get("coding", [{}]) if o.get("interpretation") else [{}]
    interpretation = interp_codings[0].get("display") if interp_codings else None

    # Reference range
    ref_range = o.get("referenceRange", [{}])[0] if o.get("referenceRange") else {}
    try:
        ref_low = float(ref_range.get("low", {}).get("value")) if ref_range.get("low") else None
        ref_high = float(ref_range.get("high", {}).get("value")) if ref_range.get("high") else None
    except (TypeError, ValueError):
        ref_low, ref_high = None, None

    return {
        "observation_id": o.get("id"),
        "patient_id": o.get("subject", {}).get("reference", "").replace("urn:uuid:", ""),
        "encounter_id": o.get("encounter", {}).get("reference", "").replace("urn:uuid:", ""),
        "status": o.get("status"),
        "category": category,
        "loinc_code": loinc.get("code"),
        "loinc_display": loinc.get("display"),
        "effective_datetime": o.get("effectiveDateTime"),
        "value_quantity": qty_value,
        "value_unit": value_qty.get("unit"),
        "value_string": value_string,
        "value_code": value_cc.get("coding", [{}])[0].get("display") if value_cc else None,
        "interpretation": interpretation,
        "reference_range_low": ref_low,
        "reference_range_high": ref_high,
    }


parse_observation_udf = F.udf(parse_observation, observation_schema)

# COMMAND ----------

df_parsed = (
    df_raw
    .withColumn("parsed", parse_observation_udf(F.col("raw_json")))
    .select(
        F.col("parsed.*"),
        F.col("ingestion_timestamp"),
        F.col("batch_id"),
        F.current_timestamp().alias("silver_timestamp"),
    )
    .filter(F.col("patient_id").isNotNull() & (F.col("patient_id") != ""))
    .withColumn("effective_datetime", F.to_timestamp("effective_datetime"))
)

print(f"Silver observation rows: {df_parsed.count()}")

# Quick breakdown by category
df_parsed.groupBy("category").count().orderBy(F.col("count").desc()).show()

# COMMAND ----------

(
    df_parsed.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{SILVER_BASE}/observation")
)

print(f"Written to {SILVER_BASE}/observation")
df_parsed.select("observation_id", "patient_id", "loinc_display", "value_quantity", "value_unit", "interpretation").show(10)
