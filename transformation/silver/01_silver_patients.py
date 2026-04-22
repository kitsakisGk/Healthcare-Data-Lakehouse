"""
transformation/silver/01_silver_patients.py

Bronze → Silver: Parse the raw Patient FHIR JSON into a clean flat table.

What this does:
  - Reads raw_json from bronze.patient
  - Parses FHIR Patient resource fields: id, name, birthDate, gender, address, etc.
  - Standardises date formats
  - Flags PII columns for FADP compliance (handled in governance layer)
  - Writes to silver.patient Delta table

FHIR Patient resource reference:
  https://www.hl7.org/fhir/patient.html
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

df_raw = spark.read.format("delta").load(f"{BRONZE_BASE}/patient")
print(f"Bronze patient rows: {df_raw.count()}")

# COMMAND ----------

patient_schema = StructType([
    StructField("patient_id", StringType()),
    StructField("birth_date", StringType()),
    StructField("gender", StringType()),
    StructField("family_name", StringType()),
    StructField("given_name", StringType()),
    StructField("marital_status", StringType()),
    StructField("race", StringType()),
    StructField("ethnicity", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("country", StringType()),
    StructField("postal_code", StringType()),
    StructField("mrn", StringType()),
    StructField("deceased", BooleanType()),
    StructField("deceased_date", StringType()),
    StructField("language", StringType()),
    StructField("multiple_birth", BooleanType()),
])


def parse_patient(raw_json: str) -> dict:
    try:
        p = json.loads(raw_json)
    except Exception:
        return {}

    name_obj = next((n for n in p.get("name", []) if n.get("use") == "official"), {})
    addr_obj = next(iter(p.get("address", [])), {})
    mrn = next(
        (i.get("value") for i in p.get("identifier", []) if "MR" in str(i.get("type", {}))),
        None,
    )

    return {
        "patient_id": p.get("id"),
        "birth_date": p.get("birthDate"),
        "gender": p.get("gender"),
        "family_name": name_obj.get("family", ""),         # PII
        "given_name": " ".join(name_obj.get("given", [])), # PII
        "marital_status": p.get("maritalStatus", {}).get("text"),
        "race": next(
            (ext.get("valueString") for ext in p.get("extension", []) if "race" in ext.get("url", "")),
            None,
        ),
        "ethnicity": next(
            (ext.get("valueString") for ext in p.get("extension", []) if "ethnicity" in ext.get("url", "")),
            None,
        ),
        "city": addr_obj.get("city"),           # PII
        "state": addr_obj.get("state"),
        "country": addr_obj.get("country"),
        "postal_code": addr_obj.get("postalCode"),  # PII
        "mrn": mrn,                             # PII
        "deceased": p.get("deceasedBoolean", p.get("deceasedDateTime") is not None),
        "deceased_date": p.get("deceasedDateTime"),
        "language": next(
            (lc.get("language", {}).get("text") for lc in p.get("communication", [])),
            None,
        ),
        "multiple_birth": p.get("multipleBirthBoolean"),
    }


parse_patient_udf = F.udf(parse_patient, patient_schema)

# COMMAND ----------

df_parsed = (
    df_raw
    .withColumn("parsed", parse_patient_udf(F.col("raw_json")))
    .select(
        F.col("parsed.*"),
        F.col("ingestion_timestamp"),
        F.col("source_file"),
        F.col("batch_id"),
        F.current_timestamp().alias("silver_timestamp"),
    )
)

# COMMAND ----------
# Dedup — keep latest record per patient

null_ids = df_parsed.filter(F.col("patient_id").isNull()).count()
assert null_ids == 0, f"Found {null_ids} rows with null patient_id"

from pyspark.sql.window import Window
w = Window.partitionBy("patient_id").orderBy(F.col("ingestion_timestamp").desc())
df_parsed = (
    df_parsed
    .withColumn("_row_num", F.row_number().over(w))
    .filter(F.col("_row_num") == 1)
    .drop("_row_num")
)

print(f"Silver patient rows after dedup: {df_parsed.count()}")

# COMMAND ----------

(
    df_parsed.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{SILVER_BASE}/patient")
)

print(f"Written to {SILVER_BASE}/patient")
df_parsed.select("patient_id", "birth_date", "gender", "city", "country", "deceased").show(10)
