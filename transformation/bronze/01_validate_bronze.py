"""
transformation/bronze/01_validate_bronze.py

Bronze validation — run after ingestion to confirm data landed correctly.

Checks:
  - Row counts per resource type
  - No null resource_id values
  - Batch IDs present
  - raw_json is valid JSON (sample check)
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import json

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

BRONZE_BASE = "./spark-warehouse/bronze"  # Change to ADLS path in production

RESOURCE_TYPES = [
    "patient",
    "condition",
    "observation",
    "medicationrequest",
    "encounter",
    "procedure",
]

# COMMAND ----------

print("=== Bronze Layer — Row Counts ===")
for rt in RESOURCE_TYPES:
    try:
        df = spark.read.format("delta").load(f"{BRONZE_BASE}/{rt}")
        count = df.count()
        null_ids = df.filter(F.col("resource_id").isNull() | (F.col("resource_id") == "")).count()
        batches = df.select("batch_id").distinct().count()
        print(f"  {rt:25s} rows={count:6d}  null_ids={null_ids:4d}  batches={batches}")
    except Exception as e:
        print(f"  {rt:25s} ERROR: {e}")

# COMMAND ----------

print("\n=== Sample raw_json parse check (patient) ===")
df_patient = spark.read.format("delta").load(f"{BRONZE_BASE}/patient")
sample = df_patient.select("raw_json").limit(5).collect()

for row in sample:
    try:
        parsed = json.loads(row["raw_json"])
        print(f"  OK — resourceType={parsed.get('resourceType')}  id={parsed.get('id', '')[:20]}")
    except json.JSONDecodeError as e:
        print(f"  FAIL — {e}")

# COMMAND ----------

print("\nBronze validation complete.")
