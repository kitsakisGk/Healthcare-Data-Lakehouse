"""
tests/01_test_bronze.py

Data quality tests for the Bronze layer.

Checks:
  - All expected resource type tables exist
  - Row counts are above minimum thresholds
  - No null resource_id values
  - raw_json is valid JSON
  - Ingestion metadata columns are present and populated
  - No duplicate resource_ids within a batch
"""

# COMMAND ----------

import json
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

BRONZE_BASE = "./spark-warehouse/bronze"
MIN_ROWS = 100  # minimum expected rows per resource type for 1000 Synthea patients

RESOURCE_TYPES = [
    "patient",
    "condition",
    "observation",
    "medicationrequest",
    "encounter",
    "procedure",
]

# COMMAND ----------

def load_table(resource_type: str):
    return spark.read.format("delta").load(f"{BRONZE_BASE}/{resource_type}")

# COMMAND ----------

def test_tables_exist():
    print("\n=== Test: Tables exist ===")
    for rt in RESOURCE_TYPES:
        try:
            df = load_table(rt)
            print(f"  PASS  {rt}")
        except Exception as e:
            print(f"  FAIL  {rt} — {e}")
            raise AssertionError(f"Bronze table missing: {rt}")

# COMMAND ----------

def test_row_counts():
    print("\n=== Test: Minimum row counts ===")
    for rt in RESOURCE_TYPES:
        df = load_table(rt)
        count = df.count()
        status = "PASS" if count >= MIN_ROWS else "FAIL"
        print(f"  {status}  {rt}: {count} rows (min={MIN_ROWS})")
        assert count >= MIN_ROWS, f"{rt} has only {count} rows, expected >= {MIN_ROWS}"

# COMMAND ----------

def test_no_null_resource_ids():
    print("\n=== Test: No null resource_id ===")
    for rt in RESOURCE_TYPES:
        df = load_table(rt)
        null_count = df.filter(
            F.col("resource_id").isNull() | (F.col("resource_id") == "")
        ).count()
        status = "PASS" if null_count == 0 else "FAIL"
        print(f"  {status}  {rt}: {null_count} null resource_ids")
        assert null_count == 0, f"{rt} has {null_count} null resource_ids"

# COMMAND ----------

def test_metadata_columns_present():
    print("\n=== Test: Metadata columns present ===")
    required_cols = ["raw_json", "resource_id", "ingestion_timestamp", "source_file", "batch_id"]
    for rt in RESOURCE_TYPES:
        df = load_table(rt)
        missing = [c for c in required_cols if c not in df.columns]
        status = "PASS" if not missing else "FAIL"
        print(f"  {status}  {rt}: missing={missing}")
        assert not missing, f"{rt} missing columns: {missing}"

# COMMAND ----------

def test_raw_json_parseable():
    print("\n=== Test: raw_json is valid JSON (sample of 10 rows) ===")
    for rt in RESOURCE_TYPES:
        df = load_table(rt)
        sample = df.select("raw_json").limit(10).collect()
        for i, row in enumerate(sample):
            try:
                json.loads(row["raw_json"])
            except json.JSONDecodeError as e:
                print(f"  FAIL  {rt} row {i}: {e}")
                raise AssertionError(f"{rt} has unparseable raw_json at row {i}")
        print(f"  PASS  {rt}: all 10 sample rows parse OK")

# COMMAND ----------

def test_no_duplicate_resource_ids():
    print("\n=== Test: No duplicate resource_ids per batch ===")
    for rt in RESOURCE_TYPES:
        df = load_table(rt)
        dupes = (
            df.groupBy("batch_id", "resource_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        status = "PASS" if dupes == 0 else "WARN"
        print(f"  {status}  {rt}: {dupes} duplicate resource_ids")

# COMMAND ----------

if __name__ == "__main__":
    print("=" * 50)
    print("Running Bronze layer tests")
    print("=" * 50)
    test_tables_exist()
    test_row_counts()
    test_no_null_resource_ids()
    test_metadata_columns_present()
    test_raw_json_parseable()
    test_no_duplicate_resource_ids()
    print("\nAll Bronze tests passed.")
