"""
tests/02_test_silver.py

Data quality tests for the Silver layer.

Checks:
  - Referential integrity: every condition/encounter/medication has a valid patient_id
  - Date format validity
  - No orphaned records
  - Expected coded values (SNOMED, LOINC, RxNorm) are populated
  - PII columns exist in silver but are flagged
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

SILVER_BASE = "./spark-warehouse/silver"

# COMMAND ----------

def load(table: str):
    return spark.read.format("delta").load(f"{SILVER_BASE}/{table}")

# COMMAND ----------

def test_referential_integrity():
    """Every clinical record must link to a known patient."""
    print("\n=== Test: Referential integrity ===")
    df_patients = load("patient").select("patient_id")
    patient_ids = set(r.patient_id for r in df_patients.collect())

    clinical_tables = ["condition", "encounter", "medication", "observation", "procedure"]

    for table in clinical_tables:
        df = load(table)
        total = df.count()
        orphans = df.filter(~F.col("patient_id").isin(patient_ids)).count()
        pct = orphans / total * 100 if total > 0 else 0
        status = "PASS" if orphans == 0 else "FAIL"
        print(f"  {status}  {table}: {orphans}/{total} orphaned records ({pct:.1f}%)")
        assert orphans == 0, f"{table} has {orphans} records with no matching patient"

# COMMAND ----------

def test_date_columns_not_null():
    """Key date columns should be populated for the majority of records."""
    print("\n=== Test: Key date columns populated (>90%) ===")
    checks = {
        "patient":     "birth_date",
        "condition":   "onset_date",
        "encounter":   "start_datetime",
        "medication":  "authored_on",
        "observation": "effective_datetime",
        "procedure":   "performed_start",
    }
    for table, date_col in checks.items():
        df = load(table)
        total = df.count()
        populated = df.filter(F.col(date_col).isNotNull()).count()
        pct = populated / total * 100 if total > 0 else 0
        status = "PASS" if pct >= 90 else "FAIL"
        print(f"  {status}  {table}.{date_col}: {pct:.1f}% populated")
        assert pct >= 90, f"{table}.{date_col} only {pct:.1f}% populated"

# COMMAND ----------

def test_coded_values_populated():
    """Standard coding systems should be present on the majority of records."""
    print("\n=== Test: Coded values populated (>80%) ===")
    checks = {
        "condition":   "snomed_code",
        "observation": "loinc_code",
        "medication":  "rxnorm_code",
        "procedure":   "snomed_code",
    }
    for table, code_col in checks.items():
        df = load(table)
        total = df.count()
        coded = df.filter(F.col(code_col).isNotNull()).count()
        pct = coded / total * 100 if total > 0 else 0
        status = "PASS" if pct >= 80 else "FAIL"
        print(f"  {status}  {table}.{code_col}: {pct:.1f}% coded")
        assert pct >= 80, f"{table}.{code_col} only {pct:.1f}% coded"

# COMMAND ----------

def test_pii_columns_present_in_silver():
    """PII columns must exist in Silver (they get masked before Gold)."""
    print("\n=== Test: PII columns present in silver.patient ===")
    pii_cols = ["family_name", "given_name", "mrn", "city", "postal_code"]
    df = load("patient")
    for col in pii_cols:
        exists = col in df.columns
        status = "PASS" if exists else "FAIL"
        print(f"  {status}  silver.patient.{col} exists={exists}")
        assert exists, f"PII column missing from silver.patient: {col}"

# COMMAND ----------

def test_patient_age_range():
    """All patients should have a realistic age (0–120)."""
    print("\n=== Test: Patient age range ===")
    df = load("patient")
    df_age = df.withColumn(
        "age",
        F.floor(F.months_between(F.current_date(), F.to_date("birth_date", "yyyy-MM-dd")) / 12)
    )
    invalid = df_age.filter((F.col("age") < 0) | (F.col("age") > 120)).count()
    status = "PASS" if invalid == 0 else "FAIL"
    print(f"  {status}  {invalid} patients with invalid age")
    assert invalid == 0, f"{invalid} patients have age outside 0-120"

# COMMAND ----------

if __name__ == "__main__":
    print("=" * 50)
    print("Running Silver layer tests")
    print("=" * 50)
    test_referential_integrity()
    test_date_columns_not_null()
    test_coded_values_populated()
    test_pii_columns_present_in_silver()
    test_patient_age_range()
    print("\nAll Silver tests passed.")
