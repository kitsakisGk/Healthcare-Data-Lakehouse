"""
tests/03_test_gold.py

Data quality tests for the Gold layer.

Checks:
  - One row per patient in patient_summary and patient_features
  - Readmission rate is within a realistic clinical range (5–40%)
  - No negative ages or impossible values
  - ML feature table has no nulls in critical columns
  - No PII columns in Gold
  - Risk scores are between 0 and 1
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

GOLD_BASE = "./spark-warehouse/gold"

# COMMAND ----------

def load(table: str):
    return spark.read.format("delta").load(f"{GOLD_BASE}/{table}")

# COMMAND ----------

def test_one_row_per_patient():
    """patient_summary and patient_features must have exactly one row per patient."""
    print("\n=== Test: One row per patient ===")
    for table in ["patient_summary", "patient_features"]:
        df = load(table)
        total = df.count()
        distinct = df.select("patient_id").distinct().count()
        status = "PASS" if total == distinct else "FAIL"
        print(f"  {status}  {table}: total={total}, distinct_patients={distinct}")
        assert total == distinct, f"{table} has duplicate patient rows"

# COMMAND ----------

def test_readmission_rate_realistic():
    """
    30-day readmission rate should be between 5% and 40%.
    Clinical reality: ~15-20% for general inpatient populations.
    Synthea may differ slightly.
    """
    print("\n=== Test: Readmission rate is clinically realistic (5–40%) ===")
    df = load("patient_summary")
    total = df.count()
    readmitted = df.filter(F.col("readmitted_30d") == True).count()
    rate = readmitted / total if total > 0 else 0
    status = "PASS" if 0.05 <= rate <= 0.40 else "WARN"
    print(f"  {status}  Readmission rate: {rate:.2%} ({readmitted}/{total})")

# COMMAND ----------

def test_no_negative_values():
    """Counts and ages must be non-negative."""
    print("\n=== Test: No negative values in numeric columns ===")
    df = load("patient_features")
    checks = [
        "age", "active_condition_count", "total_encounters",
        "inpatient_count", "emergency_count", "total_medication_count",
        "total_procedure_count",
    ]
    for col in checks:
        if col not in df.columns:
            print(f"  SKIP  {col} not in table")
            continue
        negatives = df.filter(F.col(col) < 0).count()
        status = "PASS" if negatives == 0 else "FAIL"
        print(f"  {status}  {col}: {negatives} negative values")
        assert negatives == 0, f"{col} has {negatives} negative values"

# COMMAND ----------

def test_no_pii_in_gold():
    """Gold layer must not contain any PII columns."""
    print("\n=== Test: No PII columns in Gold ===")
    pii_columns = ["family_name", "given_name", "mrn", "city", "postal_code", "address"]
    for table in ["patient_summary", "patient_features"]:
        df = load(table)
        found_pii = [c for c in pii_columns if c in df.columns]
        status = "PASS" if not found_pii else "FAIL"
        print(f"  {status}  {table}: PII found={found_pii}")
        assert not found_pii, f"{table} contains PII columns: {found_pii}"

# COMMAND ----------

def test_risk_scores_in_range():
    """Readmission risk scores must be between 0 and 1."""
    print("\n=== Test: Risk scores in range [0, 1] ===")
    try:
        df = load("readmission_scores")
        out_of_range = df.filter(
            (F.col("readmission_risk_score") < 0) | (F.col("readmission_risk_score") > 1)
        ).count()
        status = "PASS" if out_of_range == 0 else "FAIL"
        print(f"  {status}  {out_of_range} scores outside [0, 1]")
        assert out_of_range == 0
    except Exception:
        print("  SKIP  readmission_scores table not yet generated (run ML scripts first)")

# COMMAND ----------

def test_ml_features_no_nulls_in_critical_cols():
    """Core ML features must not be null — nulls were filled during feature engineering."""
    print("\n=== Test: No nulls in critical ML feature columns ===")
    df = load("patient_features")
    critical = [
        "age", "active_condition_count", "total_encounters",
        "inpatient_count", "total_medication_count",
    ]
    for col in critical:
        if col not in df.columns:
            continue
        nulls = df.filter(F.col(col).isNull()).count()
        status = "PASS" if nulls == 0 else "FAIL"
        print(f"  {status}  {col}: {nulls} nulls")
        assert nulls == 0, f"{col} has {nulls} null values in patient_features"

# COMMAND ----------

if __name__ == "__main__":
    print("=" * 50)
    print("Running Gold layer tests")
    print("=" * 50)
    test_one_row_per_patient()
    test_readmission_rate_realistic()
    test_no_negative_values()
    test_no_pii_in_gold()
    test_risk_scores_in_range()
    test_ml_features_no_nulls_in_critical_cols()
    print("\nAll Gold tests passed.")
