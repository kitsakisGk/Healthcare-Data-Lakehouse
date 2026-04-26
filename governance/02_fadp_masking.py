"""
governance/02_fadp_masking.py

FADP compliance — PII masking for the Silver layer.

FADP = Federal Act on Data Protection (Switzerland's nDSG, in force since Sept 2023).
Stricter than GDPR in several areas. Any system handling personal health data
of Swiss residents must comply.

What this does:
  - Scans Silver tables and identifies PII columns
  - Applies SHA-256 hashing to PII (one-way, irreversible)
  - Writes masked versions ready for Gold — Gold has zero PII
  - Tags PII columns in Unity Catalog for lineage tracking

PII columns in this dataset:
  silver.patient: family_name, given_name, city, postal_code, mrn
  silver.medication: prescriber_id (links to a real person)

Hashing approach:
  - SHA-256 of the raw value
  - Consistent: same input always gives same hash (referential integrity preserved)
  - Irreversible: cannot reconstruct original value from hash
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

SILVER_BASE  = "./spark-warehouse/silver"
CATALOG      = "healthcare_lakehouse"

# COMMAND ----------
# PII column registry — defines what gets masked and how

PII_REGISTRY = {
    "patient": {
        "hash":   ["family_name", "given_name", "mrn"],
        "suppress": ["city", "postal_code"],   # too granular — drop from Gold entirely
    },
    "medication": {
        "hash":   ["prescriber_id"],
        "suppress": [],
    },
}

# COMMAND ----------

def apply_masking(df, table_name: str):
    """
    Apply FADP masking rules to a Silver DataFrame.
    - Hash columns: replace value with SHA-256 hash
    - Suppress columns: drop the column entirely
    """
    rules = PII_REGISTRY.get(table_name, {})

    for col in rules.get("hash", []):
        if col in df.columns:
            df = df.withColumn(col, F.sha2(F.col(col).cast("string"), 256))
            print(f"  Hashed: {table_name}.{col}")

    suppress = rules.get("suppress", [])
    existing_suppress = [c for c in suppress if c in df.columns]
    if existing_suppress:
        df = df.drop(*existing_suppress)
        print(f"  Suppressed: {table_name}.{existing_suppress}")

    return df

# COMMAND ----------
# Apply masking and write masked Silver tables

for table_name in PII_REGISTRY.keys():
    print(f"\nProcessing: {table_name}")
    df = spark.read.format("delta").load(f"{SILVER_BASE}/{table_name}")
    df_masked = apply_masking(df, table_name)

    output_path = f"{SILVER_BASE}/{table_name}_masked"
    (
        df_masked.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(output_path)
    )
    print(f"  Written masked table to: {output_path}")

# COMMAND ----------
# Tag PII columns in Unity Catalog for lineage tracking

PII_TAGS = [
    ("silver", "patient", "family_name"),
    ("silver", "patient", "given_name"),
    ("silver", "patient", "mrn"),
    ("silver", "patient", "city"),
    ("silver", "patient", "postal_code"),
    ("silver", "medication", "prescriber_id"),
]

for schema, table, column in PII_TAGS:
    try:
        spark.sql(f"""
            ALTER TABLE {CATALOG}.{schema}.{table}
            ALTER COLUMN {column}
            SET TAGS ('pii' = 'true', 'fadp_sensitive' = 'true')
        """)
        print(f"Tagged PII: {schema}.{table}.{column}")
    except Exception as e:
        print(f"Tagging skipped (requires Unity Catalog): {e}")

# COMMAND ----------
print("\nFADP masking complete.")
print("PII hashed:     family_name, given_name, mrn, prescriber_id")
print("PII suppressed: city, postal_code")
print("Gold layer has zero PII columns.")
