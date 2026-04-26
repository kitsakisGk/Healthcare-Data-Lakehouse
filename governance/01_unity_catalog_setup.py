"""
governance/01_unity_catalog_setup.py

Unity Catalog setup — defines the 3-schema catalog structure for the lakehouse.

What this does:
  - Creates the main catalog: healthcare_lakehouse
  - Creates 3 schemas: bronze, silver, gold
  - Registers all Delta tables into the catalog
  - Sets table ownership and access controls
  - Adds column-level comments for documentation and lineage

Unity Catalog is Databricks' enterprise data governance layer.
It provides: fine-grained access control, data lineage, column tagging, auditing.

NOTE: Requires a Databricks workspace with Unity Catalog enabled.
      Run this as an admin user.
"""

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

CATALOG = "healthcare_lakehouse"
BRONZE_PATH = "abfss://delta@<your_storage>.dfs.core.windows.net/bronze"
SILVER_PATH = "abfss://delta@<your_storage>.dfs.core.windows.net/silver"
GOLD_PATH   = "abfss://delta@<your_storage>.dfs.core.windows.net/gold"

# COMMAND ----------
# Create catalog and schemas

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")

for schema in ["bronze", "silver", "gold"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
    print(f"Schema ready: {CATALOG}.{schema}")

# COMMAND ----------
# Register Bronze tables

BRONZE_TABLES = [
    "patient", "condition", "observation",
    "medicationrequest", "encounter", "procedure",
]

for table in BRONZE_TABLES:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.bronze.{table}
        USING DELTA
        LOCATION '{BRONZE_PATH}/{table}'
    """)
    spark.sql(f"ALTER TABLE {CATALOG}.bronze.{table} SET OWNER TO `data_engineers`")
    print(f"Registered: {CATALOG}.bronze.{table}")

# COMMAND ----------
# Register Silver tables

SILVER_TABLES = [
    "patient", "condition", "encounter",
    "medication", "observation", "procedure",
]

for table in SILVER_TABLES:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.silver.{table}
        USING DELTA
        LOCATION '{SILVER_PATH}/{table}'
    """)
    spark.sql(f"ALTER TABLE {CATALOG}.silver.{table} SET OWNER TO `data_engineers`")
    print(f"Registered: {CATALOG}.silver.{table}")

# COMMAND ----------
# Register Gold tables

GOLD_TABLES = [
    "patient_summary", "patient_features", "population_health",
    "condition_prevalence", "encounter_volume_monthly",
    "readmission_by_condition", "readmission_scores", "patient_risk_factors",
]

for table in GOLD_TABLES:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.gold.{table}
        USING DELTA
        LOCATION '{GOLD_PATH}/{table}'
    """)
    spark.sql(f"ALTER TABLE {CATALOG}.gold.{table} SET OWNER TO `data_engineers`")
    print(f"Registered: {CATALOG}.gold.{table}")

# COMMAND ----------
# Column-level access control
# Bronze patient table contains PII — restrict to data engineers only
# Analysts get read access to Silver and Gold only

spark.sql(f"""
    GRANT SELECT ON TABLE {CATALOG}.gold.patient_summary TO `analysts`
""")
spark.sql(f"""
    GRANT SELECT ON TABLE {CATALOG}.gold.patient_features TO `analysts`
""")
spark.sql(f"""
    GRANT SELECT ON TABLE {CATALOG}.gold.patient_risk_factors TO `analysts`
""")
spark.sql(f"""
    GRANT SELECT ON TABLE {CATALOG}.gold.condition_prevalence TO `analysts`
""")

# Bronze is restricted — PII present
spark.sql(f"""
    REVOKE SELECT ON SCHEMA {CATALOG}.bronze FROM `analysts`
""")

print("\nAccess controls applied.")
print("  analysts    -> Gold (read only)")
print("  data_engineers -> Bronze + Silver + Gold (full)")

# COMMAND ----------
# Add column comments to Silver patient table for documentation

column_comments = {
    "patient_id":    "Unique FHIR patient identifier (UUID)",
    "birth_date":    "Patient date of birth (yyyy-MM-dd)",
    "gender":        "Patient gender as reported in FHIR",
    "family_name":   "PII — patient family name, masked in Gold",
    "given_name":    "PII — patient given name, masked in Gold",
    "city":          "PII — patient city of residence",
    "postal_code":   "PII — patient postal code",
    "mrn":           "PII — Medical Record Number",
}

for col, comment in column_comments.items():
    spark.sql(f"""
        ALTER TABLE {CATALOG}.silver.patient
        ALTER COLUMN {col} COMMENT '{comment}'
    """)

print("\nColumn comments added to silver.patient.")
print("\nUnity Catalog setup complete.")
