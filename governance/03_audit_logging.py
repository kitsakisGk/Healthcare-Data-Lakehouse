"""
governance/03_audit_logging.py

Audit logging for FADP compliance.

FADP requires organisations to be able to demonstrate WHO accessed personal
health data, WHEN, and WHAT they did with it.

What this does:
  - Creates an audit_log Delta table
  - Provides a log_access() function to record every read/write on sensitive tables
  - Queries the audit log for compliance reporting (e.g. "who accessed patient data
    in the last 30 days?")

In a full Databricks deployment this is complemented by:
  - Databricks system tables (system.access.audit) — automatic cluster-level logging
  - Unity Catalog audit logs — automatic table-level access logging
  This script adds application-level logging on top for FADP-specific reporting.
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from datetime import datetime, timezone

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

AUDIT_LOG_PATH = "./spark-warehouse/governance/audit_log"

# COMMAND ----------
# Create audit log table if it doesn't exist

audit_schema = StructType([
    StructField("event_timestamp", TimestampType(), False),
    StructField("user",            StringType(),    False),
    StructField("action",          StringType(),    False),   # READ, WRITE, DELETE
    StructField("catalog",         StringType(),    True),
    StructField("schema",          StringType(),    True),
    StructField("table",           StringType(),    True),
    StructField("row_count",       StringType(),    True),
    StructField("query_hash",      StringType(),    True),    # SHA-256 of the query text
    StructField("session_id",      StringType(),    True),
    StructField("notes",           StringType(),    True),
])

try:
    spark.read.format("delta").load(AUDIT_LOG_PATH)
    print("Audit log table exists.")
except Exception:
    empty_df = spark.createDataFrame([], audit_schema)
    empty_df.write.format("delta").mode("overwrite").save(AUDIT_LOG_PATH)
    print(f"Audit log table created at: {AUDIT_LOG_PATH}")

# COMMAND ----------

def log_access(
    user: str,
    action: str,
    schema: str,
    table: str,
    row_count: int = None,
    query: str = None,
    notes: str = None,
    catalog: str = "healthcare_lakehouse",
):
    """
    Log a data access event to the audit log.

    Usage:
        log_access(user="analyst_1", action="READ", schema="gold", table="patient_summary", row_count=5000)
        log_access(user="pipeline", action="WRITE", schema="silver", table="patient", row_count=1200)
    """
    import hashlib

    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16] if query else None
    session_id = spark.conf.get("spark.databricks.clusterUsageTags.sessionId", "local")

    row = [(
        datetime.now(timezone.utc),
        user,
        action.upper(),
        catalog,
        schema,
        table,
        str(row_count) if row_count is not None else None,
        query_hash,
        session_id,
        notes,
    )]

    df_log = spark.createDataFrame(row, audit_schema)
    df_log.write.format("delta").mode("append").save(AUDIT_LOG_PATH)


# COMMAND ----------
# Example: log the pipeline run that built the Gold layer

log_access(
    user="pipeline_service_account",
    action="READ",
    schema="silver",
    table="patient",
    row_count=None,
    notes="Gold patient_features build — 03_gold_patient_features.py",
)

log_access(
    user="pipeline_service_account",
    action="WRITE",
    schema="gold",
    table="patient_features",
    notes="Gold patient_features build complete",
)

print("Example audit log entries written.")

# COMMAND ----------
# Compliance report: all access to sensitive tables in the last 30 days

df_audit = spark.read.format("delta").load(AUDIT_LOG_PATH)

print("\n=== Audit Log — Last 30 Days ===")
(
    df_audit
    .filter(F.col("event_timestamp") >= F.date_sub(F.current_timestamp(), 30))
    .filter(F.col("schema").isin("bronze", "silver"))   # sensitive layers only
    .orderBy(F.col("event_timestamp").desc())
    .select("event_timestamp", "user", "action", "schema", "table", "row_count", "notes")
    .show(50, truncate=False)
)

# COMMAND ----------
# Compliance summary: unique users who accessed patient data

print("\n=== Users Who Accessed Patient-Level Data ===")
(
    df_audit
    .filter(F.col("table").isin("patient", "patient_features", "patient_summary"))
    .groupBy("user", "action")
    .agg(
        F.count("*").alias("access_count"),
        F.max("event_timestamp").alias("last_access"),
    )
    .orderBy("user", "action")
    .show(truncate=False)
)
