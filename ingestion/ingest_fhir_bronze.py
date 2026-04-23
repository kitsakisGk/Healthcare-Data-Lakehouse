"""
ingestion/ingest_fhir_bronze.py

Phase 1 — Bronze ingestion of Synthea FHIR JSON bundles into Delta Lake.

What this does:
  - Reads raw FHIR Bundle JSON files from a source directory (local or ADLS)
  - Explodes each bundle into individual resource types (Patient, Condition, etc.)
  - Writes them as Delta tables in the Bronze schema, partitioned by resource type
  - Adds ingestion metadata columns to every record for traceability

Run locally:  python ingestion/ingest_fhir_bronze.py --source ./ingestion/synthea/output/fhir
Run on ADLS:  set SOURCE_PATH env var to your abfss:// path
"""

import os
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BRONZE_CATALOG = os.getenv("BRONZE_CATALOG", "healthcare_lakehouse")
BRONZE_SCHEMA = os.getenv("BRONZE_SCHEMA", "bronze")
BATCH_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

RESOURCE_TYPES = [
    "Patient",
    "Condition",
    "Observation",
    "MedicationRequest",
    "Encounter",
    "Procedure",
    "DiagnosticReport",
    "AllergyIntolerance",
    "Immunization",
    "CarePlan",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

def get_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("FHIR Bronze Ingestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return builder.getOrCreate()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_fhir_bundles(source_path: str) -> list[dict]:
    source = Path(source_path)
    bundles = []
    files = list(source.glob("*.json"))
    log.info(f"Found {len(files)} FHIR bundle files in {source_path}")

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                bundle = json.load(fp)
            if bundle.get("resourceType") == "Bundle":
                bundles.append({"source_file": f.name, "bundle": bundle})
        except Exception as e:
            log.warning(f"Failed to parse {f.name}: {e}")

    log.info(f"Successfully loaded {len(bundles)} bundles")
    return bundles


def extract_resources(bundles: list[dict]) -> dict[str, list[dict]]:
    resources: dict[str, list[dict]] = {rt: [] for rt in RESOURCE_TYPES}
    ingestion_ts = datetime.now(timezone.utc).isoformat()

    for item in bundles:
        source_file = item["source_file"]
        entries = item["bundle"].get("entry", [])

        for entry in entries:
            resource = entry.get("resource", {})
            rt = resource.get("resourceType")

            if rt not in resources:
                continue

            resource["_ingestion_timestamp"] = ingestion_ts
            resource["_source_file"] = source_file
            resource["_batch_id"] = BATCH_ID

            resources[rt].append(resource)

    for rt, recs in resources.items():
        log.info(f"  {rt}: {len(recs)} records extracted")

    return resources


def write_bronze_table(
    spark: SparkSession,
    resource_type: str,
    records: list[dict],
    output_base: str,
) -> None:
    if not records:
        log.info(f"No records for {resource_type}, skipping.")
        return

    output_path = f"{output_base}/{resource_type.lower()}"

    rows = [
        {
            "raw_json": json.dumps({k: v for k, v in r.items() if not k.startswith("_")}),
            "resource_id": r.get("id", ""),
            "ingestion_timestamp": r["_ingestion_timestamp"],
            "source_file": r["_source_file"],
            "batch_id": r["_batch_id"],
        }
        for r in records
    ]

    df = spark.createDataFrame(rows)

    (
        df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .save(output_path)
    )

    log.info(f"Written {len(records)} records to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(source_path: str, output_base: str) -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info(f"=== FHIR Bronze Ingestion | Batch: {BATCH_ID} ===")
    log.info(f"Source: {source_path}")
    log.info(f"Output: {output_base}")

    bundles = load_fhir_bundles(source_path)
    resources = extract_resources(bundles)

    for resource_type, records in resources.items():
        write_bronze_table(spark, resource_type, records, output_base)

    log.info("=== Ingestion complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Synthea FHIR bundles into Bronze Delta tables")
    parser.add_argument(
        "--source",
        default=os.getenv("SOURCE_PATH", "./ingestion/synthea/output/fhir"),
    )
    parser.add_argument(
        "--output",
        default=os.getenv("OUTPUT_PATH", "./spark-warehouse/bronze"),
    )
    args = parser.parse_args()
    run(args.source, args.output)
