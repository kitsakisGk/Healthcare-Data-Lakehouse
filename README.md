# Healthcare Data Lakehouse

**Azure Databricks · Delta Lake · FHIR · Unity Catalog · FADP Compliance**

A production-grade healthcare data pipeline ingesting synthetic FHIR patient data through a medallion architecture, with enterprise data governance built for Swiss data privacy requirements (FADP), and a patient readmission prediction model.

---

## Architecture

![Architecture Diagram](docs/architecture.png)

> Diagram coming in Phase 4. Pipeline: `Synthea → ADLS Gen2 → Databricks Bronze → Silver → Gold → Unity Catalog → Power BI + ML Model`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud Storage | Azure Data Lake Storage Gen2 |
| Processing | Azure Databricks (PySpark + SQL) |
| Table Format | Delta Lake |
| Governance | Unity Catalog |
| Data Standard | FHIR R4 (Synthea synthetic data) |
| ML | XGBoost + SHAP + MLflow |
| Visualisation | Power BI |
| Compliance | FADP (Swiss Federal Act on Data Protection) |
| Language | Python 3.10+ · SQL |

---

## Pipeline Walkthrough

| Phase | Folder | Description |
|---|---|---|
| Ingestion | [ingestion/](ingestion/) | Load raw Synthea FHIR JSON bundles into Bronze Delta tables — [ingest_fhir_bronze.py](ingestion/ingest_fhir_bronze.py) |
| Bronze | [transformation/bronze/](transformation/bronze/) | Raw FHIR data, partitioned by resource type |
| Silver | [transformation/silver/](transformation/silver/) | Cleaned, parsed, joined patient records — Patient, Condition, Encounter, Medication, Observation, Procedure |
| Gold | [transformation/gold/](transformation/gold/) | Aggregated KPIs, readmission flags, ML-ready feature table |
| ML | [ml/](ml/) | XGBoost readmission model, SHAP explainability |
| Governance | [governance/](governance/) | Unity Catalog setup, FADP masking, audit logging |
| Dashboard | [dashboard/](dashboard/) | Power BI reports connected to Gold layer |

### Silver Layer Scripts

| Script | FHIR Resource | Key Fields |
|---|---|---|
| [01_silver_patients.py](transformation/silver/01_silver_patients.py) | Patient | Demographics, PII flagged, dedup |
| [02_silver_conditions.py](transformation/silver/02_silver_conditions.py) | Condition | SNOMED CT + ICD-10 coding, clinical status |
| [03_silver_encounters.py](transformation/silver/03_silver_encounters.py) | Encounter | Visit type, LOS, timestamps — source of 30-day readmission flag |
| [04_silver_medications.py](transformation/silver/04_silver_medications.py) | MedicationRequest | RxNorm coding, active/chronic flags |
| [05_silver_observations.py](transformation/silver/05_silver_observations.py) | Observation | LOINC coding, lab values, vitals (HbA1c, BMI, BP, eGFR) |
| [06_silver_procedures.py](transformation/silver/06_silver_procedures.py) | Procedure | SNOMED coding, surgical history, dialysis flag |

### Governance Scripts

| Script | Description |
|---|---|
| [01_unity_catalog_setup.py](governance/01_unity_catalog_setup.py) | Creates catalog + 3 schemas, registers all Delta tables, applies access controls and column comments |
| [02_fadp_masking.py](governance/02_fadp_masking.py) | Hashes PII columns (name, MRN), suppresses granular location data, tags columns in Unity Catalog |
| [03_audit_logging.py](governance/03_audit_logging.py) | Logs every read/write on sensitive tables — user, timestamp, action, row count |

### ML Scripts

| Script | Description |
|---|---|
| [01_train_readmission_model.py](ml/01_train_readmission_model.py) | XGBoost binary classifier for 30-day readmission, MLflow experiment tracking, AUC/precision/recall evaluation |
| [02_explain_shap.py](ml/02_explain_shap.py) | SHAP feature importance, beeswarm and waterfall plots, per-patient top risk factors written to Gold |

### Tests

| Script | What it checks |
|---|---|
| [01_test_bronze.py](tests/01_test_bronze.py) | Table existence, row counts, null resource_ids, valid JSON, no duplicates |
| [02_test_silver.py](tests/02_test_silver.py) | Referential integrity, date completeness, SNOMED/LOINC/RxNorm coverage, PII presence, age range |
| [03_test_gold.py](tests/03_test_gold.py) | One row per patient, realistic readmission rate, no negative values, zero PII, risk scores in range |

### Gold Layer Tables

| Script | Output Table | Description |
|---|---|---|
| [01_gold_patient_summary.py](transformation/gold/01_gold_patient_summary.py) | `patient_summary` | One row per patient — demographics, encounter counts, 30-day readmission flag |
| [02_gold_population_health.py](transformation/gold/02_gold_population_health.py) | `condition_prevalence`, `encounter_volume_monthly`, `readmission_by_condition` | Population-level KPIs for Power BI |
| [03_gold_patient_features.py](transformation/gold/03_gold_patient_features.py) | `patient_features` | ML-ready feature table — labs, medications, procedures, derived risk flags |

---

## ML Model Results

> Results will be added after Phase 3 completion.

- **Task:** 30-day patient readmission (binary classification)
- **Model:** XGBoost
- **Explainability:** SHAP values
- **Metrics:** AUC-ROC, Precision/Recall (TBD)

---

## FADP Compliance Approach

This project implements a compliance layer aligned with Switzerland's Federal Act on Data Protection (FADP/nDSG):

- PII columns identified and tagged at Silver layer
- Data masking (hashing/tokenisation) applied before Gold
- Audit logs on all sensitive table access
- Retention policy metadata tags on patient-level tables

---

## How to Run

### Prerequisites
- Azure subscription (free tier sufficient)
- Databricks workspace (Community Edition works for dev)
- Python 3.10+
- Java 11+ (required for Synthea)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/Healthcare-Med-Data-Lakehouse.git
cd Healthcare-Med-Data-Lakehouse

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Generate synthetic patient data
cd ingestion/synthea
java -jar synthea-with-dependencies.jar -p 1000 --exporter.fhir.export true
```

> Full setup guide: [docs/setup_guide.md](docs/setup_guide.md)

---

## Project Status

- [x] Phase 1 — Foundation (repo structure, Synthea, Azure)
- [x] Phase 2 — Medallion pipeline (Bronze → Silver → Gold)
- [x] Phase 3 — Unity Catalog + FADP + ML model + Tests
- [ ] Phase 4 — Power BI dashboard + architecture diagram + polish

---

## Roadmap — Phase 4

Everything remaining before this project is fully complete and recruiter-ready.

### 4.1 — Synthea Data Generation
Generate 1000 synthetic patients locally and run the full pipeline end-to-end.

```bash
# Install Java 11+ from https://adoptium.net
java -version

# Download synthea-with-dependencies.jar from:
# https://github.com/synthetichealth/synthea/releases/latest
mkdir ingestion/synthea && cd ingestion/synthea

# Generate 1000 patients in FHIR R4 format
java -jar synthea-with-dependencies.jar -p 1000 --exporter.fhir.export true

# Run the bronze ingestion
cd ../..
python ingestion/ingest_fhir_bronze.py --source ./ingestion/synthea/output/fhir
```

### 4.2 — Architecture Diagram
Draw the full pipeline using [draw.io](https://draw.io) (free).

Show: `Synthea → ADLS Gen2 → Databricks Bronze → Silver → Gold → Unity Catalog → Power BI + ML`

Highlight the FADP compliance layer and Unity Catalog as separate components. Export as `docs/architecture.png` and embed in this README.

### 4.3 — Power BI Dashboard
Connect Power BI Desktop to the Gold layer via the Databricks SQL connector.

Visuals to build:
- Patient population overview — age distribution, top 10 conditions
- 30-day readmission rate by condition type
- Top risk patients table — `readmission_risk_score` + `top_risk_factors`
- SHAP feature importance chart

### 4.4 — README Final Polish
Once the pipeline has run and the ML model has been evaluated:
- Embed `docs/architecture.png`
- Fill in real AUC-ROC, precision, recall scores
- Add SHAP feature importance plot to `docs/`

---

## About

Built as a portfolio project to demonstrate production-grade data engineering skills in regulated healthcare environments, targeting Swiss pharma and MedTech companies (Roche, Novartis, Lonza).

Need to fix the screenshots and all pictures
