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
| Ingestion | [ingestion/](ingestion/) | Load raw FHIR JSON bundles from ADLS into Delta |
| Bronze | [transformation/bronze/](transformation/bronze/) | Raw FHIR data, partitioned by resource type |
| Silver | [transformation/silver/](transformation/silver/) | Cleaned, parsed, joined patient records |
| Gold | [transformation/gold/](transformation/gold/) | Aggregated KPIs, readmission flags, risk scores |
| ML | [ml/](ml/) | XGBoost readmission model, SHAP explainability |
| Governance | [governance/](governance/) | Unity Catalog setup, FADP masking, audit logging |
| Dashboard | [dashboard/](dashboard/) | Power BI reports connected to Gold layer |

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
- [ ] Phase 2 — Medallion pipeline (Bronze → Silver → Gold)
- [ ] Phase 3 — Unity Catalog + FADP + ML model
- [ ] Phase 4 — Power BI dashboard + architecture diagram + polish

---

## About

Built as a portfolio project to demonstrate production-grade data engineering skills in regulated healthcare environments, targeting Swiss pharma and MedTech companies (Roche, Novartis, Lonza).
