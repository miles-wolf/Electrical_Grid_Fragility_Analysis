# Makefile (run commands from project root)
# Usage:
#   make ingest
#   make qc
#   make all
#   make clean
#   make features

.PHONY: ingest features fragility_daily qc pipeline

ingest:
	python src/ingest_eia930.py

qc:
	python src/qc_eia930.py

features:
	python -m src.features_eia930 \
	  --in data/processed/eia930/eia930_balance_2025_pjm_ciso.parquet \
	  --out data/processed/features/eia930_daily_features_2025_pjm_ciso.parquet

# Daily fragility metrics + plots
fragility_daily:
	python src/fragility_daily.py

# Full end-to-end pipeline
pipeline: ingest features fragility_daily qc


