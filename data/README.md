# Data Directory

This directory contains data files at various stages of the data processing pipeline.

## Structure
- `raw/`: Original, immutable raw water quality datasets (e.g. Kaggle Water Quality, EPA, or IoT sensor logs). Never edit files in this folder directly.
- `interim/`: Intermediate transformed datasets undergoing cleaning, imputation, or feature engineering.
- `processed/`: Final canonical datasets used for training and evaluation.

## Policy
- Large raw data files (`data/raw/*`) and intermediate data files (`data/interim/*`) are excluded from Git version control via `.gitignore`.
- Download links and replication instructions for raw datasets will be documented here as datasets are added.
