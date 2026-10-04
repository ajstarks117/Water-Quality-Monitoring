# Water Quality Monitoring Makefile
.PHONY: help setup install verify test clean run-app

# Python interpreter
PYTHON ?= python

help:
	@echo "Water Quality Monitoring - Available commands:"
	@echo "  make setup     Create virtual environment and install pinned dependencies"
	@echo "  make install   Install requirements into active environment"
	@echo "  make verify    Run package import verification check"
	@echo "  make test      Run pytest suite"
	@echo "  make clean     Remove virtual environment and cache files"

setup:
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'"
	@$(PYTHON) -m venv venv
	@echo "Virtual environment created. Installing dependencies..."
	@./venv/bin/pip install --upgrade pip || .\venv\Scripts\pip.exe install --upgrade pip
	@./venv/bin/pip install -r requirements.txt || .\venv\Scripts\pip.exe install -r requirements.txt
	@echo "Setup completed successfully."

install:
	pip install --upgrade pip
	pip install -r requirements.txt

verify:
	python -c "import pandas, sklearn, xgboost, shap, streamlit; print('All core packages successfully imported!')"

test:
	pytest tests/ -v

clean:
	rm -rf venv .pytest_cache __pycache__ .ipynb_checkpoints
