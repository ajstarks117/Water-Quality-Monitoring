#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Water Quality Monitoring - Environment Bootstrap (macOS/Linux)"
echo "============================================================"

# Check Python version (>= 3.10 required)
PYTHON_BIN=""
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "ERROR: Python is not installed or not found on PATH."
    exit 1
fi

PYTHON_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.minor)")

echo "Detected Python version: $PYTHON_VER (using $PYTHON_BIN)"

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required. Detected Python $PYTHON_VER."
    echo "Please install Python 3.10 or Python 3.11 and re-run setup."
    exit 1
fi

# Create virtual environment if not present
if [ ! -d "venv" ]; then
    echo "Creating virtual environment in ./venv..."
    $PYTHON_BIN -m venv venv
else
    echo "Virtual environment ./venv already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
# shellcheck source=/dev/null
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing pinned dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo "Verifying core package imports..."
python -c "import pandas, sklearn, xgboost, shap, streamlit; print('SUCCESS: Core ML/XAI/UI packages successfully imported!')"

echo ""
echo "============================================================"
echo " Setup Completed Successfully!"
echo " To activate your virtual environment:"
echo "   source venv/bin/activate"
echo "============================================================"
