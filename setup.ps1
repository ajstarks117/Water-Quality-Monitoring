# Water Quality Monitoring - Environment Bootstrap (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Water Quality Monitoring - Environment Bootstrap (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Locate Python
$pythonCmd = $null
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
} else {
    Write-Error "ERROR: Python is not installed or not found on PATH."
    exit 1
}

# Check Python version (>= 3.10 required)
$versionOutput = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$major = [int](& $pythonCmd -c "import sys; print(sys.version_info.major)")
$minor = [int](& $pythonCmd -c "import sys; print(sys.version_info.minor)")

Write-Host "Detected Python version: $versionOutput (using $pythonCmd)" -ForegroundColor Yellow

if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    Write-Error "ERROR: Python 3.10+ required. Detected Python $versionOutput. Please install Python 3.10, 3.11, or 3.12."
    exit 1
}

# Create virtual environment if not present
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment in .\venv..." -ForegroundColor Green
    & $pythonCmd -m venv venv
} else {
    Write-Host "Virtual environment .\venv already exists." -ForegroundColor Yellow
}

$venvPython = ".\venv\Scripts\python.exe"
$venvPip = ".\venv\Scripts\pip.exe"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
& $venvPython -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing pinned dependencies from requirements.txt..." -ForegroundColor Green
& $venvPip install -r requirements.txt

# Verify installation
Write-Host "Verifying core package imports..." -ForegroundColor Green
& $venvPython -c "import pandas, sklearn, xgboost, shap, streamlit; print('SUCCESS: Core ML/XAI/UI packages successfully imported!')"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Setup Completed Successfully!" -ForegroundColor Green
Write-Host " To activate your virtual environment in PowerShell:" -ForegroundColor Yellow
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host " Or in Command Prompt:" -ForegroundColor Yellow
Write-Host "   venv\Scripts\activate.bat" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
