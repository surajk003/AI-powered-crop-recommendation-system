<#
build_exe.ps1

Usage (PowerShell):
  # create venv and install dependencies
  python -m venv .venv; .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt pyinstaller

  # build single-file exe (from project root)
  .\build_exe.ps1

Notes:
- This script creates a single-file PyInstaller build of `run_app.py`. Adjust the --add-data options if you need to bundle the CSV or model files.
#>

param()

Write-Host "Starting PyInstaller build..."

$proj = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $proj

# Clean previous builds
Remove-Item -Recurse -Force "dist","build","run_app.spec" -ErrorAction SilentlyContinue

# Example: include the default model files if you want them bundled (optional)
# $dataArgs = "--add-data ""crop_model.joblib;."" --add-data ""crop_preprocessor.joblib;."""
$dataArgs = ""

# Build single-file executable
pyinstaller --noconsole --onefile $dataArgs run_app.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Build completed. See the dist\run_app.exe file."
