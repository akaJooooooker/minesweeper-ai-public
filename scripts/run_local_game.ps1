$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonWindowed = Join-Path $projectRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $pythonWindowed)) {
    throw "Missing .venv\Scripts\pythonw.exe. Create the project virtual environment first."
}
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
Start-Process -FilePath $pythonWindowed -ArgumentList "-m", "minesweeper_ai.local_app" -WorkingDirectory $projectRoot
