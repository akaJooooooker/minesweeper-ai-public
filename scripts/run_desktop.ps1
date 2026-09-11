$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonWindowed = Join-Path $projectRoot ".venv\Scripts\pythonw.exe"

if (-not (Test-Path -LiteralPath $pythonWindowed)) {
    throw "Missing .venv\Scripts\pythonw.exe. Create the project virtual environment first."
}

Set-Location -LiteralPath $projectRoot
& $pythonWindowed -m minesweeper_ai.desktop_app
