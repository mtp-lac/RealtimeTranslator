param(
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$pyinstaller = Join-Path $PSScriptRoot ".venv\Scripts\pyinstaller.exe"
if (-not (Test-Path $pyinstaller)) {
    throw "PyInstaller not found in .venv. Run: .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt"
}

if ($Cuda) {
    $env:RTT_BUILD_CUDA = "1"
    Write-Host "Building RealtimeTranslator (CUDA + cuDNN runtime included)..."
} else {
    $env:RTT_BUILD_CUDA = "0"
    Write-Host "Building RealtimeTranslator (CPU runtime)..."
}

& $pyinstaller --noconfirm RealtimeTranslator.spec
Write-Host "Build output: $(Join-Path $PSScriptRoot 'dist\RealtimeTranslator')"
