# PCCF Execution Script using the configured Python 3.10 environment
$pythonPath = "C:\Users\elmas\AppData\Local\Programs\Python\Python310\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Error "Python 3.10 executable not found at: $pythonPath"
    exit 1
}

& $pythonPath -u main.py @args
