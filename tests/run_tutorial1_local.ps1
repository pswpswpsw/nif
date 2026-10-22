[CmdletBinding()]
param(
    [switch]$Smoke,
    [string]$EnvironmentPath = (Join-Path $env:USERPROFILE '.cache\nif-colab-qa\colab-matched')
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$uvExecutable = (Get-Command uv -ErrorAction Stop).Source
$pythonExecutable = Join-Path $EnvironmentPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    & $uvExecutable venv --python 3.13.11 $EnvironmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python environment.' }
}
& $uvExecutable pip install --python $pythonExecutable -r (Join-Path $projectRoot 'tutorial\colab_support\requirements-qa.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

$runMode = if ($Smoke) { 'smoke' } else { 'full' }
$outputPath = Join-Path $projectRoot ".codex\validation\tutorial1-colab\$runMode.colab-matched.executed.ipynb"
$workPath = Join-Path (Split-Path -Parent $EnvironmentPath) "colab-matched-$runMode-run"
$runArguments = @(
    (Join-Path $projectRoot 'tests\execute_tutorial1.py'),
    '--notebook', (Join-Path $projectRoot 'tutorial\1_simple_1d_wave_colab.ipynb'),
    '--output', $outputPath,
    '--workdir', $workPath
)
if ($Smoke) { $runArguments += '--smoke' }
& $pythonExecutable @runArguments
if ($LASTEXITCODE -ne 0) { throw 'A notebook cell failed. Inspect the executed notebook and traceback.' }
Write-Output "Executed notebook: $outputPath"
