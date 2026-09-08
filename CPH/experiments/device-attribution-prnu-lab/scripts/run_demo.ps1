param(
  [string]$Dataset = "dataset",
  [string]$ModelDir = "models",
  [int]$Port = 8002,
  [switch]$AllowDemoModel
)

$ErrorActionPreference = "Stop"

$candidates = @(
  $env:PYTHON,
  "python"
) | Where-Object { $_ -and $_.Trim() }

$python = $null
foreach ($candidate in $candidates) {
  if (Test-Path $candidate) {
    $python = $candidate
    break
  }
  $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
  if ($cmd) {
    $python = $candidate
    break
  }
}

if (-not $python) {
  throw "No Python executable found."
}

$launchArgs = @(
  "-m", "prnu_attribution.api",
  "--host", "127.0.0.1",
  "--port", "$Port",
  "--dataset", $Dataset,
  "--model-dir", $ModelDir
)
if ($AllowDemoModel) {
  $launchArgs += "--allow-demo-model"
}
& $python @launchArgs
