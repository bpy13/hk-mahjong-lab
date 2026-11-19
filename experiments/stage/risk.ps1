# Plan 2026-11-16 Task 15: the risk network. Three steps, each one may be run alone:
#   -Step data   log deals of four `stage` bots with the true per-kind answers (cpu, about
#                5 min per 30,000 deals on 8 threads; resumes after the last complete chunk)
#   -Step fit    fit the network on the cpu (or -Device cuda in the gpu lane), about 1 hour
#   -Step check  the ten-bin tables, scores and verdict on the held-out chunks (Markdown)
# Everything lands in runs\risk\ (data\, the fit in -FitDir with its check.md); the exported
# network is <FitDir>\risk.mjnn, copied to models\risk.mjnn (the last fit to finish wins).
#
# Usage (from this checkout; the binary is a scratch build outside every checkout):
#   jobq.ps1 add -Lane cpu -Dir <checkout> -Script <checkout>\experiments\stage\risk.ps1 -ScriptArgs "-Step data"
#   jobq.ps1 add -Lane cpu -After <data job> -Dir <checkout> -Script <checkout>\experiments\stage\risk.ps1 -ScriptArgs "-Step fit"
#   jobq.ps1 add -Lane cpu -After <fit job> -Dir <checkout> -Script <checkout>\experiments\stage\risk.ps1 -ScriptArgs "-Step check"
# 2026-11-19: all three steps ran directly from the worker's checkout (orchestrator's call),
# with the defaults below (that check wrote runs\risk\check.md). Follow-up in the gpu lane, a
# longer fit at a lower learning rate:
#   -Step fit -Device cuda -Epochs 30 -Lr 0.0005 -Minutes 180 -FitDir runs\risk\fit-gpu
#   -Step check -FitDir runs\risk\fit-gpu
param(
    [ValidateSet('data', 'fit', 'check', 'all')][string]$Step = 'all',
    [int]$Deals = 30000,
    [int]$Chunk = 1000,
    [int]$Seed = 15,
    [int[]]$Holdout = @(27, 28, 29),
    [int]$Epochs = 6,
    [int]$Minutes = 100,
    [int]$Threads = 8,
    [string]$Device = 'cpu',
    [double]$Lr = 0.001,
    [string]$FitDir = 'runs\risk\fit',
    [string]$Bin = 'C:\Users\brian\.cache\hk-mahjong-t15\release\examples'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

# The shared conda env may import another checkout's package: insist on this one.
$py = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONPATH = Join-Path $root 'python'
$where = & $py -c "import mahjong; print(mahjong.__file__)"
Write-Output "mahjong.__file__ = $where"
if (-not $where -or -not ([IO.Path]::GetFullPath($where)).StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { Write-Output "mahjong is not imported from $root"; exit 1 }

$data = Join-Path $root 'runs\risk\data'
$fit = Join-Path $root $FitDir
$holdout = $Holdout | ForEach-Object { "$_" }

if ($Step -eq 'data' -or $Step -eq 'all') {
    $exe = Join-Path $Bin 'risk_log.exe'
    if (-not (Test-Path $exe)) { Write-Output "missing $exe (cargo build --release -p mahjong-search --example risk_log with CARGO_TARGET_DIR outside the checkout)"; exit 1 }
    $env:RAYON_NUM_THREADS = "$Threads"
    & $exe --deals $Deals --chunk $Chunk --seed $Seed --out $data --belief (Join-Path $root 'models\belief.mjnn')
    if ($LASTEXITCODE -ne 0) { Write-Output "risk_log exited with code $LASTEXITCODE"; exit $LASTEXITCODE }
}
if ($Step -eq 'fit' -or $Step -eq 'all') {
    $fitArgs = @('-m', 'mahjong.risk.fit', '--data', $data, '--out', $fit, '--models', (Join-Path $root 'models'),
        '--holdout-chunks') + $holdout + @('--epochs', "$Epochs", '--minutes', "$Minutes",
        '--threads', "$Threads", '--device', $Device, '--lr', "$Lr")
    if (Test-Path (Join-Path $fit 'ckpt.pt')) { $fitArgs += '--resume' }
    & $py @fitArgs
    if ($LASTEXITCODE -ne 0) { Write-Output "fit exited with code $LASTEXITCODE"; exit $LASTEXITCODE }
}
if ($Step -eq 'check' -or $Step -eq 'all') {
    $checkArgs = @('-m', 'mahjong.risk.check', '--data', $data, '--model', (Join-Path $fit 'best.pt'),
        '--out', (Join-Path $fit 'check.md'), '--threads', "$Threads", '--chunks') + $holdout
    & $py @checkArgs
    exit $LASTEXITCODE
}
