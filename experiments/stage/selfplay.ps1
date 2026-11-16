# Plan 2026-11-16 Task 5/6: self-play of the network against a pool of opponents (gpu lane).
# python/mahjong/rl/ppo_cnn.py --pool: each of the learner's three opponents is drawn per hand
# (current network 50 %, a saved older network 30 %, `stage` 20 %); paired reward against the
# frozen init (imitation) network in the same seat; a pull KL(init || net) on every decision
# that fades linearly to zero by hour 24; a checkpoint every 30 min, a pool member per 2 h;
# every 2 h a result of 2,000 deals against three `stage` bots and against each earlier pool
# member in runs\rl-stage\log.json; stops at 48 h of training time (across restarts) or by
# the plan's rules (draws above 15 %, or four results in a row without a gain), written to
# runs\rl-stage\stopped.txt. Do not change these settings between restarts (Task 6).
#
# Resumable: adding the job again continues from the last checkpoint (state.json); a run that
# has stopped exits at once.
#
# Once, before queueing (freezes this checkout's package and extension against later edits;
# build mahjong._core into this checkout first):
#   powershell -NoProfile -File experiments\stage\selfplay.ps1 -Freeze
# Queue (Init: Task 4's imitation network, by default models\stage-imitation.pt; a SuitCNN .pt):
#   powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane gpu -Label rl-stage -Script "<full path of this file>" -ScriptArgs "-Init <full path of the .pt>"
# Check:  Get-Content runs\rl-stage\progress.txt ; runs\rl-stage\log.json ; runs\rl-stage\stopped.txt
#
# -DryRun: the Task 5 check on the cpu (4 threads), times scaled down (save every 2 min, a result
# every 5 min with 200 deals, a pool member every 5 min, pull fading by 20 min), 20 minutes,
# in runs\ppo\smoke\selfplay-dry, without freezing (runs this checkout's python folder).
param([string]$Init = 'models\stage-imitation.pt', [switch]$Freeze, [switch]$DryRun)

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$runs = Join-Path $repo 'runs\rl-stage'
$pkg = Join-Path $runs 'pkg'

if ($Freeze) {
    if (-not (Test-Path (Join-Path $repo 'python\mahjong\_core.pyd'))) { Write-Output 'build mahjong._core first'; exit 1 }
    if (Test-Path (Join-Path $runs 'state.json')) { Write-Output "$runs has a run already: not refreezing"; exit 1 }
    New-Item -ItemType Directory -Force -Path $runs | Out-Null
    if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
    New-Item -ItemType Directory -Force -Path $pkg | Out-Null
    Copy-Item -Recurse (Join-Path $repo 'python\mahjong') (Join-Path $pkg 'mahjong')
    Get-ChildItem -Path $pkg -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
    Write-Output "frozen into $pkg"
    exit 0
}

if ($DryRun) {
    $out = Join-Path $repo 'runs\ppo\smoke\selfplay-dry'
    $env:PYTHONPATH = Join-Path $repo 'python'
    $env:RAYON_NUM_THREADS = '4'
    $a = @('--minutes', '20', '--device', 'cpu', '--torch-threads', '4', '--envs', '32',
           '--hands-per-iter', '64', '--batch', '512',
           '--ckpt-minutes', '2', '--eval-minutes', '5', '--pool-minutes', '5', '--eval-deals', '200',
           '--pull-fade-hours', '0.3333')
} else {
    $out = $runs
    if (-not (Test-Path (Join-Path $pkg 'mahjong\_core.pyd'))) { Write-Output 'run with -Freeze first'; exit 1 }
    $env:PYTHONPATH = $pkg
    $env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
    $a = @('--hours', '48', '--device', 'cuda', '--priority', 'above_normal',
           '--ckpt-minutes', '30', '--eval-minutes', '120', '--pool-minutes', '120',
           '--eval-deals', '2000', '--pull-fade-hours', '24')
}
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $repo
New-Item -ItemType Directory -Force -Path $out | Out-Null

# The package must come from this checkout (the conda env is shared; never an editable install).
$file = & $python -c 'import mahjong; print(mahjong.__file__)'
Write-Output "mahjong.__file__ = $file"
if (-not $file -or -not ([IO.Path]::GetFullPath($file)).StartsWith($repo, [StringComparison]::OrdinalIgnoreCase)) {
    Write-Output "mahjong is not imported from $repo"; exit 1
}

$fresh = -not (Test-Path (Join-Path $out 'state.json'))
if ($fresh -and $Init -ne 'none' -and -not (Test-Path $Init)) { Write-Output "missing $Init (Task 4's imitation network)"; exit 1 }
# -Init is ignored when resuming: the run continues from state.json.

& $python -u -m mahjong.rl.ppo_cnn --init $Init --out $out --resume --pool --no-curriculum `
    --anchor-weight 0 --pull 0.1 @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $out 'selfplay.out')
exit $LASTEXITCODE
