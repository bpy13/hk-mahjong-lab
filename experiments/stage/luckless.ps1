# Plan 2026-11-16 Task 29, part 2 (gpu lane, job name hk-luckless): a 24-hour self-play run with
# the luck taken out of the reward. Task 10's settings (lownoise.ps1: same start network and
# opponent pool, pull toward the start constant at 0.02, 4,096 hands per update, one pass,
# learning rate 3e-5, stall rule off, a result against three `stage` every hour, a pool member
# every 2 h) plus the setting part 1's table chose (-Setting, see below), and a copy of the
# network every hour (--keep-hourly: runs\luckless\hour-NN.pt / .mjnn), so any hour can be
# matched later. Hourly row: learner minus frozen start on the same tiles with its 95 % range,
# entropy, distance from the start (runs\luckless\hourly.txt).
#
# -Setting (the trainer's switches, python\mahjong\rl\ppo_cnn.py "Luck taken out"):
#   paired       same-deal baseline only (Task 10's run; no new switch)
#   both         same-deal baseline + the oracle critic as baseline (--oracle-baseline)
#   perdecision  both, with per-decision credit (--oracle-baseline --per-decision-credit)
#   oracle       unpaired, the oracle critic as baseline (--no-paired-reward --oracle-baseline)
#
# Resumable: adding the job again, with the same -Setting, continues from the last checkpoint
# (state.json); a run that has stopped exits at once.
#
# Once, before queueing (freezes this checkout's package and extension against later edits;
# build mahjong._core into this checkout first):
#   powershell -NoProfile -File experiments\stage\luckless.ps1 -Freeze
# Queue:
#   powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane gpu -Label hk-luckless -Script "<full path of this file>" -ScriptArgs "-Setting both"
# Check:  Get-Content runs\luckless\hourly.txt ; runs\luckless\progress.txt ; runs\luckless\stopped.txt
#
# -DryRun: a check of this script on the cpu (4 threads), 6 minutes, a score row and a result
# every 2 minutes (200 deals), 256 hands per update, in runs\ppo\smoke\luckless-dry, without
# freezing (runs this checkout's python folder).
#
# -Final (cpu lane, after the run): the end network runs\luckless\final.mjnn plus three strict
# `stage` bots, 40,000 deals (seats rotated; this checkout's target\release\examples\compare.exe,
# build it first: cargo build --release --example compare -p mahjong-search), into
# runs\luckless\final\vs-stage-40000.txt.
#   jobq.ps1 add -Lane cpu -Label hk-luckless-final -After <run job> -Script "<this file>" -ScriptArgs "-Final"
param([string]$Init = 'models\stage-imitation.pt', [string]$Setting = 'both', [double]$Hours = 24,
      [switch]$Freeze, [switch]$DryRun, [switch]$Final)

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$runs = Join-Path $repo 'runs\luckless'
$pkg = Join-Path $runs 'pkg'

if ($Final) {
    $net = Join-Path $runs 'final.mjnn'
    $compare = Join-Path $repo 'target\release\examples\compare.exe'
    foreach ($f in $net, $compare, (Join-Path $runs 'stopped.txt')) {
        if (-not (Test-Path $f)) { Write-Output "missing $f (the run has not ended, or compare is not built)"; exit 1 }
    }
    $dir = Join-Path $runs 'final'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '16' }
    & $compare "net:path=$net" stage stage stage 40000 1 | Out-File -Encoding utf8 (Join-Path $dir 'vs-stage-40000.txt')
    exit $LASTEXITCODE
}

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

switch ($Setting) {
    'paired'      { $luck = @() }
    'both'        { $luck = @('--oracle-baseline') }
    'perdecision' { $luck = @('--oracle-baseline', '--per-decision-credit') }
    'oracle'      { $luck = @('--no-paired-reward', '--oracle-baseline') }
    default       { Write-Output "unknown -Setting $Setting (paired, both, perdecision, oracle)"; exit 1 }
}
if ($Setting -eq 'oracle') {
    # The opponent pool needs paired games; an unpaired run plays the current network in all seats.
    $pool = @()
} else {
    $pool = @('--pool')
}

# Task 10's settings plus the luck switches and the hourly copies.
$common = @('--epochs', '1', '--lr', '3e-5', '--pull', '0.02', '--no-pull-fade', '--no-stall',
            '--anchor-weight', '0', '--no-curriculum', '--keep-hourly') + $pool + $luck
if ($DryRun) {
    $out = Join-Path $repo 'runs\ppo\smoke\luckless-dry'
    $env:PYTHONPATH = Join-Path $repo 'python'
    $env:RAYON_NUM_THREADS = '4'
    $a = @('--minutes', '6', '--device', 'cpu', '--torch-threads', '4', '--envs', '64',
           '--hands-per-iter', '256', '--batch', '512', '--score-minutes', '2',
           '--ckpt-minutes', '2', '--eval-minutes', '2', '--pool-minutes', '2', '--eval-deals', '200')
} else {
    $out = $runs
    if (-not (Test-Path (Join-Path $pkg 'mahjong\_core.pyd'))) { Write-Output 'run with -Freeze first'; exit 1 }
    $env:PYTHONPATH = $pkg
    $env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
    $a = @('--hours', "$Hours", '--device', 'cuda', '--priority', 'above_normal',
           '--hands-per-iter', '4096', '--score-minutes', '60',
           '--ckpt-minutes', '30', '--eval-minutes', '60', '--pool-minutes', '120',
           '--eval-deals', '2000')
}
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $repo
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out 'selfplay.out'

# The package must come from this checkout (the conda env is shared and holds an editable
# install of another checkout); the first line of the log names it.
$file = & $python -c 'import mahjong; print(mahjong.__file__)'
Write-Output "mahjong.__file__ = $file"
Add-Content -Encoding utf8 -Path $log -Value "mahjong.__file__ = $file ; setting $Setting"
if (-not $file -or -not ([IO.Path]::GetFullPath($file)).StartsWith($repo, [StringComparison]::OrdinalIgnoreCase)) {
    Write-Output "mahjong is not imported from $repo"
    Add-Content -Encoding utf8 -Path $log -Value "mahjong is not imported from $repo"
    exit 1
}

$fresh = -not (Test-Path (Join-Path $out 'state.json'))
if ($fresh -and -not (Test-Path $Init)) { Write-Output "missing $Init (Task 4b's imitation network)"; exit 1 }
# -Init is ignored when resuming (the run continues from state.json); -Setting must be the same.

& $python -u -m mahjong.rl.ppo_cnn --init $Init --out $out --resume @common @a 2>&1 |
    ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
exit $LASTEXITCODE
