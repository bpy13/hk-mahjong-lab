# Plan 2026-11-16 Task 10: a 4-hour self-play run with less noise and an accurate hourly score
# (gpu lane, job name rl-lownoise). Same start network and opponent pool as Task 6
# (selfplay.ps1); changed (docs/study-results/2026-11-17-self-play-analysis.md, experiment 4):
#   - the pull toward the start constant at 0.02 (--no-pull-fade), not 0.1 fading by hour 24;
#   - 4,096 hands per update (was 1,024), one pass per update (was 3), learning rate 3e-5;
#   - the stall rule off (--no-stall); the draw and collapse rules stay;
#   - every hour: the mean paired difference (learner minus the frozen start, points a hand)
#     with its 95 % range over that hour's hands, entropy and distance from the start, in
#     runs\rl-lownoise\hourly.txt (table), hourly.json and selfplay.out;
#   - a result against three `stage` bots (2,000 deals) every hour, so the collapse rule
#     can act within 4 hours; a pool member every 2 h as in Task 6.
# Ends after 4 h of training (across restarts): runs\rl-lownoise\final.pt / final.mjnn.
#
# Resumable: adding the job again continues from the last checkpoint (state.json); a run that
# has stopped exits at once.
#
# Once, before queueing (freezes this checkout's package and extension against later edits;
# build mahjong._core into this checkout first):
#   powershell -NoProfile -File experiments\stage\lownoise.ps1 -Freeze
# Queue:
#   powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane gpu -Label rl-lownoise -Script "<full path of this file>"
# Check:  Get-Content runs\rl-lownoise\hourly.txt ; runs\rl-lownoise\progress.txt ; runs\rl-lownoise\stopped.txt
#
# -DryRun: a check of this script on the cpu (4 threads), 6 minutes, a score row and a result
# every 2 minutes (200 deals), 256 hands per update, in runs\ppo\smoke\lownoise-dry, without
# freezing (runs this checkout's python folder).
#
# -Final (cpu lane, after the run): the end network runs\rl-lownoise\final.mjnn plus three strict
# `stage` bots, 40,000 deals (seats rotated, Task 7's compare.exe in runs\rl-stage\bin), into
# runs\rl-lownoise\final\vs-stage-40000.txt; set beside Task 7's -0.09 +- 0.22.
#   jobq.ps1 add -Lane cpu -Label rl-lownoise-final -After <run job> -Script "<this file>" -ScriptArgs "-Final"
param([string]$Init = 'models\stage-imitation.pt', [switch]$Freeze, [switch]$DryRun, [switch]$Final)

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$runs = Join-Path $repo 'runs\rl-lownoise'
$pkg = Join-Path $runs 'pkg'

if ($Final) {
    $net = Join-Path $runs 'final.mjnn'
    $compare = Join-Path $repo 'runs\rl-stage\bin\compare.exe'
    foreach ($f in $net, $compare, (Join-Path $runs 'stopped.txt')) {
        if (-not (Test-Path $f)) { Write-Output "missing $f (the run has not ended)"; exit 1 }
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

# Task 10's settings (the same in the dry run, except its length and sizes).
$common = @('--epochs', '1', '--lr', '3e-5', '--pull', '0.02', '--no-pull-fade', '--no-stall',
            '--anchor-weight', '0', '--no-curriculum', '--pool')
if ($DryRun) {
    $out = Join-Path $repo 'runs\ppo\smoke\lownoise-dry'
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
    $a = @('--hours', '4', '--device', 'cuda', '--priority', 'above_normal',
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
Add-Content -Encoding utf8 -Path $log -Value "mahjong.__file__ = $file"
if (-not $file -or -not ([IO.Path]::GetFullPath($file)).StartsWith($repo, [StringComparison]::OrdinalIgnoreCase)) {
    Write-Output "mahjong is not imported from $repo"
    Add-Content -Encoding utf8 -Path $log -Value "mahjong is not imported from $repo"
    exit 1
}

$fresh = -not (Test-Path (Join-Path $out 'state.json'))
if ($fresh -and -not (Test-Path $Init)) { Write-Output "missing $Init (Task 4b's imitation network)"; exit 1 }
# -Init is ignored when resuming: the run continues from state.json.

& $python -u -m mahjong.rl.ppo_cnn --init $Init --out $out --resume @common @a 2>&1 |
    ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
exit $LASTEXITCODE
