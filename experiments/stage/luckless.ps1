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
#   oracle       unpaired, the oracle critic as baseline (--no-paired-reward --oracle-baseline);
#                the learner plays all four seats itself (no pool: the pool needs paired games),
#                the hourly row has no paired figure (hands, entropy, distance only), and the
#                hourly result is 2,000 hands against three `value` bots and 2,000 against three
#                frozen start networks (the trainer's evaluate()). Chosen on 2026-11-23 (plan
#                Task 29 CHANGE 16:30, Brian: "okay") after the `both` run was stopped at 1.3 h.
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
# -Hourly (cpu lane, after the run; the unpaired setting has no hourly paired figure): the hourly
# copies hour-04, -08, -12, -16, -20, -24 (-HourList) and the frozen start (runs\luckless\teacher.mjnn)
# each plus three strict `stage` bots on the same 20,000 deals (-HourDeals, seed 2), into
# runs\luckless\hourly-match\<name>.txt, so the report can say whether the network was still
# rising at hour 24 (each copy's gain minus the start's on the same deals).
#   jobq.ps1 add -Lane cpu -Label hk-luckless-hourly -After <run job> -Script "<this file>" -ScriptArgs "-Hourly"
#
# Task 34 (plan 2026-11-16): 24 more hours of the `oracle` run in a folder of its own, so Task 29's
# runs\luckless (its hour-24 network is on the game server) stays as it is. -Run names the run
# folder under runs\ (default luckless). Once, before queueing:
#   luckless.ps1 -Run luckless48 -Freeze
#   luckless.ps1 -Run luckless48 -ContinueFrom luckless
# (-ContinueFrom copies the source's state.json, its newest checkpoint, the frozen start, the log
# and the hourly rows, and the source's end network as hour-24.pt / .mjnn and its hourly-copy
# matches, so the run resumes at hour 24 and the copies' hour numbers go on from 25.) Then one
# gpu job per 4 hours, -Hours set to the run's total at the job's end (28, 32, ... 48), each after
# the one before: a run stopped by its time goes on when resumed with more time.
# -Copy <hour> (cpu lane, one job after each 4-hour gpu job; Brian, 2026-11-27: "during the queued
# continuation training it should also match against its older version like in big2"): the copy
# hour-<hour> (1) plus three strict `stage` on Task 29's 20,000 deals (seed 2, as -Hourly), into
# hourly-match\hour-<hour>-vs-stage-20000.txt; (2) head to head, networks only, against the copy
# 4 hours before and (3) against hour 24 (the start of this run), each 2v2 in both seatings
# (A B A B and A A B B, rotated) on the same -H2HDeals walls (7,500, seed 4: about +-0.3 a hand
# from a 300-wall pilot), with bin\h2h.exe, into h2h\hour-<hour>-vs-hour-<other>.txt. Uses the
# copies of compare.exe and h2h.exe that -ContinueFrom puts in runs\<Run>\bin, so a later build
# in this checkout cannot change a match half way. A match whose file is complete is skipped.
param([string]$Init = 'models\stage-imitation.pt', [string]$Setting = 'both', [double]$Hours = 24,
      [string]$Run = 'luckless', [string]$ContinueFrom = '', [int]$Copy = 0, [int]$H2HDeals = 7500,
      [switch]$Freeze, [switch]$DryRun, [switch]$Final, [switch]$Hourly,
      [string]$HourList = '4,8,12,16,20,24', [int]$HourDeals = 20000)

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$runs = Join-Path $repo "runs\$Run"
$pkg = Join-Path $runs 'pkg'

if ($ContinueFrom) {
    $src = Join-Path $repo "runs\$ContinueFrom"
    if (-not (Test-Path (Join-Path $pkg 'mahjong\_core.pyd'))) { Write-Output "run -Run $Run -Freeze first"; exit 1 }
    if (Test-Path (Join-Path $runs 'state.json')) { Write-Output "$runs has a run already: not seeding it"; exit 1 }
    $state = Get-Content -Raw (Join-Path $src 'state.json') | ConvertFrom-Json
    if (-not ("$($state.stopped)".StartsWith('time:'))) { Write-Output "$src has not stopped on its time"; exit 1 }
    $files = @($state.checkpoint, 'teacher.pt', 'teacher.mjnn', 'log.json', 'hourly.json', 'hourly.txt',
               'progress.txt', 'stopped.txt', 'state.json')
    foreach ($f in $files) {
        $from = Join-Path $src $f
        if (-not (Test-Path $from)) { Write-Output "missing $from"; exit 1 }
        Copy-Item $from (Join-Path $runs $f)
    }
    $h = [int][math]::Floor($state.elapsed_s / 3600)
    Copy-Item (Join-Path $src 'final.pt') (Join-Path $runs ('hour-{0:D2}.pt' -f $h))
    Copy-Item (Join-Path $src 'final.mjnn') (Join-Path $runs ('hour-{0:D2}.mjnn' -f $h))
    $bin = Join-Path $runs 'bin'
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    foreach ($exe in 'compare.exe', 'h2h.exe') {
        $from = Join-Path $repo "target\release\examples\$exe"
        if (-not (Test-Path $from)) { Write-Output "missing $from (cargo build --release --example compare --example h2h -p mahjong-search)"; exit 1 }
        Copy-Item $from (Join-Path $bin $exe)
    }
    # The source's hourly-copy matches were played on the same deals as -Hourly plays here.
    $srcMatch = Join-Path $src 'hourly-match'
    if (Test-Path $srcMatch) { Copy-Item -Recurse $srcMatch (Join-Path $runs 'hourly-match') }
    Write-Output "seeded $runs from $src ($($state.checkpoint), iteration $($state.iteration), $([math]::Round($state.elapsed_s / 3600, 2)) h)"
    exit 0
}

if ($Copy -gt 0) {
    $bin = Join-Path $runs 'bin'
    $compare = Join-Path $bin 'compare.exe'
    $h2h = Join-Path $bin 'h2h.exe'
    $net = Join-Path $runs ('hour-{0:D2}.mjnn' -f $Copy)
    foreach ($f in $compare, $h2h, $net) {
        if (-not (Test-Path $f)) { Write-Output "missing $f (the gpu job has not reached hour $Copy, or not seeded)"; exit 1 }
    }
    $env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '16' }
    $code = 0
    $dir = Join-Path $runs 'hourly-match'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $outFile = Join-Path $dir ('hour-{0:D2}-vs-stage-{1}.txt' -f $Copy, $HourDeals)
    if (-not ((Test-Path $outFile) -and (Get-Content $outFile | Select-String 'draws'))) {
        & $compare "net:path=$net" stage stage stage $HourDeals 2 | Out-File -Encoding utf8 $outFile
        if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE }
    }
    $dir = Join-Path $runs 'h2h'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($other in @(($Copy - 4), 24) | Select-Object -Unique) {
        if ($other -ge $Copy) { continue }
        $old = Join-Path $runs ('hour-{0:D2}.mjnn' -f $other)
        if (-not (Test-Path $old)) { Write-Output "missing $old"; $code = 1; continue }
        $outFile = Join-Path $dir ('hour-{0:D2}-vs-hour-{1:D2}.txt' -f $Copy, $other)
        if ((Test-Path $outFile) -and (Get-Content $outFile | Select-String 'draws')) { continue }
        & $h2h "net:path=$net" "net:path=$old" $H2HDeals 4 | Out-File -Encoding utf8 $outFile
        if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE }
    }
    exit $code
}

if ($Final) {
    $net = Join-Path $runs 'final.mjnn'
    $compare = Join-Path $runs 'bin\compare.exe'
    if (-not (Test-Path $compare)) { $compare = Join-Path $repo 'target\release\examples\compare.exe' }
    foreach ($f in $net, $compare, (Join-Path $runs 'stopped.txt')) {
        if (-not (Test-Path $f)) { Write-Output "missing $f (the run has not ended, or compare is not built)"; exit 1 }
    }
    $dir = Join-Path $runs 'final'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '16' }
    & $compare "net:path=$net" stage stage stage 40000 1 | Out-File -Encoding utf8 (Join-Path $dir 'vs-stage-40000.txt')
    exit $LASTEXITCODE
}

if ($Hourly) {
    $compare = Join-Path $repo 'target\release\examples\compare.exe'
    foreach ($f in $compare, (Join-Path $runs 'stopped.txt'), (Join-Path $runs 'teacher.mjnn')) {
        if (-not (Test-Path $f)) { Write-Output "missing $f (the run has not ended, or compare is not built)"; exit 1 }
    }
    $dir = Join-Path $runs 'hourly-match'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '16' }
    $nets = @(@{ name = 'start'; path = (Join-Path $runs 'teacher.mjnn') })
    foreach ($h in ($HourList -split ',' | ForEach-Object { [int]$_ })) {
        $p = Join-Path $runs ('hour-{0:D2}.mjnn' -f $h)
        if (Test-Path $p) { $nets += @{ name = ('hour-{0:D2}' -f $h); path = $p } }
        else { Write-Output "no copy for hour $h ($p): skipped" }
    }
    $code = 0
    foreach ($n in $nets) {
        $outFile = Join-Path $dir ("$($n.name)-vs-stage-$HourDeals.txt")
        if ((Test-Path $outFile) -and (Get-Content $outFile | Select-String 'draws')) { continue }  # done in an earlier attempt
        & $compare "net:path=$($n.path)" stage stage stage $HourDeals 2 | Out-File -Encoding utf8 $outFile
        if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE }
    }
    exit $code
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
    # Without the pool the hourly result is evaluate(): --eval-walls walls x 4 seatings.
    if ($Setting -eq 'oracle') { $a += @('--eval-walls', '500') }
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
