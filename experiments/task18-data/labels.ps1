# Plan 2026-11-09 Task 18: teacher labels with the passed-win rule, for retraining the
# distilled network. Same teacher, field, size and seed as Task 13's labels
# (experiments\task13_label.ps1), in the new row layout 2 (adds `passed`,
# features::encode_passed; the faan_shanten fix also changes some features).
#
# The teacher and the other seats read a belief model. Use the retrained one
# (models\belief-v2.mjnn or wherever Task 18's belief training writes it) once it exists;
# docs\open-items.md has the question. Long: Task 13's 1 M rows took 14.4 hours on 8 threads.
#
# 1. Freeze the labeller once (a build, outside the queue):
#      powershell -NoProfile -File <this file> -Freeze
# 2. Queue it on the cpu lane (resumable: add it again after a failure or restart):
#      powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\bot-stream\tools\queue\hkq.ps1 add -Lane cpu -Script <this file> -Label task18-labels -ScriptArgs "-Belief 'models/belief.mjnn'"
#
# Output: runs\labels-v2 (chunks, meta.json, progress.txt). A rerun with a different
# -Belief into the same folder is refused by the labeller (one teacher per folder).
# Check:  Get-Content C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab\runs\labels-v2\progress.txt
param([switch]$Freeze, [string]$Belief = 'models/belief.mjnn', [int]$Decisions = 1000000)
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$job = Join-Path $repo 'runs\labels-v2-job'
$out = Join-Path $repo 'runs\labels-v2'
$exe = Join-Path $job 'bin\label.exe'
$log = Join-Path $job 'run.log'
New-Item -ItemType Directory -Force (Join-Path $job 'bin') | Out-Null
Set-Location $repo

function Log([string]$msg) { Add-Content -Path $log -Value "$(Get-Date -Format s) $msg" -Encoding utf8 }

if ($Freeze) {
    $env:PATH = "C:\Users\brian\.cargo\bin;" + $env:PATH
    $env:CARGO_BUILD_JOBS = '8'
    & cargo build --release -p mahjong-search --example label
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Copy-Item (Join-Path $repo 'target\release\examples\label.exe') $exe -Force
    $head = (& git -C $repo rev-parse --short HEAD)
    $dirty = if (& git -C $repo status --porcelain -- crates) { ' (crates had uncommitted changes)' } else { '' }
    Log "frozen label.exe from $head$dirty"
    exit 0
}
if (-not (Test-Path $exe)) { Write-Output "no frozen labeller in $exe; run with -Freeze first"; exit 1 }
if (-not (Test-Path (Join-Path $repo $Belief))) { Log "belief model $Belief not found"; exit 1 }

$env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
$teacher = "pimc:samples=64,candidates=4,belief=$Belief,weights=1"
$others = "ev:belief=$Belief,floor=none"
$arguments = @('--decisions', "$Decisions", '--out', $out, '--chunk', '20000', '--seed', '3',
    '--batch', '16', '--teacher', $teacher, '--others', $others)
Log "start: label.exe $($arguments -join ' ')"
& $exe @arguments 2>&1 | Out-File -Append -Encoding utf8 (Join-Path $job 'run.out')
$code = $LASTEXITCODE
Log "label.exe exited with code $code"
exit $code
