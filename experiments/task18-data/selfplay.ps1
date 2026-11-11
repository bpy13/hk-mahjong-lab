# Plan 2026-11-09 Task 18: self-play logs with the passed-win rule, for retraining the
# belief model (and the danger tables). Same fields, hand counts and seeds as the first
# logs (runs\logs\run.ps1), in the new row layout 2 (adds `passed`, features::encode_passed;
# the faan_shanten fix also changes some features).
#
# 1. Freeze the logger once (a build, outside the queue), so later edits to the tree do
#    not change a queued or resumed job:
#      powershell -NoProfile -File <this file> -Freeze
# 2. Queue it on the cpu lane:
#      powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane cpu -Script <this file> -Label task18-selfplay
#
# Output: runs\logs-v2\train (25,000 hands, seed 1) then runs\logs-v2\holdout (5,000 hands,
# field holdout, seed 2). Resumable: add the job again; each log continues after its last
# complete chunk. The first logs took 69 minutes for train on 14 threads.
# Check:  Get-Content C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab\runs\logs-v2\run.log -Tail 5
param([switch]$Freeze)
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$dir = Join-Path $repo 'runs\logs-v2'
$exe = Join-Path $dir 'bin\selfplay_log.exe'
$log = Join-Path $dir 'run.log'
New-Item -ItemType Directory -Force (Join-Path $dir 'bin') | Out-Null
Set-Location $repo

function Log([string]$msg) { Add-Content -Path $log -Value "$(Get-Date -Format s) $msg" -Encoding utf8 }

if ($Freeze) {
    $env:PATH = "C:\Users\brian\.cargo\bin;" + $env:PATH
    $env:CARGO_BUILD_JOBS = '8'
    & cargo build --release -p mahjong-search --example selfplay_log
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Copy-Item (Join-Path $repo 'target\release\examples\selfplay_log.exe') $exe -Force
    $head = (& git -C $repo rev-parse --short HEAD)
    $dirty = if (& git -C $repo status --porcelain -- crates) { ' (crates had uncommitted changes)' } else { '' }
    Log "frozen selfplay_log.exe from $head$dirty"
    exit 0
}
if (-not (Test-Path $exe)) { Write-Output "no frozen logger in $exe; run with -Freeze first"; exit 1 }

$threads = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
$runs = @(
    @{ Name = 'train'; Args = @('--hands', '25000', '--out', (Join-Path $dir 'train'), '--seed', '1') },
    @{ Name = 'holdout'; Args = @('--hands', '5000', '--field', 'holdout', '--out', (Join-Path $dir 'holdout'), '--seed', '2') }
)
foreach ($r in $runs) {
    Log "$($r.Name): start ($threads threads)"
    & $exe @($r.Args) --threads $threads 2>&1 | Out-File -Append -Encoding utf8 $log
    $code = $LASTEXITCODE
    Log "$($r.Name): exit $code"
    if ($code -ne 0) { exit $code }
}
exit 0
