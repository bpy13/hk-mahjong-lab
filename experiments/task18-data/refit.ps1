# Plan 2026-11-09 Task 18: refit the danger tables with the passed-win rule and the
# faan_shanten fix, and check calibration before and after; refit the passed-tile factor
# against the retrained belief model. CPU lane; builds inside the job.
# Steps:
#   1. build danger_check, compare, fit_passed with today's tables; keep copies as *-old.exe
#   2. danger_check (heuristic tables, 2,000 hands) and `defensive` vs three `value`
#      (1,000 walls x 4 seatings) with the old tables
#   3. fit_danger 8000 (rewrites crates\search\src\danger_tables.rs in the working tree)
#   4. rebuild; the same checks with the new tables; danger_check --belief for the old and
#      the retrained belief model; fit_passed --belief models\belief-v2.mjnn (8,000 hands)
# Output: runs\task18-refit\*.txt. The tables file is left changed for review (git diff).
# Queue:  powershell -NoProfile -File <jobq.ps1> add -Lane cpu -Script <this file> -Label task18-refit
# Check:  Get-ChildItem C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab\runs\task18-refit
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$out = Join-Path $repo 'runs\task18-refit'
$bin = Join-Path $out 'bin'
New-Item -ItemType Directory -Force $bin | Out-Null
Set-Location $repo
$env:PATH = "C:\Users\brian\.cargo\bin;" + $env:PATH
$env:CARGO_BUILD_JOBS = '8'
$env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
$examples = Join-Path $repo 'target\release\examples'

function Stamp($msg) { Write-Host ("{0:s} {1}" -f (Get-Date), $msg) }
function Build([string]$suffix) {
    & cargo build --release -p mahjong-search --example danger_check --example compare --example fit_passed --example fit_danger
    if ($LASTEXITCODE -ne 0) { Stamp 'build failed'; exit $LASTEXITCODE }
    foreach ($e in 'danger_check', 'compare', 'fit_passed', 'fit_danger') {
        Copy-Item (Join-Path $examples "$e.exe") (Join-Path $bin "$e-$suffix.exe") -Force
    }
}
function Run([string]$name, [string]$exe, [string[]]$arguments) {
    $file = Join-Path $out "$name.txt"
    if (Test-Path $file) { Stamp "${name}: done already"; return }
    Stamp "${name}: start"
    & (Join-Path $bin $exe) @arguments 2>&1 | Out-File -Encoding utf8 "$file.part"
    if ($LASTEXITCODE -ne 0) { Stamp "${name}: FAILED ($LASTEXITCODE)"; exit $LASTEXITCODE }
    Move-Item "$file.part" $file -Force
    Stamp "${name}: done"
}

# 1-2. Old tables (skipped when the fit already ran: its output marks the tables as new).
if (-not (Test-Path (Join-Path $out 'fit-danger.txt'))) {
    Build 'old'
    Run 'danger-check-old' 'danger_check-old.exe' @('2000', '0')
    Run 'defensive-old' 'compare-old.exe' @('defensive', 'value', 'value', 'value', '1000')
    # 3. Refit (writes the tables source).
    Run 'fit-danger' 'fit_danger-old.exe' @('8000')
}
# 4. New tables.
Build 'new'
Run 'danger-check-new' 'danger_check-new.exe' @('2000', '0')
Run 'defensive-new' 'compare-new.exe' @('defensive', 'value', 'value', 'value', '1000')
Run 'danger-check-belief-old' 'danger_check-new.exe' @('--belief', 'models/belief.mjnn', '2000')
Run 'danger-check-belief-v2' 'danger_check-new.exe' @('--belief', 'models/belief-v2.mjnn', '2000')
Run 'fit-passed-belief-v2' 'fit_passed-new.exe' @('--hands', '8000', '--seed', '1', '--belief', 'models/belief-v2.mjnn')
exit 0
