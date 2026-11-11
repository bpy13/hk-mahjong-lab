# Plan 2026-11-09 Task 16 (Phase 6): what the passed-win rule and the race bot's passed-tile
# safety (Task 14) and risk read (Task 15) are worth. Queue it AFTER the review-fix jobs:
#   powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane cpu -Script <this file> -Label passed-tiles -After 32
# Steps: (1) rebuild the extension so the CLI has Tasks 12-15 (pip install -e, a few minutes);
# (2) race-bot matches, 1,000 walls x 4 seatings, seed 2026; (3) the teacher sanity rows;
# (4) summary.py writes runs\matches\passed-summary.json.
# Output: runs\matches\passed-*.json. A rerun skips finished files; delete one to redo it.
# Check:  Get-ChildItem runs\matches\passed-*.json; Get-Content runs\matches\passed-summary.json
param([int]$Deals = 1000, [int]$SanityDeals = 300, [switch]$SkipBuild)
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
Set-Location $repo
$out = Join-Path $repo 'runs\matches'
New-Item -ItemType Directory -Force $out | Out-Null

function Stamp($msg) { Write-Host ("{0:s} {1}" -f (Get-Date), $msg) }

# 1. Rebuild the extension inside the job (CLAUDE.md, "Windows (this PC)").
if (-not $SkipBuild) {
    $env:PATH = "C:\Users\brian\.cargo\bin;" + $env:PATH
    $env:PYO3_PYTHON = $python
    $env:CARGO_BUILD_JOBS = '8'
    Stamp 'rebuilding the extension'
    & $python -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) { Stamp 'extension build failed'; exit $LASTEXITCODE }
}

# 2. One tournament; skipped when its output exists.
function Match($name, [string[]]$agents, [int]$deals, [string[]]$extra = @()) {
    $file = Join-Path $out "passed-$name.json"
    if (Test-Path $file) { Stamp "${name}: done already"; return }
    Stamp "${name}: start"
    & $python -c "from mahjong.cli.main import app; app()" hk tournament @agents --deals $deals --seed 2026 --out $file @extra
    if ($LASTEXITCODE -ne 0) { Stamp "${name}: FAILED ($LASTEXITCODE)"; exit $LASTEXITCODE }
    Stamp "${name}: done"
}

$b = 'race:belief=models/belief.mjnn'
$full = $b
$noPassed = "$b,passed=off"
$noRisk = "$b,risk=off"
$off = "$b,passed=off,risk=off"

# Solo: each variant against three `value` bots (absolute points per hand, same walls).
Match 'solo-full' @($full, 'value', 'value', 'value') $Deals
Match 'solo-off' @($off, 'value', 'value', 'value') $Deals
Match 'solo-nopassed' @($noPassed, 'value', 'value', 'value') $Deals
Match 'solo-norisk' @($noRisk, 'value', 'value', 'value') $Deals
# Head to head: two variants and two `value` bots on the same walls. The report's "pairs"
# row is the wall-paired difference with its bootstrap interval. The variants sit opposite
# each other, and the four rotations give each one every seat.
Match 'h2h-full-vs-off' @($full, 'value', $off, 'value') $Deals
Match 'h2h-full-vs-nopassed' @($full, 'value', $noPassed, 'value') $Deals
Match 'h2h-full-vs-norisk' @($full, 'value', $noRisk, 'value') $Deals

# 3. Sanity row: the teacher (runs\labels-job\teacher-used.txt) against three `value`, with
# the passed-win rule on and off (the same walls; the rule changes the game, so not paired).
$teacher = ((Get-Content (Join-Path $repo 'runs\labels-job\teacher-used.txt') -Encoding UTF8 -TotalCount 1) -replace "^\uFEFF", '').Trim()
Match 'teacher-rule-on' @($teacher, 'value', 'value', 'value') $SanityDeals
Match 'teacher-rule-off' @($teacher, 'value', 'value', 'value') $SanityDeals @('--no-passed-win-lock')

# 4. Summary.
& $python -I (Join-Path $repo 'experiments\passed-tiles\summary.py')
exit $LASTEXITCODE
