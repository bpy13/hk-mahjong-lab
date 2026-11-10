# Plan 2026-11-09 Task 18 "done when": are the retrained models not weaker than the current
# ones in a duplicate match? Head to head on the same walls: the retrained model and the
# current one sit opposite each other with two `value` bots; the four rotations give each one
# every seat. The report's `pairs` row for seats 0 and 2 is the wall-paired bootstrap verdict.
#   - distilled network: models\cnn-distill-v2.mjnn (passed block) vs models\cnn-distill.mjnn
#   - teacher (belief-PIMC-64, runs\labels-job\teacher-used.txt) with models\belief-v2.mjnn vs
#     with models\belief.mjnn
# Solo rows (each network against three `value`) give absolute points per hand.
# 1,000 walls x 4 seatings, seed 2026, passed-win rule on. Rebuilds the extension first.
# Queue:  powershell -NoProfile -File <hkq.ps1> add -Lane cpu -Script <this file> -Label task18-match
# Output: runs\matches\task18-*.json. A rerun skips finished files.
# Check:  Get-ChildItem C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab\runs\matches\task18-*.json
param([int]$Deals = 1000, [int]$TeacherDeals = 1000, [switch]$SkipBuild)
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
Set-Location $repo
$out = Join-Path $repo 'runs\matches'
New-Item -ItemType Directory -Force $out | Out-Null

function Stamp($msg) { Write-Host ("{0:s} {1}" -f (Get-Date), $msg) }

if (-not $SkipBuild) {
    $env:PATH = "C:\Users\brian\.cargo\bin;" + $env:PATH
    $env:PYO3_PYTHON = $python
    $env:CARGO_BUILD_JOBS = '8'
    Stamp 'rebuilding the extension'
    & $python -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) { Stamp 'extension build failed'; exit $LASTEXITCODE }
}

function Match($name, [string[]]$agents, [int]$deals) {
    $file = Join-Path $out "task18-$name.json"
    if (Test-Path $file) { Stamp "${name}: done already"; return }
    Stamp "${name}: start"
    & $python -c "from mahjong.cli.main import app; app()" hk tournament @agents --deals $deals --seed 2026 --out $file
    if ($LASTEXITCODE -ne 0) { Stamp "${name}: FAILED ($LASTEXITCODE)"; exit $LASTEXITCODE }
    Stamp "${name}: done"
}

$cnnNew = 'net:path=models/cnn-distill-v2.mjnn'
$cnnOld = 'net:path=models/cnn-distill.mjnn'
$teacherOld = ((Get-Content (Join-Path $repo 'runs\labels-job\teacher-used.txt') -Encoding UTF8 -TotalCount 1) -replace "^\uFEFF", '').Trim()
$teacherNew = $teacherOld -replace 'models/belief\.mjnn', 'models/belief-v2.mjnn'
if ($teacherNew -eq $teacherOld) { Stamp "teacher spec has no models/belief.mjnn: $teacherOld"; exit 1 }

Match 'cnn-solo-new' @($cnnNew, 'value', 'value', 'value') $Deals
Match 'cnn-solo-old' @($cnnOld, 'value', 'value', 'value') $Deals
Match 'cnn-h2h' @($cnnNew, 'value', $cnnOld, 'value') $Deals
Match 'teacher-h2h' @($teacherNew, 'value', $teacherOld, 'value') $TeacherDeals
exit 0
