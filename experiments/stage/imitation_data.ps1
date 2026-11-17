# Plan 2026-11-16 Task 4, step 1 (cpu lane): log 3 million decisions of the three-stage
# bot `stage` (all four seats `stage`, one labelled seat per hand) with the full discard
# history (label layout 3) into runs\rl-stage\labels: 30 chunks of 100,000 rows. Chunks
# hold whole hands, so holding out chunks 28 and 29 splits the data by hand.
# About 10 minutes on 20 threads (10,000 rows in 2 s, measured 2026-11-16).
#
# Usage (from this checkout, once runs\rl-stage\bin\label.exe is a frozen copy of
# target\release\examples\label.exe):
#   jobq.ps1 add -Lane cpu -Script <checkout>\experiments\stage\imitation_data.ps1 -Label stage-imitation-data
# Rerunning resumes after the last complete chunk.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

# The shared conda env may import another checkout's package: insist on this one.
$py = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONPATH = Join-Path $root 'python'
$where = & $py -c "import mahjong; print(mahjong.__file__)"
Write-Output "mahjong.__file__ = $where"
if (-not $where -or -not ([IO.Path]::GetFullPath($where)).StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { Write-Output "mahjong is not imported from $root"; exit 1 }

$exe = Join-Path $root 'runs\rl-stage\bin\label.exe'
if (-not (Test-Path $exe)) { Write-Output "missing $exe"; exit 1 }
$env:RAYON_NUM_THREADS = $(if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '16' })
$out = Join-Path $root 'runs\rl-stage\labels'
$labelArgs = @('--decisions', '3000000', '--chunk', '100000', '--seed', '16', '--batch', '64',
    '--teacher', 'stage', '--others', 'stage', '--out', $out)
# App Control sometimes blocks a fresh executable once (os error 4551): one retry.
for ($attempt = 1; $attempt -le 2; $attempt++) {
    & $exe @labelArgs
    $code = $LASTEXITCODE
    Write-Output "label.exe exited with code $code (attempt $attempt)"
    if ($code -eq 0 -or $code -eq 2) { break }
    Start-Sleep -Seconds 60
}
exit $code
