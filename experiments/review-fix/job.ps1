# Plan 2026-11-09 Task 4 (AC3) through the job queue (cpu lane). One job per variant, then
# a compare job after both:
#   $hkq = 'C:\Users\brian\OneDrive\Documents\workspace\repo\bot-stream\tools\queue\hkq.ps1'
#   powershell -NoProfile -File $hkq add -Lane cpu -Script <this file> -ScriptArgs "-Variant old"
#   powershell -NoProfile -File $hkq add -Lane cpu -Script <this file> -ScriptArgs "-Variant new"
#   powershell -NoProfile -File $hkq add -Lane cpu -Script <this file> -ScriptArgs "-Compare" -After '<old id>,<new id>'
# Resume: add the same job again; finished chunks are skipped.
# Check:  Get-Content runs\review-fix\old.progress, new.progress; runs\review-fix\compare.json
# Needs runs\review-fix\pkg-old and pkg-new (the two extensions, unpacked wheels).
param([ValidateSet('old', 'new')][string]$Variant, [switch]$Compare)

$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$driver = Join-Path $repo 'experiments\review-fix\run_matches.py'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $repo
if ($Compare) {
    & $python -u $driver --compare
    exit $LASTEXITCODE
}
$env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
$env:PYTHONPATH = Join-Path $repo "runs\review-fix\pkg-$Variant"
& $python -u $driver --variant $Variant
exit $LASTEXITCODE
