# Plan 2026-11-09 Task 7 (AC5): race bot v0 (trained belief) against three `value` bots,
# 1,000 walls x 4 seatings, seed 2026, through the job queue's cpu lane:
#   powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\bot-stream\tools\queue\hkq.ps1 add -Lane cpu -Script <this file>
# Output: runs\matches\race-v0-belief.json (rerun overwrites; about 16 s per 1,000 walls for ev,
# so minutes). Check: Test-Path runs\matches\race-v0-belief.json
param([int]$Deals = 1000, [string]$Out = 'runs\matches\race-v0-belief.json')
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
Set-Location $repo
& $python -c "from mahjong.cli.main import app; app()" hk tournament "race:belief=models/belief.mjnn" value value value --deals $Deals --seed 2026 --out $Out
exit $LASTEXITCODE
