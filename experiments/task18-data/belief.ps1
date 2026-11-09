# Plan 2026-11-09 Task 18: retrain the belief model on the passed-win-rule logs (runs\logs-v2),
# with the passed-tile block (--passed). Writes models\belief-v2.pt / .mjnn (the current
# models\belief.* stay untouched), then the hold-out evaluation. GPU lane; about 6 GPU minutes.
# Queue:  powershell -NoProfile -File <hkq.ps1> add -Lane gpu -Script <this file> -Label task18-belief
# Check:  Get-Content C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab\runs\belief-v2\train.log -Tail 3
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = Join-Path $repo 'python'
Set-Location $repo
$threads = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
$ckpt = Join-Path $repo 'runs\belief-v2'
New-Item -ItemType Directory -Force $ckpt | Out-Null
$resume = if (Test-Path (Join-Path $ckpt 'ckpt-0.pt')) { @('--resume') } else { @() }

& $python -m mahjong.belief.train --logs runs/logs-v2/train --holdout runs/logs-v2/holdout `
    --passed --out models/belief-v2.pt --ckpt-dir runs/belief-v2 --epochs 5 --threads $threads @resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m mahjong.belief.evaluate --logs runs/logs-v2/holdout --model models/belief-v2.pt `
    --out runs/belief-v2/eval.json
exit $LASTEXITCODE
