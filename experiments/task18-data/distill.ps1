# Plan 2026-11-09 Task 18: retrain the teacher's distilled network (the suit-shared CNN) on the
# passed-win-rule labels (runs\labels-v2, queue job 40), with the passed-tile block (--passed).
# Same settings as the first distillation (experiments\phase4_build.py stage distill: 12 epochs,
# at most 110 minutes, last chunk held out). Writes models\cnn-distill-v2.pt / -fp32.mjnn /
# .mjnn (the current models\cnn-distill.* stay untouched) and runs\distill-v2\result.json.
# GPU lane; about 25 GPU minutes. Resumable: a checkpoint every 15 minutes, --resume picks it up.
# Queue:  powershell -NoProfile -File <hkq.ps1> add -Lane gpu -Script <this file> -Label task18-distill
# Check:  Get-Content C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab\runs\distill-v2\train.log -Tail 3
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab'
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = Join-Path $repo 'python'
Set-Location $repo
$threads = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
$out = Join-Path $repo 'runs\distill-v2'
New-Item -ItemType Directory -Force $out | Out-Null
$resume = if (Test-Path (Join-Path $out 'ckpt.pt')) { @('--resume') } else { @() }

& $python -m mahjong.rl.distill_cnn --labels runs/labels-v2 --out runs/distill-v2 --models models `
    --name cnn-distill-v2 --epochs 12 --minutes 110 --passed --threads $threads @resume
exit $LASTEXITCODE
