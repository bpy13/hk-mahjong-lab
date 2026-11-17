# Plan 2026-11-16 Task 4, step 2 (gpu lane, after imitation_data.ps1): train the
# suit-shared CNN with the passed and full-history inputs to copy `stage`, then play
# it against three `stage` bots over 4,000 duplicate deals.
#
# Data: runs\rl-stage\labels (chunks 0-27 train, 28-29 held out, split by hand).
# Out:  runs\rl-stage\imitation\ (train.log, ckpt.pt, result.json with the held-out
#       agreement and its 95 % range), models\stage-imitation.pt / -fp32.mjnn / .mjnn,
#       runs\rl-stage\imitation\vs-stage.txt (the match).
# About 2 hours on the graphics card (watch-session estimate); rerunning resumes the
# training from ckpt.pt.
#
# Usage:
#   jobq.ps1 add -Lane gpu -After <data job> -Script <checkout>\experiments\stage\imitation.ps1 -Label stage-imitation
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

# The shared conda env may import another checkout's package: insist on this one.
$py = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONPATH = Join-Path $root 'python'
$where = & $py -c "import mahjong, mahjong._core; print(mahjong.__file__)"
Write-Output "mahjong.__file__ = $where"
if (-not $where -or -not ([IO.Path]::GetFullPath($where)).StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { Write-Output "mahjong is not imported from $root"; exit 1 }

$labels = Join-Path $root 'runs\rl-stage\labels'
$meta = Join-Path $labels 'meta.json'
if (-not (Test-Path $meta)) { Write-Output "missing $meta (run imitation_data.ps1 first)"; exit 1 }
$chunks = @((Get-Content $meta -Raw -Encoding utf8 | ConvertFrom-Json).chunks).Count
if ($chunks -ne 30) { Write-Output "expected 30 label chunks, found $chunks"; exit 1 }

$out = Join-Path $root 'runs\rl-stage\imitation'
New-Item -ItemType Directory -Force $out | Out-Null
$trainArgs = @('-m', 'mahjong.rl.distill_cnn', '--labels', $labels, '--out', $out,
    '--models', (Join-Path $root 'models'), '--name', 'stage-imitation',
    '--holdout-chunks', '28', '29', '--epochs', '6', '--minutes', '150',
    '--batch', '2048', '--device', 'cuda', '--passed', '--history')
if (Test-Path (Join-Path $out 'ckpt.pt')) { $trainArgs += '--resume' }
& $py @trainArgs
if ($LASTEXITCODE -ne 0) { Write-Output "distill_cnn exited with code $LASTEXITCODE"; exit $LASTEXITCODE }

# The match (CPU): the network in seat rotation against three `stage` bots.
$compare = Join-Path $root 'runs\rl-stage\bin\compare.exe'
$env:RAYON_NUM_THREADS = $(if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '16' })
$net = 'net:path=' + (Join-Path $root 'models\stage-imitation.mjnn')
& $compare $net stage stage stage 4000 1 | Out-File -Encoding utf8 (Join-Path $out 'vs-stage.txt')
$code = $LASTEXITCODE
Get-Content (Join-Path $out 'vs-stage.txt')
exit $code
