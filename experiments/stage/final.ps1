# Plan 2026-11-16 Task 7 (cpu lane): the final measurement of the best self-play network.
# Needs runs\rl-stage\best.mjnn (int8 export of runs\rl-stage\best.pt, the best 2-hourly
# result of job 167) and runs\rl-stage\teacher.mjnn (the imitation network, the run's teacher).
# Out (runs\rl-stage\final\):
#   vs-stage-40000.txt  best network + three `stage`, 40,000 deals
#   table-4000.txt      best network, teacher, `stage`, `value` at one table, 4,000 deals
#   dealin.txt          where `stage` deal-ins come from (examples\dealin, 20,000 deals)
# Queue: jobq.ps1 add -Lane cpu -Label rl-stage-final -Script <this file>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$py = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONPATH = Join-Path $root 'python'
$where = & $py -c "import mahjong, mahjong._core; print(mahjong.__file__)"
Write-Output "mahjong.__file__ = $where"
if (-not $where -or -not $where.StartsWith($root)) { Write-Output "mahjong is not imported from $root"; exit 1 }

$out = Join-Path $root 'runs\rl-stage\final'
New-Item -ItemType Directory -Force $out | Out-Null
$bin = Join-Path $root 'runs\rl-stage\bin'
$best = 'net:path=' + (Join-Path $root 'runs\rl-stage\best.mjnn')
$teacher = 'net:path=' + (Join-Path $root 'runs\rl-stage\teacher.mjnn')
foreach ($f in 'best.mjnn', 'teacher.mjnn') {
    if (-not (Test-Path (Join-Path $root "runs\rl-stage\$f"))) { Write-Output "missing $f"; exit 1 }
}
$env:RAYON_NUM_THREADS = $(if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '16' })

& (Join-Path $bin 'compare.exe') $best stage stage stage 40000 1 | Out-File -Encoding utf8 (Join-Path $out 'vs-stage-40000.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $bin 'compare.exe') $best $teacher stage value 4000 1 | Out-File -Encoding utf8 (Join-Path $out 'table-4000.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $bin 'dealin.exe') 20000 1 1 | Out-File -Encoding utf8 (Join-Path $out 'dealin.txt')
exit $LASTEXITCODE
