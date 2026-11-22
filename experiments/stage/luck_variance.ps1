# Plan 2026-11-16 Task 29, part 1 (gpu lane, job name hk-luck-variance): the variance of the
# advantage per decision under each luck-removal setting, on one fixed set of 50,000 self-play
# hands from the start network (python -m mahjong.rl.luck; its docstring has the table's
# meaning). Out: runs\luck\variance.md (the table), variance.json, summary.json, luck.out (log).
#
# Once, before queueing (freezes this checkout's package and extension against later edits;
# build mahjong._core into this checkout first):
#   powershell -NoProfile -File experiments\stage\luck_variance.ps1 -Freeze
# Queue:
#   powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane gpu -Label hk-luck-variance -Script "<full path of this file>"
# Check:  Get-Content runs\luck\variance.md ; Get-Content runs\luck\luck.out -Tail 20
#
# -DryRun: a check of this script on the cpu (4 threads): 400 hands, 64 games, chunks of 100,
# in runs\ppo\smoke\luck-dry, without freezing (runs this checkout's python folder).
param([string]$Init = 'models\stage-imitation.pt', [switch]$Freeze, [switch]$DryRun)

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$runs = Join-Path $repo 'runs\luck'
$pkg = Join-Path $runs 'pkg'

if ($Freeze) {
    if (-not (Test-Path (Join-Path $repo 'python\mahjong\_core.pyd'))) { Write-Output 'build mahjong._core first'; exit 1 }
    New-Item -ItemType Directory -Force -Path $runs | Out-Null
    if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
    New-Item -ItemType Directory -Force -Path $pkg | Out-Null
    Copy-Item -Recurse (Join-Path $repo 'python\mahjong') (Join-Path $pkg 'mahjong')
    Get-ChildItem -Path $pkg -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
    Write-Output "frozen into $pkg"
    exit 0
}

if ($DryRun) {
    $out = Join-Path $repo 'runs\ppo\smoke\luck-dry'
    $env:PYTHONPATH = Join-Path $repo 'python'
    $env:RAYON_NUM_THREADS = '4'
    $a = @('--hands', '400', '--envs', '64', '--chunk', '100', '--time-hands', '100',
           '--epochs', '2', '--device', 'cpu', '--torch-threads', '4')
} else {
    $out = $runs
    if (-not (Test-Path (Join-Path $pkg 'mahjong\_core.pyd'))) { Write-Output 'run with -Freeze first'; exit 1 }
    $env:PYTHONPATH = $pkg
    $env:RAYON_NUM_THREADS = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
    $a = @('--hands', '50000', '--envs', '256', '--chunk', '2000', '--time-hands', '4000',
           '--epochs', '5', '--device', 'cuda')
}
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $repo
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out 'luck.out'

# The package must come from this checkout (the conda env is shared and holds an editable
# install of another checkout); the first line of the log names it.
$file = & $python -c 'import mahjong; print(mahjong.__file__)'
Write-Output "mahjong.__file__ = $file"
Add-Content -Encoding utf8 -Path $log -Value "mahjong.__file__ = $file"
if (-not $file -or -not ([IO.Path]::GetFullPath($file)).StartsWith($repo, [StringComparison]::OrdinalIgnoreCase)) {
    Write-Output "mahjong is not imported from $repo"
    Add-Content -Encoding utf8 -Path $log -Value "mahjong is not imported from $repo"
    exit 1
}
if (-not (Test-Path $Init)) { Write-Output "missing $Init (Task 4b's imitation network)"; exit 1 }

& $python -u -m mahjong.rl.luck --init $Init --out $out @a 2>&1 |
    ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
exit $LASTEXITCODE
