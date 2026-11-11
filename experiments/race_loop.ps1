# Plan 2026-11-09 Task 9 (AC6): the race bot's self-play loop as one queued job
# (docs/race-bot.md §8; driver experiments\race_loop.py). Self-play and matches use the CPU,
# training the GPU, in one job, as experiments\phase4_job.ps1 mixed them: queue it in the
# gpu lane (8 threads; the cpu lane drops to one job meanwhile).
#
# Once, before queueing (freezes the code so later edits to the tree do not change the job):
#   powershell -NoProfile -File <this file> -Freeze
# Queue:
#   powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane gpu -Script <this file>
# Resume (after a restart or a failure): add the same job again. Logs resume by chunk,
# training from its newest epoch checkpoint, matches by chunk; finished steps are skipped.
# A finished loop (runs\race\done.json) exits at once.
# Check:  Get-Content runs\race\progress.txt; Get-Content runs\race\run.log -Tail 5;
#         runs\race\report.md
# Smoke test (minutes, CPU, separate folder): <this file> -Root runs\race-smoke\loop -Freeze; <this file> -Root runs\race-smoke\loop -Smoke
# -Repo: the checkout the job runs in (Task 20 reruns it from a worker clone, so the code that
# is frozen is the worker's, not the standalone checkout's).
param([string]$Root = 'runs\race', [switch]$Freeze, [switch]$Smoke,
    [string]$Repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab')

# CPUs 0-3 stay free for the desktop; children inherit the mask. Normal priority.
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0

$repo = $Repo
$job = Join-Path $repo $Root
$pkg = Join-Path $job 'pkg'
$log = Join-Path $job 'run.log'
$lock = Join-Path $job 'job.pid'
$python = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
New-Item -ItemType Directory -Force -Path $job | Out-Null

function Log([string]$msg) {
    Add-Content -Path $log -Value "$(Get-Date -Format s) job: $msg" -Encoding utf8
}

if ($Freeze) {
    # The package (with its built _core.pyd), the logger and the driver.
    $exe = Join-Path $repo 'target\release\examples\race_log.exe'
    if (-not (Test-Path $exe)) { Write-Output "build it first: cargo build --release -p mahjong-search --example race_log"; exit 1 }
    if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
    New-Item -ItemType Directory -Force -Path (Join-Path $pkg 'bin') | Out-Null
    Copy-Item -Recurse (Join-Path $repo 'python\mahjong') (Join-Path $pkg 'mahjong')
    Get-ChildItem -Path $pkg -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
    Copy-Item $exe (Join-Path $pkg 'bin\race_log.exe')
    Copy-Item (Join-Path $repo 'experiments\race_loop.py') (Join-Path $pkg 'race_loop.py')
    $head = (& git -C $repo rev-parse --short HEAD)
    Set-Content -Path (Join-Path $pkg 'FROZEN.txt') -Value "frozen $(Get-Date -Format s) from $head (plus the working tree)" -Encoding utf8
    Write-Output "frozen into $pkg"
    exit 0
}

if (Test-Path $lock) {
    $old = Get-Content $lock -ErrorAction SilentlyContinue
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        Log "another run (pid $old) is active; exiting"
        exit 0
    }
}
Set-Content -Path $lock -Value $PID -Encoding ascii
if (-not (Test-Path (Join-Path $pkg 'race_loop.py'))) {
    Log "no frozen package in $pkg; run with -Freeze first"
    exit 1
}

# The queue sets HKQ_THREADS (8 in the gpu and cpu lanes); 8 when run by hand.
$threads = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
$env:RAYON_NUM_THREADS = $threads
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = $pkg
$env:HK_LAB_REPO = $repo
Set-Location $repo
$driverArgs = @('-u', (Join-Path $pkg 'race_loop.py'), '--root', $Root, '--threads', $threads,
    '--race-log', (Join-Path $pkg 'bin\race_log.exe'))
if ($Smoke) { $driverArgs += '--smoke' }
Log "start: race_loop.py $($driverArgs -join ' ')"
& $python @driverArgs *>> (Join-Path $job 'driver.out')
$code = $LASTEXITCODE
Log "race_loop.py exited with code $code"
Remove-Item $lock -ErrorAction SilentlyContinue
exit $code
