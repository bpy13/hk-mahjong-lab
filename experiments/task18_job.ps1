# Plan Task 18 as one queued job (detached, headless), in plan order:
#   main 48 h, seed 2 24 h, ablation without the paired reward 6 h, ablation without the
#   target head 6 h, then the AC8 matches (phase4_build.py --stage ppo-eval).
# Normal priority: Brian has not allowed above-normal (docs/open-items.md).
# The Python code and extension are frozen in runs\task18\pkg\ at queue time
# (experiments\task18_job.ps1 -Freeze), so later edits to the tree do not change the job.
# Resume: rerun the same Invoke-CimMethod command. A finished run (final.pt) is skipped,
# an unfinished one continues from its newest checkpoint (at most 30 minutes lost), and
# finished match chunks are skipped.
# Check:  Get-Content runs\task18\progress.txt; Get-Content runs\ppo\main\progress.txt;
#         Get-Content runs\task18\run.log -Tail 5
param([switch]$Freeze)

# CPUs 0-3 stay free for the desktop; children inherit the mask.
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0

$repo = "C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab"
$job = Join-Path $repo "runs\task18"
$pkg = Join-Path $job "pkg"
$log = Join-Path $job "run.log"
$lock = Join-Path $job "job.pid"
$python = "C:\Users\brian\miniconda3\envs\mahjong\python.exe"
New-Item -ItemType Directory -Force -Path $job | Out-Null

function Log([string]$msg) {
    Add-Content -Path $log -Value "$(Get-Date -Format s) job: $msg" -Encoding utf8
}
function Progress([string]$msg) {
    Set-Content -Path (Join-Path $job "progress.txt") -Value "$msg ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))" -Encoding utf8
}

if ($Freeze) {
    # Copy the package (with its built _core.pyd) and the driver script.
    if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
    New-Item -ItemType Directory -Force -Path $pkg | Out-Null
    Copy-Item -Recurse (Join-Path $repo "python\mahjong") (Join-Path $pkg "mahjong")
    Get-ChildItem -Path $pkg -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    Copy-Item (Join-Path $repo "experiments\phase4_build.py") (Join-Path $pkg "phase4_build.py")
    $head = (& git -C $repo rev-parse --short HEAD)
    Set-Content -Path (Join-Path $pkg "FROZEN.txt") -Value "frozen $(Get-Date -Format s) from $head" -Encoding utf8
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
if (-not (Test-Path (Join-Path $pkg "phase4_build.py"))) {
    Log "no frozen package in $pkg; run with -Freeze first"
    exit 1
}

$env:RAYON_NUM_THREADS = "8"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $pkg
$env:HK_LAB_REPO = $repo
Set-Location $repo

# The plan's rule: no GPU work while another training run holds more than 4 GB.
function WaitForGpu([string]$what) {
    while ($true) {
        $used = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1)
        if ($used -le 4000) { return }
        Log "GPU holds $used MiB; $what waits"
        Progress "$what waits for the GPU ($used MiB used)"
        Start-Sleep -Seconds 300
    }
}

# name, hours, extra arguments, in plan order.
$runs = @(
    @{ Name = "main"; Hours = 48; Extra = @() },
    @{ Name = "seed2"; Hours = 24; Extra = @("--seed", "2") },
    @{ Name = "ablate-5a"; Hours = 6; Extra = @("--no-paired-reward") },
    @{ Name = "ablate-1a"; Hours = 6; Extra = @("--no-target-head") }
)
foreach ($r in $runs) {
    $name = $r.Name
    $out = "runs\ppo\$name"
    if (Test-Path (Join-Path $repo "$out\final.pt")) { Log "$name done already"; continue }
    WaitForGpu $name
    # Three tries: App Control sometimes blocks a freshly copied extension once.
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Log "start $name (attempt $attempt): $($r.Hours) h"
        Progress "$name running, see $out\progress.txt"
        $args18 = @("-u", "-m", "mahjong.rl.ppo_cnn", "--init", "models/cnn-distill.pt", "--hours", "$($r.Hours)", "--out", $out, "--resume") + $r.Extra
        & $python @args18 *>> (Join-Path $job "$name.out")
        $code = $LASTEXITCODE
        Log "$name exited with code $code"
        if ($code -eq 0) { break }
        Start-Sleep -Seconds 60
    }
    if (-not (Test-Path (Join-Path $repo "$out\final.pt"))) {
        Log "$name has no final.pt after 3 tries; stopping the job"
        Progress "stopped: $name failed, see runs\task18\$name.out"
        Remove-Item $lock -ErrorAction SilentlyContinue
        exit 1
    }
}

# The matches use the CPU only; more threads, as the other long jobs should be over.
$env:RAYON_NUM_THREADS = "14"
for ($attempt = 1; $attempt -le 2; $attempt++) {
    Log "start ppo-eval (attempt $attempt)"
    Progress "ppo-eval running, see runs\phase4\progress.txt"
    & $python -u (Join-Path $pkg "phase4_build.py") --stage ppo-eval *>> (Join-Path $job "ppo-eval.out")
    $code = $LASTEXITCODE
    Log "ppo-eval exited with code $code"
    if ($code -eq 0) { break }
    Start-Sleep -Seconds 60
}
Progress "job finished (last exit code $code)"
Remove-Item $lock -ErrorAction SilentlyContinue
