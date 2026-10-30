# Plan Tasks 15-16 as one queued job (detached, headless):
#   1. wait until Task 13's labelling job has exited and runs\labels holds 1,000,000 rows;
#   2. run the frozen experiments\phase4_build.py --stage all (distil on the GPU, export,
#      evaluate the bare network, the 24-seating netpimc match, Brian's rule).
# The Python code and extension are frozen in runs\phase4\pkg\ at queue time
# (experiments\phase4_job.ps1 -Freeze), so later edits to the tree do not change the job.
# Resume: rerun the same Invoke-CimMethod command; finished stages and chunks are skipped.
# Check:  Get-Content runs\phase4\progress.txt; Get-Content runs\phase4\run.log -Tail 5
param([switch]$Freeze, [switch]$DryRun)

# CPUs 0-3 stay free for the desktop; children inherit the mask. Normal priority.
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0

$repo = "C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab"
$job = Join-Path $repo "runs\phase4"
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
    Set-Content -Path (Join-Path $pkg "FROZEN.txt") -Value "frozen $(Get-Date -Format s) from $head (plus the working tree)" -Encoding utf8
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

function LabelRows {
    try {
        $meta = Get-Content (Join-Path $repo "runs\labels\meta.json") -Raw -Encoding utf8 | ConvertFrom-Json
        return [int]$meta.rows
    } catch { return -1 }
}
function LabellerRunning {
    $p = @(Get-CimInstance Win32_Process -Filter "Name = 'label.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*runs\labels-job\bin\label.exe*' })
    $s = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*task13_label.ps1*' })
    return ($p.Count + $s.Count) -gt 0
}

# 1. Wait for the labels. If the labeller stops early, wait 60 minutes (a restart may
# come), then go on with what exists when that is at least 500,000 rows.
$target = 1000000
$goneSince = $null
$announced = ""
while ($true) {
    $rows = LabelRows
    $running = LabellerRunning
    if (-not $running -and $rows -ge $target) { break }
    if ($running) {
        $goneSince = $null
        $why = "waiting for the labelling job ($rows of $target rows)"
    } else {
        if ($null -eq $goneSince) { $goneSince = Get-Date }
        $mins = [int]((Get-Date) - $goneSince).TotalMinutes
        if ($mins -ge 60 -and $rows -ge 500000) {
            Log "labeller gone for $mins min with $rows rows (< $target); going on with those"
            break
        }
        $why = "labeller not running, $rows of $target rows; waiting ($mins of 60 min)"
    }
    if ($why -ne $announced) { Log $why; $announced = $why }
    Progress $why
    if ($DryRun) { Write-Output "dry run: $why"; exit 0 }
    Start-Sleep -Seconds 300
}
if ($DryRun) { Write-Output "dry run: labels ready ($rows rows)"; exit 0 }

# 2. The GPU must not be held by another training run (> 6 GB used): wait up to 2 h.
for ($i = 0; $i -lt 24; $i++) {
    $used = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1)
    if ($used -le 6000) { break }
    Log "GPU holds $used MiB; waiting"
    Progress "waiting for the GPU ($used MiB used)"
    Start-Sleep -Seconds 300
}

# 3. Run the frozen driver (8 CPU threads for the matches). One retry: App Control
# sometimes blocks a freshly copied extension once.
$env:RAYON_NUM_THREADS = "8"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $pkg
$env:HK_LAB_REPO = $repo
Set-Location $repo
for ($attempt = 1; $attempt -le 2; $attempt++) {
    Log "start (attempt $attempt): phase4_build.py --stage all"
    & $python -u (Join-Path $pkg "phase4_build.py") --stage all *>> (Join-Path $job "run.out")
    $code = $LASTEXITCODE
    Log "phase4_build.py exited with code $code"
    if ($code -eq 0) { break }
    Start-Sleep -Seconds 60
}
Remove-Item $lock -ErrorAction SilentlyContinue
