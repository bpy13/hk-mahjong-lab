# Plan Task 13: label 1 M decisions with the best teacher (long job, headless).
#
# Start detached (rerun the same command to resume; finished chunks are kept):
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
#     CommandLine = 'conhost.exe --headless powershell.exe -NoProfile -File "C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab\experiments\task13_label.ps1"';
#     CurrentDirectory = 'C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab' }
#
# What it does:
# 1. Keeps logical CPUs 0-3 free for the desktop (affinity 0xFFFF0, inherited by the
#    labeller) at Normal priority; the labeller opts itself out of power throttling.
# 2. Waits while Task 11's match job (runs\matches\task11) runs; checks every 5 minutes.
# 3. Picks the teacher: the first non-comment line of runs\labels-job\teacher.txt, or,
#    if that says "auto" (the default), the plan's rule: the higher mean per hand of
#    Task 7's pimc64-belief and Task 11's pimc64-ev (runs\matches\*.json); Task 7's spec
#    on a tie or when Task 11's result is missing. Once chunks exist, the teacher in
#    runs\labels-job\teacher-used.txt is kept, so a resume never mixes teachers.
# 4. Runs the frozen labeller runs\labels-job\bin\label.exe with 8 threads into
#    runs\labels (progress: runs\labels\progress.txt; log: runs\labels-job\run.log).
#
# -DryRun prints whether Task 11's job runs and which teacher would be picked now,
# then exits (no lock, no waiting, no labelling).
param([switch]$DryRun)
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0

$repo = "C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab"
$job = Join-Path $repo "runs\labels-job"
$out = Join-Path $repo "runs\labels"
$log = Join-Path $job "run.log"
$lock = Join-Path $job "run.pid"
$exe = Join-Path $job "bin\label.exe"
$choiceFile = Join-Path $job "teacher.txt"
$usedFile = Join-Path $job "teacher-used.txt"
$task7Spec = "pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1"

function Log([string]$msg) {
    if ($DryRun) { Write-Output "dry run: $msg"; return }
    Add-Content -Path $log -Value "$(Get-Date -Format s) $msg" -Encoding utf8
}

New-Item -ItemType Directory -Force $job | Out-Null
if ((Test-Path $lock) -and -not $DryRun) {
    $old = Get-Content $lock -ErrorAction SilentlyContinue
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        Log "another run (pid $old) is active; exiting"
        exit 0
    }
}
if (-not $DryRun) { Set-Content -Path $lock -Value $PID -Encoding ascii }
Set-Location $repo

# 2. Wait for Task 11's match job.
$announced = $false
while ($true) {
    $task11 = @(Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and ($_.CommandLine -like '*task11\run_matches.py*' -or $_.CommandLine -like '*task11\run.ps1*') })
    if ($task11.Count -eq 0) { break }
    if ($DryRun) { Log "Task 11 match job (pid $($task11[0].ProcessId)) is running; the job would wait"; break }
    if (-not $announced) {
        Log "waiting: Task 11 match job (pid $($task11[0].ProcessId)) is running"
        New-Item -ItemType Directory -Force $out | Out-Null
        Set-Content -Path (Join-Path $out "progress.txt") -Value "waiting for Task 11's match job to finish (since $(Get-Date -Format 'yyyy-MM-dd HH:mm'))" -Encoding utf8
        $announced = $true
    }
    Start-Sleep -Seconds 300
}

# 3. The teacher.
function MatchResult([string]$name) {
    $p = Join-Path $repo "runs\matches\$name.json"
    if (-not (Test-Path $p)) { return $null }
    $r = Get-Content $p -Raw -Encoding utf8 | ConvertFrom-Json
    $spec = if ($r.spec) { $r.spec } else { $r.agents[0].agent }
    return [pscustomobject]@{ Spec = $spec; Mean = [double]$r.agents[0].mean; Ci = [double]$r.agents[0].ci95 }
}

$teacher = $null
$chunksExist = Test-Path (Join-Path $out "chunk-00000.bin")
if ($chunksExist -and (Test-Path $usedFile)) {
    $teacher = (Get-Content $usedFile -Encoding utf8 | Select-Object -First 1).Trim()
    Log "resuming with the teacher of the existing chunks: $teacher"
} else {
    $choice = "auto"
    if (Test-Path $choiceFile) {
        $line = Get-Content $choiceFile -Encoding utf8 | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith("#") } | Select-Object -First 1
        if ($line) { $choice = $line.Trim() }
    }
    if ($choice -ne "auto") {
        $teacher = $choice
        Log "teacher from teacher.txt: $teacher"
    } else {
        $t7 = MatchResult "pimc64-belief"
        $t11 = MatchResult "pimc64-ev"
        if ($null -eq $t11) {
            $teacher = $task7Spec
            Log "auto: Task 11's runs\matches\pimc64-ev.json is missing; Task 7's spec"
        } elseif ($null -eq $t7) {
            $teacher = $t11.Spec
            Log "auto: Task 7's result is missing; Task 11's spec"
        } else {
            $teacher = if ($t11.Mean -gt $t7.Mean) { $t11.Spec } else { $t7.Spec }
            Log ("auto: Task 7 {0:+0.00;-0.00} +- {1:0.00}, Task 11 {2:+0.00;-0.00} +- {3:0.00} per hand; teacher {4}" -f $t7.Mean, $t7.Ci, $t11.Mean, $t11.Ci, $teacher)
        }
    }
    if ($DryRun) { Write-Output "dry run: teacher would be $teacher"; exit 0 }
    Set-Content -Path $usedFile -Value $teacher -Encoding utf8
}
if ($DryRun) { exit 0 }

# 4. Label (8 threads; other jobs share the CPU). App Control sometimes blocks a
# new executable once (os error 4551): one retry after a minute.
$env:RAYON_NUM_THREADS = "8"
# --batch 16: hands played in parallel per step; at most 15 hands per chunk are
# played and dropped at the chunk's end (chunks stay independent of each other).
$arguments = @("--decisions", "1000000", "--out", $out, "--chunk", "20000", "--seed", "3", "--batch", "16", "--teacher", $teacher)
for ($attempt = 1; $attempt -le 2; $attempt++) {
    Log "start (attempt $attempt): label.exe $($arguments -join ' ')"
    & $exe @arguments *>> (Join-Path $job "run.out")
    $code = $LASTEXITCODE
    Log "label.exe exited with code $code"
    if ($code -eq 0 -or $code -eq 2) { break }
    Start-Sleep -Seconds 60
}
Remove-Item $lock -ErrorAction SilentlyContinue
