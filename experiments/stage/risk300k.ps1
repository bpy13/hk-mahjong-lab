# Plan 2026-11-16 Task 16: the risk network on 300,000 deals. Steps, each queued as its own job:
#   -Step data     300,000 deals of four `stage` bots with `risk_log` (cpu lane, about 3 h on
#                  8 threads; row layout 2; resumes after the last complete chunk; stops with a
#                  note if C: has under 200 GB free)
#   -Step fit      one fit of Task 15's network on a range of chunks (gpu lane): -Train 0-29,
#                  0-99 or 0-289 into -FitDir; same recipe each time (lr 5e-4 cosine over
#                  -Epochs, batch 1,024, best epoch by the held-out `win` loss, stop after
#                  -Patience epochs without a gain); resumes from ckpt.pt
#   -Step big      the fourth fit, a larger network (-Hidden) on every fitting chunk, only if
#                  the three fits named in -After took under -MaxGpuMinutes together
#   -Step check    Task 15's checks (a) to (d) for one fit on the held-out chunks (cpu lane)
#   -Step summary  the one table over every fit found under runs\risk\300k (cpu lane)
# Data: runs\risk\data-300k (not in git; about 65 GB). Fits: runs\risk\300k\fit-<size>.
#
# Usage (from this checkout; the binary is a scratch build outside every checkout):
#   jobq.ps1 add -Lane cpu -Dir <checkout> -Script <checkout>\experiments\stage\risk300k.ps1 -ScriptArgs "-Step data"
#   jobq.ps1 add -Lane gpu -After <data> -Dir <checkout> -Script ...risk300k.ps1 -ScriptArgs "-Step fit -Train 0-29 -FitDir runs\risk\300k\fit-30k"
#   jobq.ps1 add -Lane cpu -After <fit> -Dir <checkout> -Script ...risk300k.ps1 -ScriptArgs "-Step check -FitDir runs\risk\300k\fit-30k"
#   jobq.ps1 add -Lane gpu -After <fit-290k> -Dir <checkout> -Script ...risk300k.ps1 -ScriptArgs "-Step big -FitDir runs\risk\300k\fit-290k-big -After 'runs\risk\300k\fit-30k','runs\risk\300k\fit-100k','runs\risk\300k\fit-290k'"
#   jobq.ps1 add -Lane cpu -After <checks> -Dir <checkout> -Script ...risk300k.ps1 -ScriptArgs "-Step summary"
param(
    [ValidateSet('data', 'fit', 'big', 'check', 'summary')][string]$Step = 'data',
    [int]$Deals = 300000,
    [int]$Chunk = 1000,
    [int]$Seed = 16,
    [string]$Holdout = '290-299',
    [string]$Train = '0-289',
    [string]$FitDir = 'runs\risk\300k\fit-290k',
    [int]$Epochs = 12,
    [int]$Patience = 3,
    [double]$Lr = 0.0005,
    [int[]]$Hidden = @(768, 384),
    [int]$HoldoutRows = 200000,
    [int]$Minutes = 300,
    [string]$Device = 'cuda',
    [string[]]$After = @(),
    [int]$MaxGpuMinutes = 180,
    [int]$MinFreeGb = 200,
    [string]$Bin = 'C:\Users\brian\.cache\hk-mahjong-t16\release\examples'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$threads = if ($env:HKQ_THREADS) { [int]$env:HKQ_THREADS } else { 8 }

# The shared conda env may import another checkout's package: insist on this one.
$py = 'C:\Users\brian\miniconda3\envs\mahjong\python.exe'
$env:PYTHONPATH = Join-Path $root 'python'
$where = & $py -c "import mahjong; print(mahjong.__file__)"
Write-Output "mahjong.__file__ = $where"
if (-not $where -or -not ([IO.Path]::GetFullPath($where)).StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { Write-Output "mahjong is not imported from $root"; exit 1 }

$data = Join-Path $root 'runs\risk\data-300k'
$fit = Join-Path $root $FitDir
$lo, $hi = $Holdout -split '-' | ForEach-Object { [int]$_ }
$holdoutChunks = ($lo..$hi) | ForEach-Object { "$_" }
$started = Get-Date
Write-Output "step $Step started $($started.ToString('s')), $threads threads"

function Fit-Minutes([string]$dir) {
    # Minutes a finished fit took, from the `done` line of its train.log (-1 if not finished).
    $log = Join-Path $root (Join-Path $dir 'train.log')
    if (-not (Test-Path $log)) { return -1 }
    $done = Get-Content $log | Where-Object { $_ -match '"event": "done"' } | Select-Object -Last 1
    if (-not $done) { return -1 }
    return [double]($done | ConvertFrom-Json).minutes
}

function Run-Fit([string]$train, [string]$dir, [int[]]$hidden) {
    $fitArgs = @('-m', 'mahjong.risk.fit', '--data', $data, '--out', $dir,
        '--holdout-chunks') + $holdoutChunks + @('--train-chunks', $train, '--holdout-rows', "$HoldoutRows",
        '--epochs', "$Epochs", '--patience', "$Patience", '--select', 'win', '--minutes', "$Minutes",
        '--threads', "$threads", '--device', $Device, '--lr', "$Lr", '--hidden') + ($hidden | ForEach-Object { "$_" })
    if (Test-Path (Join-Path $dir 'ckpt.pt')) { $fitArgs += '--resume' }
    & $py @fitArgs
    if ($LASTEXITCODE -ne 0) { Write-Output "fit exited with code $LASTEXITCODE"; exit $LASTEXITCODE }
}

switch ($Step) {
    'data' {
        $free = (Get-PSDrive C).Free / 1GB
        Write-Output ("C: free {0:N0} GB" -f $free)
        if ($free -lt $MinFreeGb) { Write-Output "under $MinFreeGb GB free: not logging (Task 16's stop rule)"; exit 1 }
        $exe = Join-Path $Bin 'risk_log.exe'
        if (-not (Test-Path $exe)) { Write-Output "missing $exe (cargo build --release -p mahjong-search --example risk_log with CARGO_TARGET_DIR outside the checkout)"; exit 1 }
        $env:RAYON_NUM_THREADS = "$threads"
        & $exe --deals $Deals --chunk $Chunk --seed $Seed --out $data --belief (Join-Path $root 'models\belief.mjnn')
        if ($LASTEXITCODE -ne 0) { Write-Output "risk_log exited with code $LASTEXITCODE"; exit $LASTEXITCODE }
        $size = (Get-ChildItem $data -Filter 'chunk-*.bin' | Measure-Object Length -Sum).Sum / 1GB
        Write-Output ("data: {0} chunks, {1:N1} GB on disk" -f (Get-ChildItem $data -Filter 'chunk-*.bin').Count, $size)
    }
    'fit' { Run-Fit $Train $fit $Hidden }
    'big' {
        $total = 0.0
        foreach ($d in $After) {
            $m = Fit-Minutes $d
            if ($m -lt 0) { Write-Output "$d has not finished: the larger fit waits for the three"; exit 1 }
            $total += $m
        }
        Write-Output ("the three fits took {0:N1} gpu minutes together" -f $total)
        if ($total -ge $MaxGpuMinutes) {
            New-Item -ItemType Directory -Force $fit | Out-Null
            Set-Content -Encoding utf8 (Join-Path $fit 'skipped.txt') ("skipped: the three fits took {0:N1} gpu minutes, the plan allows the larger network only under {1}" -f $total, $MaxGpuMinutes)
            Write-Output (Get-Content (Join-Path $fit 'skipped.txt'))
        } else {
            Run-Fit $Train $fit $Hidden
        }
    }
    'check' {
        if (-not (Test-Path (Join-Path $fit 'best.pt'))) { Write-Output "no best.pt in $fit (skipped fit?)"; exit 0 }
        $checkArgs = @('-m', 'mahjong.risk.check', '--data', $data, '--model', (Join-Path $fit 'best.pt'),
            '--out', (Join-Path $fit 'check.md'), '--threads', "$threads", '--chunks') + $holdoutChunks
        & $py @checkArgs
        if ($LASTEXITCODE -ne 0) { Write-Output "check exited with code $LASTEXITCODE"; exit $LASTEXITCODE }
    }
    'summary' {
        $fits = @(
            @('network, 30,000 deals', 'runs\risk\300k\fit-30k'),
            @('network, 100,000 deals', 'runs\risk\300k\fit-100k'),
            @('network, 290,000 deals', 'runs\risk\300k\fit-290k'),
            @('larger network, 290,000 deals', 'runs\risk\300k\fit-290k-big'))
        $sumArgs = @('-m', 'mahjong.risk.summary', '--data', $data, '--chunks', $Holdout, '--threads', "$threads",
            '--out', (Join-Path $root 'runs\risk\300k\summary.md'))
        foreach ($f in $fits) {
            if (Test-Path (Join-Path $root (Join-Path $f[1] 'best.pt'))) { $sumArgs += @('--model', "$($f[0])=$($f[1])") }
            else { Write-Output "no best.pt in $($f[1]): left out of the summary" }
        }
        & $py @sumArgs
        if ($LASTEXITCODE -ne 0) { Write-Output "summary exited with code $LASTEXITCODE"; exit $LASTEXITCODE }
    }
}
$ended = Get-Date
Write-Output ("step {0} ended {1}, wall time {2:N1} min" -f $Step, $ended.ToString('s'), ($ended - $started).TotalMinutes)
