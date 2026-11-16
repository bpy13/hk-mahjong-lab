# Task 9 (stage 3 gate): one match or one stage count from a scratch build (cpu lane).
# The binaries come from a scratch CARGO_TARGET_DIR outside every checkout ($Bin), so no
# checkout's target folder is rebuilt while other jobs read it. Output is UTF-8 in runs\stage.
# Usage (from the checkout root):
#   jobq.ps1 add -Lane cpu -Label t9-gate1-vs-value -Dir <checkout> -Script <checkout>\experiments\stage\gate.ps1 -ScriptArgs "-Name gate1-vs-value -A stage-gate1 -B value -C value -D value"
#   jobq.ps1 add -Lane cpu -Label t9-stage-counts -Dir <checkout> -Script <checkout>\experiments\stage\gate.ps1 -ScriptArgs "-Name stage-counts -Exe dealin -Deals 10000"
param(
    [string]$Name,
    [ValidateSet('compare', 'dealin')][string]$Exe = 'compare',
    [string]$A, [string]$B, [string]$C, [string]$D,
    [int]$Deals = 40000,
    [int]$Seed = 1,
    [string]$Bin = 'C:\Users\brian\.cache\hk-mahjong-t9\bin'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$env:RAYON_NUM_THREADS = $(if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' })
Set-Location $root
New-Item -ItemType Directory -Force "$root\runs\stage" | Out-Null
$out = "$root\runs\stage\t9-$Name.txt"
if ($Exe -eq 'compare') {
    & "$Bin\compare.exe" $A $B $C $D $Deals $Seed | Out-File -Encoding utf8 $out
    exit $LASTEXITCODE
}
# dealin: the stage counts for gate 0, 1 and 2 on the same deals, one after another.
Remove-Item -ErrorAction SilentlyContinue $out
foreach ($gate in 0, 1, 2) {
    & "$Bin\dealin.exe" $Deals $Seed $gate | Out-File -Append -Encoding utf8 $out
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
exit 0
