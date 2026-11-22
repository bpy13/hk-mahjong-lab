# Plan 2026-11-16 Task 24: the paired false-alarm log (cpu lane). No Python.
# Runs a scratch build of examples\defence_false_alarms.rs (outside every checkout):
#   $env:CARGO_TARGET_DIR = 'C:\Users\brian\.cache\hk-mahjong-t24'
#   cargo build --release -p mahjong-search --example defence_false_alarms
# Reads models\belief-order.mjnn of this checkout; writes runs\defence-false-alarms\<Name>.txt (UTF-8).
# Usage:
#   jobq.ps1 add -Lane cpu -Label hk-false-alarms -Dir <checkout> -Script <checkout>\experiments\stage\false_alarms.ps1 -ScriptArgs "-Deals 40000 -Seed 24"
param(
    [int]$Deals = 40000,
    [int]$Seed = 24,
    [string]$Bots = '',
    [string]$Name = '',
    [int]$Threads = 12,
    [string]$Bin = 'C:\Users\brian\.cache\hk-mahjong-t24\release\examples'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$exe = Join-Path $Bin 'defence_false_alarms.exe'
if (-not (Test-Path $exe)) { Write-Output "missing $exe (build it first, see the top of this script)"; exit 1 }
if (-not $Name) { $Name = "match-$Deals-seed$Seed" }
$out = Join-Path $root 'runs\defence-false-alarms'
New-Item -ItemType Directory -Force $out | Out-Null
$env:RAYON_NUM_THREADS = "$Threads"
$botArgs = @($Bots -split '[ ,]' | Where-Object { $_ })
& $exe $Deals $Seed (Join-Path $root 'models\belief-order.mjnn') @botArgs | Out-File -Encoding utf8 (Join-Path $out "$Name.txt")
exit $LASTEXITCODE
