# Plan 2026-11-16 Task 23 (re-run by Task 27): the ceiling match of "avoid the dealing tile"
# (cpu lane). No Python. Runs a scratch build of examples\defence_ceiling.rs (outside every checkout):
#   $env:CARGO_TARGET_DIR = 'C:\Users\brian\.cache\hk-mahjong-t27'
#   cargo build --release -p mahjong-search --example defence_ceiling
# Writes runs\defence-ceiling\<Name>.txt (UTF-8) in this checkout.
# Usage:
#   jobq.ps1 add -Lane cpu -Label hk-ceiling -Dir <checkout> -Script <checkout>\experiments\stage\defence_ceiling.ps1 -ScriptArgs "-Deals 40000 -Seed 23"
param(
    [int]$Deals = 40000,
    [int]$Seed = 23,
    [string]$Name = '',
    [int]$Threads = 12,
    [string]$Bin = 'C:\Users\brian\.cache\hk-mahjong-t27\release\examples'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$exe = Join-Path $Bin 'defence_ceiling.exe'
if (-not (Test-Path $exe)) { Write-Output "missing $exe (build it first, see the top of this script)"; exit 1 }
if (-not $Name) { $Name = "match-$Deals-seed$Seed" }
$out = Join-Path $root 'runs\defence-ceiling'
New-Item -ItemType Directory -Force $out | Out-Null
$env:RAYON_NUM_THREADS = "$Threads"
& $exe match $Deals $Seed | Out-File -Encoding utf8 (Join-Path $out "$Name.txt")
exit $LASTEXITCODE
