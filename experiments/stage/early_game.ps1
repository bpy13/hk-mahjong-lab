# Plan 2026-11-16 Task 30: the early-game variants of the three-stage bot, each one
# against three plain `stage`, every rotation beside its all-plain twin (cpu lane).
# No Python. Runs a scratch build of examples\flush_neighbours.rs (outside every checkout):
#   $env:CARGO_TARGET_DIR = 'C:\Users\brian\.cache\hk-mahjong-t30'
#   cargo build --release -p mahjong-search --example flush_neighbours
# Writes runs\early\<Name>.txt (UTF-8) in this checkout.
# Usage:
#   jobq.ps1 add -Lane cpu -Label hk-early -Dir <checkout> -Script <checkout>\experiments\stage\early_game.ps1 -ScriptArgs "-Deals 40000 -Seed 30"
param(
    [int]$Deals = 40000,
    [int]$Seed = 30,
    [string]$Name = '',
    [int]$Threads = 12,
    [string]$Bin = 'C:\Users\brian\.cache\hk-mahjong-t30\release\examples'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$exe = Join-Path $Bin 'flush_neighbours.exe'
if (-not (Test-Path $exe)) { Write-Output "missing $exe (build it first, see the top of this script)"; exit 1 }
if (-not $Name) { $Name = "one-$Deals-seed$Seed" }
$out = Join-Path $root 'runs\early'
New-Item -ItemType Directory -Force $out | Out-Null
$env:RAYON_NUM_THREADS = "$Threads"
& $exe one $Deals $Seed stage-early-suit stage-early-far stage-early-value stage-early-ab stage-early | Out-File -Encoding utf8 (Join-Path $out "$Name.txt")
exit $LASTEXITCODE
