# One duplicate match between any four bots (cpu lane). Output is UTF-8.
# Usage: jobq.ps1 add -Lane cpu -- powershell -File experiments\stage\versus.ps1 -Name stage-vs-value -A stage -B value -C value -D value
param([string]$Name, [string]$A, [string]$B, [string]$C, [string]$D, [int]$Deals = 40000, [int]$Seed = 1)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$env:RAYON_NUM_THREADS = $(if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' })
Set-Location $root
New-Item -ItemType Directory -Force "$root\runs\stage" | Out-Null
& "$root\target\release\examples\compare.exe" $A $B $C $D $Deals $Seed | Out-File -Encoding utf8 "$root\runs\stage\$Name.txt"
exit $LASTEXITCODE
