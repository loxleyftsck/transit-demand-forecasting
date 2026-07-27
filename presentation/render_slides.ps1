param(
  [Parameter(Mandatory=$true)][string]$Deck,
  [Parameter(Mandatory=$true)][string]$OutDir
)

$ErrorActionPreference = "Stop"
$deckPath = (Resolve-Path $Deck).Path
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$outPath = (Resolve-Path $OutDir).Path

$ppt = New-Object -ComObject PowerPoint.Application
try {
  $pres = $ppt.Presentations.Open($deckPath, $true, $false, $false)  # ReadOnly, no window
  try {
    $pres.Export($outPath, "PNG", 1600, 900)
    Write-Output "Exported $($pres.Slides.Count) slides to $outPath"
  } finally {
    $pres.Close()
  }
} finally {
  $ppt.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
