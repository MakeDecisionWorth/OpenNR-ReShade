<#
    OpenNR installer.

    Copies OpenNR into a game that has ReShade (the version with full add-on support), or
    removes it again.

    Right-click this file and choose "Run with PowerShell", then drag the game's .exe into the
    window when asked. Or from a PowerShell prompt:

        .\install.ps1 -Game "D:\Games\SomeGame\bin\game.exe"
        .\install.ps1 -Game "D:\Games\SomeGame\bin" -Uninstall

    Copyright (c) 2026 MakeDecisionWorth. MIT licence, see LICENSE.
#>
param(
    [string]$Game = "",
    [switch]$Uninstall,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Finish([int]$code) {
    if (-not $NoPause) { [void](Read-Host "`nPress Enter to close") }
    exit $code
}
function Say([string]$text, [string]$color = "Gray") { Write-Host $text -ForegroundColor $color }

Say "OpenNR installer" "Cyan"
Say "----------------`n"

# ---- where the game is
if (-not $Game) {
    Say "Drag the game's .exe file (the one ReShade is installed next to) into this window,"
    Say "or type the path of its folder, then press Enter."
    $Game = Read-Host ">"
}
$Game = $Game.Trim().Trim('"').Trim("'")
if (-not $Game) { Say "No game given." "Yellow"; Finish 1 }
if (Test-Path -LiteralPath $Game -PathType Leaf) { $Game = Split-Path -Parent $Game }
if (-not (Test-Path -LiteralPath $Game -PathType Container)) {
    Say "Folder not found: $Game" "Red"
    Finish 1
}
$Game = (Resolve-Path -LiteralPath $Game).Path
Say "Game folder: $Game`n"

$addon = Join-Path $Game "opennr.addon64"
$models = Join-Path $Game "OpenNR"

# ---- uninstall
if ($Uninstall) {
    $removed = $false
    if (Test-Path -LiteralPath $addon) { Remove-Item -LiteralPath $addon -Force; $removed = $true }
    foreach ($m in @("1pass", "2pass")) {
        $p = Join-Path $models $m
        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Recurse -Force; $removed = $true }
    }
    if ((Test-Path -LiteralPath $models) -and -not (Get-ChildItem -LiteralPath $models -Force)) {
        Remove-Item -LiteralPath $models -Force
    }
    if ($removed) { Say "OpenNR removed. Its settings stay in ReShade.ini under [OpenNR]." "Green" }
    $saved = Join-Path $models "Saved views"
    if (Test-Path -LiteralPath $saved) { Say "Your saved views are kept in $saved." }
    else { Say "OpenNR is not installed in this folder." "Yellow" }
    Finish 0
}

# ---- the files to install, next to this script
$srcAddon = Join-Path $here "opennr.addon64"
$srcModels = Join-Path $here "OpenNR"
foreach ($f in @($srcAddon, (Join-Path $srcModels "1pass\model.txt"), (Join-Path $srcModels "2pass\model.txt"))) {
    if (-not (Test-Path -LiteralPath $f)) {
        Say "Missing $f -- extract the whole OpenNR zip and run install.ps1 from inside it." "Red"
        Finish 1
    }
}

# ---- is ReShade there?
$reshade = $null
foreach ($n in @("dxgi.dll", "d3d11.dll", "d3d12.dll", "opengl32.dll", "ReShade64.dll")) {
    $p = Join-Path $Game $n
    if (Test-Path -LiteralPath $p) {
        $vi = (Get-Item -LiteralPath $p).VersionInfo
        if ($vi.ProductName -match "ReShade") { $reshade = $vi; break }
    }
}
$ini = Join-Path $Game "ReShade.ini"
if ($reshade) {
    Say "Found ReShade $($reshade.FileVersion)."
    $v = $reshade.FileVersion -split '\.'
    if ([int]$v[0] -lt 6 -or ([int]$v[0] -eq 6 -and [int]$v[1] -lt 8)) {
        Say "OpenNR needs ReShade 6.8 or newer. Update ReShade (with full add-on support) first." "Yellow"
    }
} elseif (Test-Path -LiteralPath $ini) {
    Say "Found ReShade.ini (ReShade may be installed as a Vulkan layer)."
} else {
    Say "ReShade does not seem to be installed in this folder." "Yellow"
    Say "Install ReShade with full add-on support into this game first (reshade.me)." "Yellow"
    $a = Read-Host "Install OpenNR here anyway? (y/N)"
    if ($a -notmatch '^[yY]') { Finish 1 }
}
Say "OpenNR needs the ReShade version WITH FULL ADD-ON SUPPORT. The standard download will not"
Say "load it.`n"

# ---- copy
try {
    Copy-Item -LiteralPath $srcAddon -Destination $addon -Force
    New-Item -ItemType Directory -Force -Path $models | Out-Null
    foreach ($m in @("1pass", "2pass")) {
        $dst = Join-Path $models $m
        if (Test-Path -LiteralPath $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
        Copy-Item -LiteralPath (Join-Path $srcModels $m) -Destination $dst -Recurse -Force
    }
} catch {
    Say "Copying failed: $($_.Exception.Message)" "Red"
    Say "If the game is running, close it and try again. A game under 'Program Files' may need" "Red"
    Say "this installer to run as administrator." "Red"
    Finish 1
}
Say "Installed:" "Green"
Say "  $addon"
Say "  $models\1pass, $models\2pass"

# ---- switched off in ReShade?
if (Test-Path -LiteralPath $ini) {
    $line = Select-String -LiteralPath $ini -Pattern '^\s*DisabledAddons\s*=' | Select-Object -First 1
    if ($line -and $line.Line -match 'OpenNR') {
        Say "`nOpenNR is switched off in ReShade's settings. In the game, open ReShade (Home key)," "Yellow"
        Say "go to the Add-ons tab and tick OpenNR." "Yellow"
    }
}

Say "`nStart the game, press Home to open ReShade and go to the Add-ons tab. OpenNR's settings"
Say "are there."
Finish 0
