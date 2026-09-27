# Builds opennr.addon64, and with -Package the release folder and zip.
#
#     .\build.ps1                     # build\opennr.addon64
#     .\build.ps1 -Package            # + dist\OpenNR-<version>\ and dist\OpenNR-<version>.zip
#     .\build.ps1 -Ext D:\sdk\ext     # the headers and tools live elsewhere
#
# Needs Visual Studio 2022 (C++ desktop workload) and, under -Ext (default: .\ext):
#   ext\reshade\include   the ReShade add-on SDK headers (github.com/crosire/reshade, include\)
#   ext\imgui             Dear ImGui at the version the SDK pins (the reshade repo's submodule)
#   ext\dxc               optional: DXC with SPIR-V (the official DirectXShaderCompiler release).
#                         Without it the add-on builds without Vulkan support.
# See BUILDING.md.
#
# Copyright (c) 2026 MakeDecisionWorth. MIT licence, see LICENSE.

param(
    [switch]$Package,
    [string]$Ext = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $root "src"
if (-not $Ext) { $Ext = Join-Path $root "ext" }
$sdk = Join-Path $Ext "reshade\include"
$imgui = Join-Path $Ext "imgui"
if (-not (Test-Path (Join-Path $sdk "reshade.hpp"))) { throw "no ReShade SDK at $sdk -- see BUILDING.md" }
if (-not (Test-Path (Join-Path $imgui "imgui.h"))) { throw "no imgui.h at $imgui -- see BUILDING.md" }

$version = [regex]::Match((Get-Content (Join-Path $src "opennr_addon.cpp") -Raw),
                          'kVersion = "([^"]+)"').Groups[1].Value
if (-not $version) { throw "no kVersion in src\opennr_addon.cpp" }

# MSVC: the add-on is a Windows DLL against the D3D and Win32 headers
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsRoot = & $vswhere -latest -products * -property installationPath
$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
& cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^(LIB|INCLUDE|LIBPATH|PATH)=(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}

# ---- the shaders compile at runtime on D3D, where an error would only show as an untouched
# frame, so every one is compiled here first
$all = (Get-ChildItem (Join-Path $src "*") -Include *.cpp, *.hpp -File | Get-Content -Raw) -join "`n"
$fxc = (Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Filter fxc.exe -Recurse -EA SilentlyContinue |
        Where-Object { $_.FullName -match "\\x64\\" } | Sort-Object FullName | Select-Object -Last 1).FullName
if ($fxc) {
    $i = 0
    foreach ($m in [regex]::Matches($all, 'R"\((.*?)\)";', 'Singleline')) {
        $i++
        $tmp = Join-Path $env:TEMP "opennr_shader_$i.hlsl"
        [System.IO.File]::WriteAllText($tmp, $m.Groups[1].Value)
        $old = $ErrorActionPreference
        $ErrorActionPreference = "Continue"         # fxc warns on stderr; the exit code decides
        $r = & $fxc /nologo /T cs_5_0 /E main /Fo "$env:TEMP\opennr_shader_$i.cso" $tmp 2>&1
        $ErrorActionPreference = $old
        if ($LASTEXITCODE -ne 0) {
            Write-Output ($r | Out-String)
            throw "shader $i does not compile"
        }
    }
    if ($i -eq 0) { throw "the shader check found no shaders" }
    Write-Output "shaders: $i compiled"
} else {
    Write-Warning "fxc not found; the shaders are only checked when the game loads them"
}

# ---- the same shaders as SPIR-V for Vulkan, compiled here and embedded in opennr_spirv.hpp
#   * `cbuffer Args : register(b0)` becomes a [[vk::push_constant]] struct, each member copied
#     into a static of its old name (every member must be a plain uint)
#   * t0, t1, t2 and u0 go to descriptor sets 0-3, binding 0, as the add-on's layouts declare
#   * a RWTexture2D<float4> is declared rgba16f, the format of the add-on's Vulkan output texture
$dxc = Join-Path $Ext "dxc\bin\x64\dxc.exe"
$spvOut = Join-Path $src "opennr_spirv.hpp"
function ConvertTo-PushConstants([string]$hlsl, [string]$name) {
    $m = [regex]::Match($hlsl, 'cbuffer\s+Args\s*:\s*register\(b0\)\s*\{(.*?)\}\s*;', 'Singleline')
    if (-not $m.Success) { throw "${name}: no 'cbuffer Args : register(b0)'" }
    $body = ($m.Groups[1].Value -replace '//[^\r\n]*', '').Trim()
    if ($body -notmatch '^uint\s') { throw "${name}: the push constants are not all uint" }
    $names = @(($body -replace '^uint\s+', '' -replace ';\s*$', '') -split '\s*,\s*' |
               ForEach-Object { $_.Trim() } | Where-Object { $_ })
    foreach ($n in $names) {
        if ($n -notmatch '^\w+$') { throw "${name}: push-constant member '$n' is not a plain uint" }
    }
    $decl = "struct ArgsT { uint " + ($names -join ', ') + "; }; [[vk::push_constant]] ArgsT Args; " +
            "static uint " + (($names | ForEach-Object { "$_ = Args.$_" }) -join ', ') + ";"
    $out = $hlsl.Substring(0, $m.Index) + $decl + $hlsl.Substring($m.Index + $m.Length)
    $out -replace '(?m)^(\s*)RWTexture2D<float4>', '$1[[vk::image_format("rgba16f")]] RWTexture2D<float4>'
}
$hdr = New-Object System.Text.StringBuilder
[void]$hdr.AppendLine("// Generated by build.ps1 from opennr_shaders.hpp -- do not edit.")
[void]$hdr.AppendLine("#pragma once")
[void]$hdr.AppendLine("#include <cstddef>")
[void]$hdr.AppendLine("#include <cstdint>")
[void]$hdr.AppendLine("#include <cstring>")
[void]$hdr.AppendLine("namespace opennr { namespace spirv {")
[void]$hdr.AppendLine("struct Blob { const char* name; const uint32_t* code; size_t size; };")
if (Test-Path $dxc) {
    $shaderSrc = [System.IO.File]::ReadAllText((Join-Path $src "opennr_shaders.hpp"))
    $blobs = @()
    foreach ($m in [regex]::Matches($shaderSrc, '(k\w+HLSL)\s*=\s*R"\((.*?)\)";', 'Singleline')) {
        $name = $m.Groups[1].Value
        $tmp = Join-Path $env:TEMP "opennr_vk_$name.hlsl"
        $spv = Join-Path $env:TEMP "opennr_vk_$name.spv"
        [System.IO.File]::WriteAllText($tmp, (ConvertTo-PushConstants $m.Groups[2].Value $name))
        $old = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $r = & $dxc -spirv -T cs_6_0 -E main -O3 "-fspv-target-env=vulkan1.1" `
                 -fvk-bind-register t0 0 0 0 -fvk-bind-register t1 0 0 1 `
                 -fvk-bind-register t2 0 0 2 -fvk-bind-register u0 0 0 3 `
                 $tmp -Fo $spv 2>&1
        $ErrorActionPreference = $old
        if ($LASTEXITCODE -ne 0) {
            Write-Output ($r | Out-String)
            throw "$name does not compile to SPIR-V"
        }
        $bytes = [System.IO.File]::ReadAllBytes($spv)
        $words = for ($k = 0; $k -lt $bytes.Length; $k += 4) {
            "0x{0:x8}u" -f [BitConverter]::ToUInt32($bytes, $k)
        }
        [void]$hdr.AppendLine("inline const uint32_t ${name}[] = {")
        for ($k = 0; $k -lt $words.Count; $k += 8) {
            $end = [Math]::Min($k + 7, $words.Count - 1)
            [void]$hdr.AppendLine("    " + ($words[$k..$end] -join ", ") + ",")
        }
        [void]$hdr.AppendLine("};")
        $blobs += $name
    }
    if ($blobs.Count -eq 0) { throw "the SPIR-V step found no shaders" }
    [void]$hdr.AppendLine("inline constexpr bool kHave = true;")
    [void]$hdr.AppendLine("inline const Blob kAll[] = {")
    foreach ($b in $blobs) { [void]$hdr.AppendLine("    {`"$b`", $b, sizeof($b)},") }
    [void]$hdr.AppendLine("};")
    [void]$hdr.AppendLine("inline const Blob* find(const char* name) {")
    [void]$hdr.AppendLine("    for (const Blob& b : kAll) if (!std::strcmp(b.name, name)) return &b;")
    [void]$hdr.AppendLine("    return nullptr;")
    [void]$hdr.AppendLine("}")
    Write-Output "SPIR-V: $($blobs.Count) compiled"
} else {
    [void]$hdr.AppendLine("inline constexpr bool kHave = false;")
    [void]$hdr.AppendLine("inline const Blob* find(const char*) { return nullptr; }")
    Write-Warning "no DXC at $dxc -- building without Vulkan support"
}
[void]$hdr.AppendLine("} }  // namespace opennr::spirv")
[System.IO.File]::WriteAllText($spvOut, $hdr.ToString())

# ---- the add-on
$build = Join-Path $root "build"
New-Item -ItemType Directory -Force $build | Out-Null
$out = Join-Path $build "opennr.addon64"
$argv = @(
    "/nologo", "/std:c++17", "/EHsc", "/O2", "/MD", "/LD",
    "/I", $sdk, "/I", $imgui,
    "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX",
    "/Fo$(Join-Path $build 'opennr_addon.obj')", (Join-Path $src "opennr_addon.cpp"),
    "/link", "/OUT:$out", "/IMPLIB:$(Join-Path $build 'opennr_addon.lib')",
    "d3dcompiler.lib", "user32.lib", "d3d11.lib", "dxgi.lib"
)
& cl @argv
if ($LASTEXITCODE -ne 0) { throw "compile failed ($LASTEXITCODE)" }
Get-ChildItem $build -Include *.obj, *.exp, *.lib -File -Recurse -EA SilentlyContinue | Remove-Item -Force
Write-Output "built $out (OpenNR $version)"

# ---- the release: what a player downloads
if ($Package) {
    $name = "OpenNR-$version"
    $dist = Join-Path $root "dist"
    $stage = Join-Path $dist $name
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force (Join-Path $stage "OpenNR") | Out-Null
    Copy-Item $out $stage
    foreach ($m in @("1pass", "2pass")) {
        $from = Join-Path $root "models\$m"
        if (-not (Test-Path (Join-Path $from "model.txt"))) { throw "no models\$m\model.txt" }
        Copy-Item -Recurse $from (Join-Path $stage "OpenNR\$m")
    }
    foreach ($f in @("install.ps1", "README.md", "LICENSE", "THIRD-PARTY-NOTICES.md")) {
        Copy-Item (Join-Path $root $f) $stage
    }
    $zip = Join-Path $dist "$name.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path $stage -DestinationPath $zip
    Write-Output "packaged $zip"
}
