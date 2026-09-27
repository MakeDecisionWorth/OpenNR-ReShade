# Building OpenNR

You need Visual Studio 2022 with the "Desktop development with C++" workload (it brings the
Windows SDK and `fxc`), and three things under `ext\` (not part of this repository):

| Path | What | From |
|---|---|---|
| `ext\reshade\include` | the ReShade add-on SDK headers | [crosire/reshade](https://github.com/crosire/reshade), `include\` (0.1.0 was built with commit `3645e30`, add-on API version 20) |
| `ext\imgui` | Dear ImGui headers, at the version the SDK pins (1.92.5 for the commit above) | the reshade repository's `deps\imgui` submodule |
| `ext\dxc` | *optional*: DXC with SPIR-V support, for Vulkan | the official [DirectXShaderCompiler release](https://github.com/microsoft/DirectXShaderCompiler/releases) (Windows zip, extracted as is). Without it the add-on builds without Vulkan support. |

Then, from a PowerShell prompt in the repository:

```powershell
.\build.ps1            # build\opennr.addon64
.\build.ps1 -Package   # also dist\OpenNR-<version>\ and dist\OpenNR-<version>.zip
```

`build.ps1 -Ext <path>` takes the three from somewhere else. The build compiles every shader with
`fxc` first (they are compiled again at runtime on D3D11/D3D12, where an error would only show as
an untouched frame), and with DXC generates `src\opennr_spirv.hpp` for Vulkan.

The two models are in `models\1pass` and `models\2pass`; the add-on loads them from an `OpenNR`
folder next to itself.
