# OpenNR for ReShade

OpenNR is a ReShade add-on that runs a small neural network on your game's picture, every frame.
It re-balances local tone, local contrast and colour for a more natural, photographic look.

Version 0.1.0 — the first release.

## What you need

- Windows 10 or 11, 64-bit
- **ReShade 6.8 or newer, the version with full add-on support**
  ([reshade.me](https://reshade.me) — the standard download cannot load add-ons). Tested with
  ReShade 6.8.0.
- A DirectX 11 or DirectX 12 game. Vulkan games work too, but that is experimental.
- A GPU with DirectX 11 compute shaders. Tested on Intel Arc A770 and A750.

ReShade with add-on support is not meant for online games with anti-cheat. Don't use it there.

## Install

1. Install ReShade **with full add-on support** into your game, if you haven't already.
2. Download `OpenNR-0.1.0.zip` from [Releases](https://github.com/MakeDecisionWorth/OpenNR-ReShade/releases)
   and extract it.
3. Right-click `install.ps1` → **Run with PowerShell**. When it asks, drag the game's `.exe` (the
   one ReShade is installed next to) into the window and press Enter.
4. Start the game, press **Home** to open ReShade, and go to the **Add-ons** tab. OpenNR and its
   settings are there.

To install by hand instead, copy `opennr.addon64` and the `OpenNR` folder next to the game's `.exe`.

## Settings

| Setting | What it does |
|---|---|
| **Run the network on** | The game's GPU, or a second GPU in your PC. On a second GPU, OpenNR doesn't slow the game's GPU down, but the picture is shown two frames late. An integrated GPU is usually too slow for this. |
| **Correct before the game draws its HUD** | On by default. OpenNR corrects the picture before the game draws its HUD, so the HUD looks as the game drew it. The line under it says whether this works in the current game and scene; when it doesn't, the whole frame is corrected, HUD included. While this is active, your other ReShade effects also run before the HUD. |
| **Strength** | How much of the correction is applied. 0% leaves the picture as it was. |
| **Local Tone Strength** | Mostly affects colour. |
| **Local Structure Strength** | Mostly affects local contrast. 10–100% gives the most predictable results for both of these. |
| **Model A / B / C** | A: neutral. B: slightly darker (−0.1 EV), −25% contrast, −10% saturation. C: −15% saturation. B and C follow Local Tone, up to 100%. |
| **Correction Model** | **1 pass** or **2 passes**. 2 passes is a stronger and differently shaped correction, not simply more of the same. Both cost the same. |
| **View** | **Final frame** is the normal picture. **Correction ×20** shows what OpenNR changes, magnified: mid-grey is unchanged, brighter is added, darker is taken away. **Source** shows the picture exactly as OpenNR reads it. |
| **Depth buffer** | Information only; OpenNR doesn't use depth. If it says *REJECTED*, ReShade's depth detection picked something that isn't the scene (a shadow map or a mirror). In the Generic Depth settings, set "Aspect ratio heuristic" to "Similar aspect ratio". |
| **Resolution** | OpenNR works at up to 704 lines (keeping your aspect ratio) and scales its correction up to your resolution. The picture itself stays at full resolution. |
| **GPU time** | How long OpenNR takes on your GPU each frame, and the amount of work (GMAC). |

All settings except View are saved in `ReShade.ini` under `[OpenNR]`.

## Uninstall

Run `install.ps1` again from a PowerShell prompt with `-Uninstall`:

```powershell
.\install.ps1 -Game "D:\Games\SomeGame\bin" -Uninstall
```

Or delete `opennr.addon64` and the `OpenNR` folder from the game folder.

## Troubleshooting

- **OpenNR isn't in the Add-ons tab.** ReShade was installed without add-on support. Reinstall it
  and choose the version with full add-on support.
- **OpenNR is listed but unticked.** Tick it in the Add-ons tab.
- **"Setup failed" in OpenNR's settings.** Open `ReShade.log` in the game folder and look for lines
  starting with `OpenNR:`.
- **The picture doesn't change.** Check that Strength is above 0% and View is "Final frame".
- **The HUD looks corrected too.** The line under "Correct before the game draws its HUD" explains
  why for the current scene. Menus and loading screens are often corrected whole.

## License

MIT — see [LICENSE](LICENSE). OpenNR is built with the ReShade add-on SDK and Dear ImGui; see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). To build it yourself, see
[BUILDING.md](BUILDING.md).
