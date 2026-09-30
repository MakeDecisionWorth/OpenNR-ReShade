// Save views: one frame's pictures, written to OpenNR\Saved views\<date>-<time>\ beside the add-on.
//
//   1_original.png        the picture as the game drew it
//   2_network_input.png   what the network sees, at the resolution it runs at
//   3_network_output.png  the network's own picture, before Strength, Style, Detail and Colour
//   4_difference_x20.png  3 minus 2, x20 on mid-grey
//   5_final.png           the finished picture, with the player's settings
//   settings.txt
//
// 16-bit PNGs. On an HDR swap chain every picture is the frame as the network is shown it: over
// the white point, sRGB-encoded and clipped.
//
// Copyright (c) 2026 MakeDecisionWorth. MIT licence, see LICENSE.

#pragma once

#include <reshade.hpp>

#include <DirectXPackedVector.h>
#include <objbase.h>
#include <wincodec.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

namespace opennr { namespace views {

using reshade::api::format;

struct Texture {
    uint32_t w = 0, h = 0;
    format fmt = format::unknown;
    std::vector<uint8_t> bytes;                  // rows tight
};

struct Info {
    std::string version, model, network;
    uint32_t strength = 100, tone = 100, structure = 100, style = 0, detail = 100, colour = 100;
    uint32_t apply = 1;
    uint32_t hdr = 0, white = 250;               // 0 SDR, 1 scRGB, 2 HDR10; percent of 80 nits
    uint32_t head_stride = 4, head_group = 0;    // the network's output: halves a pixel, which four
    format final_fmt = format::unknown;
};

struct Data {
    Texture original, composed;
    std::vector<uint16_t> head;                  // fp16
    uint32_t w = 0, h = 0;                       // the network's resolution
    Info info;
};

// what the overlay shows under the button
inline std::mutex& status_mutex() { static std::mutex m; return m; }
inline std::string& status_text() { static std::string s; return s; }
inline void set_status(std::string s) {
    std::lock_guard<std::mutex> lk(status_mutex());
    status_text() = std::move(s);
}
inline std::string get_status() {
    std::lock_guard<std::mutex> lk(status_mutex());
    return status_text();
}

inline uint32_t bytes_per_pixel(format f) {
    switch (reshade::api::format_to_default_typed(f, 0)) {
    case format::r16g16b16a16_float: return 8;
    case format::r8g8b8a8_unorm: case format::b8g8r8a8_unorm: case format::b8g8r8x8_unorm:
    case format::r10g10b10a2_unorm: case format::r11g11b10_float: return 4;
    default: return 0;
    }
}

// the stored values as the shaders read them (an _SRGB back buffer is read as its bytes)
inline std::vector<float> to_rgb(const Texture& t) {
    using namespace DirectX::PackedVector;
    std::vector<float> rgb(size_t(t.w) * t.h * 3);
    const uint32_t bpp = bytes_per_pixel(t.fmt);
    const format f = reshade::api::format_to_default_typed(t.fmt, 0);
    for (size_t i = 0, n = size_t(t.w) * t.h; i < n; ++i) {
        const uint8_t* s = t.bytes.data() + i * bpp;
        float* c = rgb.data() + i * 3;
        switch (f) {
        case format::r16g16b16a16_float: {
            HALF h[3];
            std::memcpy(h, s, 6);
            for (int k = 0; k < 3; ++k) c[k] = XMConvertHalfToFloat(h[k]);
            break;
        }
        case format::r11g11b10_float: {
            XMFLOAT3PK pk;
            std::memcpy(&pk.v, s, 4);
            DirectX::XMFLOAT3 v;
            DirectX::XMStoreFloat3(&v, XMLoadFloat3PK(&pk));
            c[0] = v.x; c[1] = v.y; c[2] = v.z;
            break;
        }
        case format::r10g10b10a2_unorm: {
            uint32_t p;
            std::memcpy(&p, s, 4);
            for (int k = 0; k < 3; ++k) c[k] = float((p >> (10 * k)) & 1023u) / 1023.0f;
            break;
        }
        case format::b8g8r8a8_unorm: case format::b8g8r8x8_unorm:
            c[0] = s[2] / 255.0f; c[1] = s[1] / 255.0f; c[2] = s[0] / 255.0f;
            break;
        default:
            c[0] = s[0] / 255.0f; c[1] = s[1] / 255.0f; c[2] = s[2] / 255.0f;
        }
    }
    return rgb;
}

// HDR: as the network is shown it -- linear BT.709 (1.0 = 80 nits) over the white point,
// sRGB-encoded (kEncodeHLSL). SDR is left as it is.
inline std::vector<float> to_display(std::vector<float> v, uint32_t hdr, float white) {
    if (hdr == 0) return v;
    auto pq = [](float e) {
        const float p = std::pow(std::clamp(e, 0.0f, 1.0f), 1.0f / 78.84375f);
        return std::pow(std::max(p - 0.8359375f, 0.0f) / (18.8515625f - 18.6875f * p),
                        1.0f / 0.1593017578125f);
    };
    auto enc = [](float x) {
        x = std::clamp(x, 0.0f, 1.0f);
        return x <= 0.0031308f ? x * 12.92f : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    };
    for (size_t i = 0; i < v.size(); i += 3) {
        float c[3] = {v[i], v[i + 1], v[i + 2]};
        if (hdr == 2) {
            const float y[3] = {pq(c[0]) * 125.0f, pq(c[1]) * 125.0f, pq(c[2]) * 125.0f};
            c[0] = 1.6604910f * y[0] - 0.5876411f * y[1] - 0.0728499f * y[2];
            c[1] = -0.1245505f * y[0] + 1.1328999f * y[1] - 0.0083494f * y[2];
            c[2] = -0.0181508f * y[0] - 0.1005789f * y[1] + 1.1187297f * y[2];
        }
        for (int k = 0; k < 3; ++k) v[i + k] = enc(c[k] / white);
    }
    return v;
}

// the frame at the network's resolution, sampled as the network's input is: pixel centres,
// bilinear, edges clamped
inline std::vector<float> resample(const std::vector<float>& rgb, uint32_t OW, uint32_t OH,
                                   uint32_t W, uint32_t H) {
    if (W == OW && H == OH) return rgb;
    std::vector<float> out(size_t(W) * H * 3);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const float fx = (float(x) + 0.5f) * (float(OW) / float(W)) - 0.5f;
            const float fy = (float(y) + 0.5f) * (float(OH) / float(H)) - 0.5f;
            const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
            const float wx = fx - float(x0), wy = fy - float(y0);
            const int xc = std::clamp(x0, 0, int(OW) - 1), x1 = std::clamp(x0 + 1, 0, int(OW) - 1);
            const int yc = std::clamp(y0, 0, int(OH) - 1), y1 = std::clamp(y0 + 1, 0, int(OH) - 1);
            for (int k = 0; k < 3; ++k) {
                auto at = [&](int xx, int yy) { return rgb[(size_t(yy) * OW + xx) * 3 + k]; };
                const float top = at(xc, yc) + (at(x1, yc) - at(xc, yc)) * wx;
                const float bot = at(xc, y1) + (at(x1, y1) - at(xc, y1)) * wx;
                out[(size_t(y) * W + x) * 3 + k] = top + (bot - top) * wy;
            }
        }
    return out;
}

inline bool write_png(const std::filesystem::path& file, uint32_t w, uint32_t h,
                      const std::vector<float>& rgb) {
    std::vector<uint16_t> px(rgb.size());
    for (size_t i = 0; i < rgb.size(); ++i)
        px[i] = uint16_t(std::lround(std::clamp(rgb[i], 0.0f, 1.0f) * 65535.0f));
    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapEncoder* encoder = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* props = nullptr;
    bool ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                         IID_PPV_ARGS(&factory)));
    ok = ok && SUCCEEDED(factory->CreateStream(&stream));
    ok = ok && SUCCEEDED(stream->InitializeFromFilename(file.c_str(), GENERIC_WRITE));
    ok = ok && SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder));
    ok = ok && SUCCEEDED(encoder->Initialize(stream, WICBitmapEncoderNoCache));
    ok = ok && SUCCEEDED(encoder->CreateNewFrame(&frame, &props));
    ok = ok && SUCCEEDED(frame->Initialize(props));
    ok = ok && SUCCEEDED(frame->SetSize(w, h));
    WICPixelFormatGUID pf = GUID_WICPixelFormat48bppRGB;
    ok = ok && SUCCEEDED(frame->SetPixelFormat(&pf)) && IsEqualGUID(pf, GUID_WICPixelFormat48bppRGB);
    ok = ok && SUCCEEDED(frame->WritePixels(h, w * 6, UINT(px.size() * 2),
                                            reinterpret_cast<BYTE*>(px.data())));
    ok = ok && SUCCEEDED(frame->Commit());
    ok = ok && SUCCEEDED(encoder->Commit());
    for (IUnknown* u : {static_cast<IUnknown*>(props), static_cast<IUnknown*>(frame),
                        static_cast<IUnknown*>(encoder), static_cast<IUnknown*>(stream),
                        static_cast<IUnknown*>(factory)})
        if (u) u->Release();
    return ok;
}

// on a thread of its own: converting and compressing five pictures takes a moment
inline void write_all(std::filesystem::path dir, Data d) {
    using namespace DirectX::PackedVector;
    set_status("saving...");
    const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    bool ok = !ec;

    const Info& i = d.info;
    const float white = float(i.white) / 100.0f;
    Texture composed = d.composed;
    composed.fmt = i.final_fmt;
    const auto original = to_display(to_rgb(d.original), i.hdr, white);
    const auto final_ = to_display(to_rgb(composed), i.hdr, white);
    const auto input = resample(original, d.original.w, d.original.h, d.w, d.h);
    std::vector<float> output(input.size()), diff(input.size());
    for (size_t p = 0, n = size_t(d.w) * d.h; p < n; ++p)
        for (int k = 0; k < 3; ++k) {
            const float h = XMConvertHalfToFloat(d.head[p * i.head_stride + 4 * i.head_group + k]);
            const float v = input[p * 3 + k];
            output[p * 3 + k] = std::clamp(v + 0.25f * h, 0.0f, 1.0f);
            diff[p * 3 + k] = 0.5f + 20.0f * (output[p * 3 + k] - v);
        }
    ok = ok && write_png(dir / "1_original.png", d.original.w, d.original.h, original);
    ok = ok && write_png(dir / "2_network_input.png", d.w, d.h, input);
    ok = ok && write_png(dir / "3_network_output.png", d.w, d.h, output);
    ok = ok && write_png(dir / "4_difference_x20.png", d.w, d.h, diff);
    ok = ok && write_png(dir / "5_final.png", composed.w, composed.h, final_);

    if (ok) {
        std::ofstream f(dir / "settings.txt");
        char buf[64];
        auto factor = [&](uint32_t pct) {
            std::snprintf(buf, sizeof buf, "%.2f", double(pct) / 100.0);
            return std::string(buf);
        };
        std::snprintf(buf, sizeof buf, "%.2f (%.0f nits)", double(white), double(i.white) * 0.8);
        const std::string wp = buf;
        f << "OpenNR " << i.version << " -- saved views\n\n"
          << "Correction Model   " << i.model << "\n"
          << "Style              " << (i.style == 1 ? "Natural" : i.style == 2 ? "Cinematic" : "Standard") << "\n"
          << "Strength           " << factor(i.strength) << "\n"
          << "Local Tone         " << factor(i.tone) << "\n"
          << "Local Structure    " << factor(i.structure) << "\n"
          << "Detail strength    " << factor(i.detail) << "\n"
          << "Colour strength    " << factor(i.colour) << "\n"
          << "Apply the model    " << (i.apply ? "on" : "off") << "\n"
          << "Output             "
          << (i.hdr == 1 ? "HDR (scRGB), white point " + wp : i.hdr == 2 ? "HDR10, white point " + wp : "SDR") << "\n"
          << "Network ran on     " << i.network << "\n"
          << "Network resolution " << d.w << " x " << d.h << "\n\n"
          << "1_original          the picture as the game drew it\n"
          << "2_network_input     what the network sees, at the resolution it runs at\n"
          << "3_network_output    the network's own picture, before Strength, Style, Detail and Colour\n"
          << "4_difference_x20    what the network changes, x20: mid-grey is unchanged, brighter is\n"
          << "                    added, darker is taken away\n"
          << "5_final             the finished picture, with the settings above\n";
        if (i.hdr)
            f << "\nThe game's output is HDR. Every picture here shows it the way OpenNR sees it:\n"
              << "scaled to the white point, and clipped above it.\n";
        ok = f.good();
    }
    if (SUCCEEDED(co)) CoUninitialize();
    set_status(ok ? "saved to " + dir.string() : "could not write to " + dir.string());
}

}}  // namespace opennr::views
