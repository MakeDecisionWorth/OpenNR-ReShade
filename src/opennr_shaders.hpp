// OpenNR's compute shaders, as HLSL source.
//
// Compiled at load time with D3DCompile on D3D11 and D3D12. For Vulkan, build.ps1 compiles the
// same strings to SPIR-V ahead of time and embeds them (opennr_spirv.hpp).
//
// Copyright (c) 2026 MakeDecisionWorth. MIT licence, see LICENSE.

#pragma once

namespace opennr {

// One 3x3 convolution with its bias and activation. Activations and weights are fp16, four
// channels packed into a uint2; the accumulator is fp32.
inline const char* kConvHLSL = R"(
#define TM 4

StructuredBuffer<uint2>   x : register(t0);   // [pixel][cin/4]
StructuredBuffer<uint2>   w : register(t1);   // [tap][cin][cout/4]
StructuredBuffer<float4>  b : register(t2);   // [cout/4]
RWStructuredBuffer<uint2> o : register(u0);   // [pixel][cout/4]

// gZeroPad: 0 clamps the flattened pixel index to the buffer, 1 reads zero outside the image
cbuffer Args : register(b0) { uint gW, gH, gCin, gCout, gAct, gZeroPad; };

float4 h4(uint2 v) {
    return float4(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y), f16tof32(v.y >> 16));
}
uint2 p4(float4 v) {
    return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
}

// GELU, tanh form. The clamp keeps tanh's exponential from overflowing to Inf/Inf = NaN on large
// inputs; tanh(10) is already 1.0 in fp32, so it changes no representable value.
float4 gelu(float4 v) {
    const float4 t = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    return 0.5 * v * (1.0 + tanh(clamp(t, -10.0, 10.0)));
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint npix = gW * gH;
    const uint cin4 = gCin / 4, cout4 = gCout / 4;
    const uint p0 = (tid.x / cout4) * TM;
    const uint n4 = tid.x % cout4;
    if (p0 >= npix) return;

    float4 acc[TM];
    [unroll] for (int z = 0; z < TM; ++z) acc[z] = 0.0;

    [loop] for (uint tap = 0; tap < 9; ++tap) {
        const int dy = int(tap / 3) - 1, dx = int(tap % 3) - 1;
        uint qm[TM];
        bool ok[TM];
        [unroll] for (int m = 0; m < TM; ++m) {
            const int pm = int(p0) + m;
            if (gZeroPad != 0) {
                const int px = pm % int(gW) + dx, py = pm / int(gW) + dy;
                ok[m] = px >= 0 && py >= 0 && px < int(gW) && py < int(gH);
                qm[m] = ok[m] ? uint(py * int(gW) + px) : 0u;
            } else {
                ok[m] = true;
                qm[m] = uint(clamp(pm + dy * int(gW) + dx, 0, int(npix) - 1));
            }
        }
        [loop] for (uint k4 = 0; k4 < cin4; ++k4) {
            float4 xv[TM];
            [unroll] for (int m = 0; m < TM; ++m)
                xv[m] = ok[m] ? h4(x[qm[m] * cin4 + k4]) : float4(0.0, 0.0, 0.0, 0.0);
            [unroll] for (uint kk = 0; kk < 4; ++kk) {
                const float4 wv = h4(w[(tap * gCin + k4 * 4 + kk) * cout4 + n4]);
                [unroll] for (int a = 0; a < TM; ++a) acc[a] += xv[a][kk] * wv;
            }
        }
    }
    const float4 bv = b[n4];
    [unroll] for (int i = 0; i < TM; ++i) {
        const uint p = p0 + uint(i);
        if (p < npix) {
            const float4 v = acc[i] + bv;
            o[p * cout4 + n4] = p4(gAct != 0 ? gelu(v) : v);
        }
    }
}
)";

// 2x2 average pooling
inline const char* kPoolHLSL = R"(
StructuredBuffer<uint2>   x : register(t0);
RWStructuredBuffer<uint2> o : register(u0);
cbuffer Args : register(b0) { uint gW, gH, gCin, gCout, gAct; };

float4 h4(uint2 v) {
    return float4(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y), f16tof32(v.y >> 16));
}
uint2 p4(float4 v) {
    return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint cin4 = gCin / 4;
    const uint ow = gW / 2, oh = gH / 2;
    const uint n = ow * oh * cin4;
    if (tid.x >= n) return;
    const uint c = tid.x % cin4, p = tid.x / cin4;
    const uint y = (p / ow) * 2, xx = (p % ow) * 2;
    const uint a = (y * gW + xx) * cin4 + c;
    o[tid.x] = p4(0.25 * (h4(x[a]) + h4(x[a + cin4]) +
                          h4(x[a + gW * cin4]) + h4(x[a + gW * cin4 + cin4])));
}
)";

// Nearest-neighbour upsampling of the lower level, concatenated with the skip connection
inline const char* kUpCatHLSL = R"(
StructuredBuffer<uint2>   lo   : register(t0);
StructuredBuffer<uint2>   skip : register(t1);
RWStructuredBuffer<uint2> o    : register(u0);
cbuffer Args : register(b0) { uint gW, gH, gClo, gCs, gScale; };

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint clo4 = gClo / 4, cs4 = gCs / 4;
    const uint cout4 = clo4 + cs4;
    const uint n = gW * gH * cout4;
    if (tid.x >= n) return;
    const uint c = tid.x % cout4, p = tid.x / cout4;
    const uint y = p / gW, xx = p % gW;
    o[tid.x] = c < clo4
        ? lo[((y / gScale) * (gW / gScale) + xx / gScale) * clo4 + c]
        : skip[p * cs4 + (c - clo4)];
}
)";

// The whole-frame branch's tail: the spatial mean of its last feature map, two linear layers, and
// the result added to the middle block's bias.
inline const char* kGlobalHLSL = R"(
StructuredBuffer<uint2>    f  : register(t0);   // [pixel][gC/4]
StructuredBuffer<float>    p  : register(t1);   // fc1 W [gH][gC], b1 [gH], fc2 W [gO][gH], b2 [gO]
StructuredBuffer<float4>   mb : register(t2);   // the middle block's own bias, [gO/4]
RWStructuredBuffer<float4> o  : register(u0);   // its bias for this frame, [gO/4]
cbuffer Args : register(b0) { uint gN, gC, gH, gO; };

groupshared float m[64];
groupshared float hid[64];

float4 h4(uint2 v) {
    return float4(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y), f16tof32(v.y >> 16));
}
float gelu1(float v) {
    const float t = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    return 0.5 * v * (1.0 + tanh(clamp(t, -10.0, 10.0)));
}

[numthreads(128, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    const uint t = tid.x;
    if (t < gC) {
        float s = 0.0;
        const uint c4 = gC / 4;
        [loop] for (uint i = 0; i < gN; ++i) s += h4(f[i * c4 + t / 4])[t % 4];
        m[t] = s / float(gN);
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < gH) {
        float a = p[gH * gC + t];
        [loop] for (uint c = 0; c < gC; ++c) a += p[t * gC + c] * m[c];
        hid[t] = gelu1(a);
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < gO / 4) {
        const uint w2 = gH * gC + gH, b2 = w2 + gO * gH;
        float4 r = mb[t];
        [unroll] for (uint j = 0; j < 4; ++j) {
            const uint oo = t * 4 + j;
            float a = p[b2 + oo];
            [loop] for (uint i = 0; i < gH; ++i) a += p[w2 + oo * gH + i] * hid[i];
            r[j] += a;
        }
        o[t] = r;
    }
}
)";

// The network's 16 input channels at its own resolution: seeded noise, the frame's colour
// (sampled down when the network runs smaller than the frame), and the settings.
inline const char* kStageHLSL = R"(
Texture2D<float4>         src : register(t0);
RWStructuredBuffer<uint2> ch  : register(u0);

uint2 p4(float4 v) {
    return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
}

// gTone and gStructure: Local Tone Strength and Local Structure Strength as fp32 bits, the
// percentage divided by 100 on the CPU -- dividing here compiles to a multiplication by 0.01,
// which puts 100% at 0.99999998 and, after the fp16 store, at 0.9995 instead of 1.0
cbuffer Args : register(b0) { uint gW, gH, gOW, gOH, gFrame, gTone, gStructure; };

uint mix32(uint v) { v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16; return v; }
float u01(uint v) { return float(v & 0x00ffffffu) / 16777216.0 + 1.0 / 33554432.0; }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gW || id.y >= gH) return;
    const uint p = id.y * gW + id.x;

    const uint s = mix32(id.x * 0x9e3779b9u ^ id.y * 0x85ebca6bu ^ gFrame * 0xc2b2ae35u);
    const float u0 = u01(mix32(s + 1u)), u1 = u01(mix32(s + 2u));
    const float u2 = u01(mix32(s + 3u)), u3 = u01(mix32(s + 4u));
    const float r0 = sqrt(-2.0 * log(u0)), r1 = sqrt(-2.0 * log(u1));
    const float TAU = 6.283185307179586;

    float3 c;
    if (gW == gOW && gH == gOH) {
        c = src.Load(int3(id.xy, 0)).rgb;
    } else {
        const float fx = (float(id.x) + 0.5) * (float(gOW) / float(gW)) - 0.5;
        const float fy = (float(id.y) + 0.5) * (float(gOH) / float(gH)) - 0.5;
        const int x0 = int(floor(fx)), y0 = int(floor(fy));
        const float wx = fx - float(x0), wy = fy - float(y0);
        const int x1 = clamp(x0 + 1, 0, int(gOW) - 1), y1 = clamp(y0 + 1, 0, int(gOH) - 1);
        const int xc = clamp(x0, 0, int(gOW) - 1), yc = clamp(y0, 0, int(gOH) - 1);
        c = lerp(lerp(src.Load(int3(xc, yc, 0)).rgb, src.Load(int3(x1, yc, 0)).rgb, wx),
                 lerp(src.Load(int3(xc, y1, 0)).rgb, src.Load(int3(x1, y1, 0)).rgb, wx), wy);
    }
    const float3 v = (c - 0.5) * 2.0 * (1.0 / 16.0);

    ch[p * 4 + 0] = p4(float4(r1 * cos(TAU * u3), r1 * sin(TAU * u3), r0 * cos(TAU * u2), 1.0));
    ch[p * 4 + 1] = p4(float4(v.r, v.g, v.b, v.r));
    ch[p * 4 + 2] = p4(float4(v.g, v.b, 0.0, asfloat(gTone)));
    ch[p * 4 + 3] = p4(float4(asfloat(gStructure), -1.0, -1.0, 0.0));
}
)";

// The network's correction on the frame, lifted back to the output resolution, with Strength,
// the Natural/Cinematic colour grade, Detail and Colour strength, Compare, and the view -- on an
// SDR or an HDR swap chain.
inline const char* kCompositeHLSL = R"(
Texture2D<float4>          src  : register(t0);
StructuredBuffer<uint2>    head : register(t1);
RWTexture2D<float4>        dst  : register(u0);

// gHeadStride: how wide the network's output buffer is per pixel, in halves.
// gHeadGroup: which four channels of it to read -- the model has one output per Style.
// gGrade: bits 0-1 the style (0 Standard, 1 Natural, 2 Cinematic), bits 8-15 Local Tone in
//   percent, capped at 100.
// gDetail, gColour: Detail and Colour strength. gApply 0: the frame goes out as it came in.
// gCompare: 1 side by side (gZoom 1-2), 2 a split line at gSplit (0-1); gSwap puts the corrected
//   picture on the left.
// gHdr: the swap chain, 0 SDR, 1 scRGB, 2 HDR10; gWhite the white point (1.0 = 80 nits).
// gDetail, gColour, gSplit, gZoom and gWhite are fp32 bits.
cbuffer Args : register(b0) {
    uint gW, gH, gOW, gOH, gStrength, gHeadStride, gView, gHeadGroup, gGrade, gDetail, gColour,
         gApply, gCompare, gSplit, gZoom, gSwap, gHdr, gWhite, gPad0, gPad1;
};

// ---- the Natural/Cinematic colour grade. Each style is 14 fields; a field is scaled from its
// default toward the style's value by Local Tone (up to 100%).
static const float kGradeField[3][14] = {
    {0.0, 1.0,  0.0, 0.0,  0.0,   0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},   // Standard
    {0.0, 1.0, -0.1, 0.0, -0.25, -0.1,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},   // Natural
    {0.0, 1.0,  0.0, 0.0,  0.0,  -0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};  // Cinematic

float grade_smooth(float x) { return x * x * (3.0 - 2.0 * x); }

float grade_hue(float p, float q, float t) {
    if (t < 0.0) t += 1.0;
    if (t > 1.0) t -= 1.0;
    if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
    if (t < 0.5) return q;
    if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    return p;
}

float3 grade_rgb2hsl(float3 c) {
    const float mx = max(max(c.r, c.g), c.b), mn = min(min(c.r, c.g), c.b);
    const float L = (mx + mn) * 0.5;
    float H = 0.0, S = 0.0;
    if (mx > mn) {
        const float d = mx - mn;
        S = L > 0.5 ? d / (2.0 - mx - mn) : d / (mx + mn);
        if (mx == c.r)      H = ((c.g - c.b) / d + (c.g >= c.b ? 0.0 : 6.0)) / 6.0;
        else if (mx == c.g) H = ((c.b - c.r) / d + 2.0) / 6.0;
        else                H = ((c.r - c.g) / d + 4.0) / 6.0;
    }
    return float3(H, S, L);
}

float3 grade_hsl2rgb(float3 hsl) {
    const float H = hsl.x, S = hsl.y, L = hsl.z;
    if (!(S > 0.0)) return float3(L, L, L);
    const float q = L < 0.5 ? L * (1.0 + S) : L + S - L * S;
    const float p = 2.0 * L - q;
    return float3(grade_hue(p, q, H + 1.0 / 3.0), grade_hue(p, q, H), grade_hue(p, q, H - 1.0 / 3.0));
}

// the temperature and tint targets: a fully saturated colour at the pixel's own lightness
float3 grade_tint_target(float3 c, bool temperature, bool positive) {
    const float L = (max(max(c.r, c.g), c.b) + min(min(c.r, c.g), c.b)) * 0.5;
    const float q = L < 0.5 ? 2.0 * L : 1.0;
    const float p = 2.0 * L - q;
    if (temperature)
        return positive ? float3(q, p + (q - p) * 6.0 * 0.061110001057386398315, p)
                        : float3(p, p + (q - p) * 6.0 * 0.10555666685104370117, q);
    return positive ? float3(p + (q - p) * 6.0 * 0.022223353385925292969, q, p)
                    : float3(p + (q - p) * 6.0 * 0.14444339275360107422, p, q);
}

float3 grade(float3 n, uint style, float tone) {
    float f[14];
    [unroll] for (int i = 0; i < 14; ++i)
        f[i] = kGradeField[0][i] + (kGradeField[style][i] - kGradeField[0][i]) * tone;
    float3 c = saturate(n);
    c = saturate((c - f[0]) * (1.0 / (f[1] - f[0] + 1e-10)));                    // levels
    if (abs(f[7]) >= 1e-6)                                                         // temperature
        c = saturate(c + (grade_tint_target(c, true, f[7] > 0.0) - c) * abs(f[7]));
    if (abs(f[8]) >= 1e-6)                                                         // tint
        c = saturate(c + (grade_tint_target(c, false, f[8] > 0.0) - c) * abs(f[8]));
    c = saturate(c * exp2(f[2]));                                                  // exposure
    c = saturate(c + (c * c * (3.0 - 2.0 * c) - c) * f[4]);                        // contrast
    float3 g3;                                                                     // tone curve
    [unroll] for (int k = 0; k < 3; ++k) {
        const float v = c[k];
        const float b0 = grade_smooth(saturate(1.0 - 4.0 * v));
        const float b1 = grade_smooth(saturate(4.0 * v)) * grade_smooth(saturate(2.0 - 4.0 * v));
        const float b2 = grade_smooth(saturate(4.0 * v - 1.0)) * grade_smooth(saturate(3.0 - 4.0 * v));
        const float b3 = grade_smooth(saturate(4.0 * v - 2.0)) * grade_smooth(saturate(4.0 - 4.0 * v));
        const float b4 = grade_smooth(saturate(4.0 * v - 3.0));
        g3[k] = b0 * f[9] + b1 * f[10] + b2 * f[11] + b3 * f[12] + b4 * f[13];
    }
    c = exp2(exp2(-g3) * log2(c));
    c = exp2(exp2(-f[3]) * log2(max(c, 0.0)));                                     // gamma
    float3 hsl = grade_rgb2hsl(c);                                                 // saturation
    hsl.y = saturate(hsl.y * (1.0 + f[5]));
    c = grade_hsl2rgb(hsl);
    hsl = grade_rgb2hsl(c);                                                        // saturation curve
    hsl.y = saturate(exp2(exp2(-f[6]) * log2(hsl.y)));
    return saturate(grade_hsl2rgb(hsl));
}

// ---- HDR. The network works on a display picture: on SDR the frame itself, on HDR the frame in
// linear BT.709 (1.0 = 80 nits) over the white point, sRGB-encoded -- what kEncodeHLSL writes.
// Its correction goes back as a change in linear light, so a pixel it does not change is left
// exactly as the game drew it, highlights above the white point and wide-gamut colour included.
static const float3x3 kBt709To2020 = {0.6274040, 0.3292820, 0.0433136,
                                      0.0690970, 0.9195400, 0.0113612,
                                      0.0163916, 0.0880132, 0.8955950};
static const float3x3 kBt2020To709 = { 1.6604910, -0.5876411, -0.0728499,
                                      -0.1245505,  1.1328999, -0.0083494,
                                      -0.0181508, -0.1005789,  1.1187297};
float3 srgb_to_linear(float3 v) {
    v = saturate(v);
    return lerp(v / 12.92, pow((v + 0.055) / 1.055, 2.4), step(0.04045, v));
}
float3 srgb_encode(float3 v) {
    v = saturate(v);
    return lerp(v * 12.92, 1.055 * pow(v, 1.0 / 2.4) - 0.055, step(0.0031308, v));
}
float3 pq_decode(float3 e) {       // SMPTE ST 2084 to nits / 10000
    const float3 p = pow(saturate(e), 1.0 / 78.84375);
    return pow(max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}
float3 pq_encode(float3 y) {
    const float3 p = pow(saturate(y), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p), 78.84375);
}
float3 hdr_to_linear(float3 v) { return gHdr == 2 ? mul(kBt2020To709, pq_decode(v) * 125.0) : v; }
float3 linear_to_hdr(float3 x) { return gHdr == 2 ? pq_encode(mul(kBt709To2020, x) * 0.008) : x; }
float3 to_display(float3 v) {
    return gHdr == 0 ? v : srgb_encode(saturate(hdr_to_linear(v) / asfloat(gWhite)));
}
// a display value out to the swap chain; on HDR at the white point
float4 emit(float3 v, float a) {
    return gHdr == 0 ? float4(v, a) : float4(linear_to_hdr(asfloat(gWhite) * srgb_to_linear(v)), a);
}
// the corrected pixel: `raw` as the game drew it, `sd` its display value, `corr` the correction
float4 finish(float4 raw, float3 sd, float3 corr) {
    if (gHdr == 0) return float4(saturate(sd + corr), raw.a);
    return float4(linear_to_hdr(hdr_to_linear(raw.rgb) + asfloat(gWhite) *
                                (srgb_to_linear(saturate(sd + corr)) - srgb_to_linear(sd))), raw.a);
}

float3 fetch(uint x, uint y) {
    const uint2 v = head[(y * gW + x) * (gHeadStride / 4) + gHeadGroup];
    return float3(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y));
}

// the network's output at an output position (pixel centres at +0.5), bilinear when the network
// runs smaller than the frame
float3 head_at(float2 pos) {
    if (gW == gOW && gH == gOH) return fetch(min(uint(pos.x), gW - 1), min(uint(pos.y), gH - 1));
    const float fx = pos.x * (float(gW) / float(gOW)) - 0.5;
    const float fy = pos.y * (float(gH) / float(gOH)) - 0.5;
    const int x0 = int(floor(fx)), y0 = int(floor(fy));
    const float wx = fx - float(x0), wy = fy - float(y0);
    const uint xc = uint(clamp(x0, 0, int(gW) - 1)), x1 = uint(clamp(x0 + 1, 0, int(gW) - 1));
    const uint yc = uint(clamp(y0, 0, int(gH) - 1)), y1 = uint(clamp(y0 + 1, 0, int(gH) - 1));
    return lerp(lerp(fetch(xc, yc), fetch(x1, yc), wx), lerp(fetch(xc, y1), fetch(x1, y1), wx), wy);
}

// the frame at an output position, bilinear (the side-by-side halves read between pixels)
float4 src_at(float2 pos) {
    const float fx = pos.x - 0.5, fy = pos.y - 0.5;
    const int x0 = int(floor(fx)), y0 = int(floor(fy));
    const float wx = fx - float(x0), wy = fy - float(y0);
    const int xc = clamp(x0, 0, int(gOW) - 1), x1 = clamp(x0 + 1, 0, int(gOW) - 1);
    const int yc = clamp(y0, 0, int(gOH) - 1), y1 = clamp(y0 + 1, 0, int(gOH) - 1);
    return lerp(lerp(src.Load(int3(xc, yc, 0)), src.Load(int3(x1, yc, 0)), wx),
                lerp(src.Load(int3(xc, y1, 0)), src.Load(int3(x1, y1, 0)), wx), wy);
}

// Standard: frame + correction. Natural/Cinematic: the corrected frame graded, blended back onto
// the frame by Strength. Then Detail scales all of it and Colour the part that is not luminance.
float3 correction(float3 s, float3 h) {
    float3 corr = 0.25 * h * (float(gStrength) / 100.0);
    const uint style = min(gGrade & 3u, 2u);
    if (style != 0) {
        const float tone = float(min((gGrade >> 8) & 0xffu, 100u)) / 100.0;
        corr = (grade(saturate(s + 0.25 * h), style, tone) - s) * (float(gStrength) / 100.0);
    }
    const float Y = dot(corr, float3(0.2126, 0.7152, 0.0722));
    return asfloat(gDetail) * (Y + asfloat(gColour) * (corr - Y));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gOW || id.y >= gOH) return;
    const float4 raw = src.Load(int3(id.xy, 0));
    const float4 s = float4(to_display(raw.rgb), raw.a);

    // Compare, on the final frame: the picture cut at a split line, or both side by side, each
    // the whole frame at half size (letterboxed at zoom 1, filling its half at zoom 2)
    if (gView == 0 && gCompare != 0) {
        const float2 uv = (float2(id.xy) + 0.5) / float2(gOW, gOH);
        const float edge = gCompare == 2 ? asfloat(gSplit) : 0.5;
        if (abs(uv.x - edge) < 1.0 / float(gOW)) { dst[id.xy] = emit(float3(0.5, 0.5, 0.5), s.a); return; }
        const bool left = uv.x < edge;
        float2 pos = float2(id.xy) + 0.5;
        float4 r = raw;
        float3 sd = s.rgb;
        if (gCompare == 1) {
            const float2 hv = float2(left ? uv.x * 2.0 : uv.x * 2.0 - 1.0, uv.y) - 0.5;
            const float zoom = max(asfloat(gZoom), 1.0);
            const float2 cuv = 0.5 + float2(hv.x, hv.y * 2.0) / zoom;
            if (any(cuv < 0.0) || any(cuv > 1.0)) { dst[id.xy] = emit(float3(0.0, 0.0, 0.0), s.a); return; }
            pos = cuv * float2(gOW, gOH);
            r = float4(src_at(pos).rgb, s.a);
            sd = to_display(r.rgb);
        }
        dst[id.xy] = (left != (gSwap != 0)) || gApply == 0 ? r : finish(r, sd, correction(sd, head_at(pos)));
        return;
    }
    // gView: 0 the final frame; 1 the correction at 20x, centred on mid-grey (grey is unchanged,
    // brighter is added, darker is taken away); 2 the frame exactly as OpenNR reads it.
    const float3 corr = correction(s.rgb, head_at(float2(id.xy) + 0.5));
    if (gView == 2)      dst[id.xy] = emit(s.rgb, 1.0);
    else if (gView == 1) dst[id.xy] = emit(saturate(0.5 + 20.0 * corr), s.a);
    else                 dst[id.xy] = gApply != 0 ? finish(raw, s.rgb, corr) : raw;
}
)";

// HDR swap chains only: the frame as the network is shown it, once a frame -- linear BT.709
// (1.0 = 80 nits) over the white point, sRGB-encoded. The same conversion as the composite's
// to_display.
inline const char* kEncodeHLSL = R"(
Texture2D<float4>   src : register(t0);
RWTexture2D<float4> dst : register(u0);
cbuffer Args : register(b0) { uint gOW, gOH, gHdr, gWhite; };

static const float3x3 kBt2020To709 = { 1.6604910, -0.5876411, -0.0728499,
                                      -0.1245505,  1.1328999, -0.0083494,
                                      -0.0181508, -0.1005789,  1.1187297};
float3 pq_decode(float3 e) {
    const float3 p = pow(saturate(e), 1.0 / 78.84375);
    return pow(max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}
float3 srgb_encode(float3 v) {
    v = saturate(v);
    return lerp(v * 12.92, 1.055 * pow(v, 1.0 / 2.4) - 0.055, step(0.0031308, v));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gOW || id.y >= gOH) return;
    const float3 v = src.Load(int3(id.xy, 0)).rgb;
    const float3 x = gHdr == 2 ? mul(kBt2020To709, pq_decode(v) * 125.0) : v;
    dst[id.xy] = float4(srgb_encode(saturate(x / asfloat(gWhite))), 1.0);
}
)";

}  // namespace opennr
