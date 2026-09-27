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
// the Model B/C colour grade, and the view.
inline const char* kCompositeHLSL = R"(
Texture2D<float4>          src  : register(t0);
StructuredBuffer<uint2>    head : register(t1);
RWTexture2D<float4>        dst  : register(u0);

// gHeadStride: how wide the network's output buffer is per pixel, in halves.
// gHeadGroup: which four channels of it to read -- the model has one output per Model A/B/C.
// gGrade: bits 0-1 the model (0 A, 1 B, 2 C), bits 8-15 Local Tone in percent, capped at 100.
cbuffer Args : register(b0) {
    uint gW, gH, gOW, gOH, gStrength, gHeadStride, gView, gHeadGroup, gGrade;
};

// ---- the Model B/C colour grade. Each model is 14 fields; a field is scaled from its default
// toward the model's value by Local Tone (up to 100%).
static const float kGradeField[3][14] = {
    {0.0, 1.0,  0.0, 0.0,  0.0,   0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},   // A
    {0.0, 1.0, -0.1, 0.0, -0.25, -0.1,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},   // B
    {0.0, 1.0,  0.0, 0.0,  0.0,  -0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};  // C

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

float3 fetch(uint x, uint y) {
    const uint2 v = head[(y * gW + x) * (gHeadStride / 4) + gHeadGroup];
    return float3(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gOW || id.y >= gOH) return;
    const float4 s = src.Load(int3(id.xy, 0));

    float3 h;
    if (gW == gOW && gH == gOH) {
        h = fetch(id.x, id.y);
    } else {
        const float fx = (float(id.x) + 0.5) * (float(gW) / float(gOW)) - 0.5;
        const float fy = (float(id.y) + 0.5) * (float(gH) / float(gOH)) - 0.5;
        const int x0 = int(floor(fx)), y0 = int(floor(fy));
        const float wx = fx - float(x0), wy = fy - float(y0);
        const uint xc = uint(clamp(x0, 0, int(gW) - 1)), x1 = uint(clamp(x0 + 1, 0, int(gW) - 1));
        const uint yc = uint(clamp(y0, 0, int(gH) - 1)), y1 = uint(clamp(y0 + 1, 0, int(gH) - 1));
        h = lerp(lerp(fetch(xc, yc), fetch(x1, yc), wx),
                 lerp(fetch(xc, y1), fetch(x1, y1), wx), wy);
    }
    // Model A: frame + correction. Model B/C: the corrected frame graded, blended back onto the
    // frame by Strength.
    float3 corr = 0.25 * h * (float(gStrength) / 100.0);
    const uint style = min(gGrade & 3u, 2u);
    if (style != 0) {
        const float tone = float(min((gGrade >> 8) & 0xffu, 100u)) / 100.0;
        corr = (grade(saturate(s.rgb + 0.25 * h), style, tone) - s.rgb) * (float(gStrength) / 100.0);
    }
    // gView: 0 the final frame; 1 the correction at 20x, centred on mid-grey (grey is unchanged,
    // brighter is added, darker is taken away); 2 the frame exactly as OpenNR reads it.
    if (gView == 2)      dst[id.xy] = float4(s.rgb, 1.0);
    else if (gView == 1) dst[id.xy] = float4(saturate(0.5 + 20.0 * corr), s.a);
    else                 dst[id.xy] = float4(saturate(s.rgb + corr), s.a);
}
)";

}  // namespace opennr
