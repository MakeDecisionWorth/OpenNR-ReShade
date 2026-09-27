// The network on a D3D11 device of its own, for "Run the network on" another GPU.
//
// ReShade's device API reaches only the device the game created, so when the player picks a
// different GPU the add-on creates a device on that adapter and runs the same shaders there,
// dispatch for dispatch as the in-frame path runs them.
//
// Copyright (c) 2026 MakeDecisionWorth. MIT licence, see LICENSE.

#pragma once

#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "opennr_shaders.hpp"

namespace opennr {

// The output packed to four channels -- one uint2 of halves a pixel -- which is what travels back
// to the game's GPU. gGroup picks which four: the model has one output per Model A/B/C.
inline const char* kPackHLSL = R"(
StructuredBuffer<uint2>   h : register(t0);   // [pixel][pad/4]
RWStructuredBuffer<uint2> o : register(u0);   // [pixel]
cbuffer Args : register(b0) { uint gN, gStride4, gGroup; };

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < gN) o[tid.x] = h[tid.x * gStride4 + gGroup];
}
)";

namespace d3d {

struct Buf {
    ID3D11Buffer* res = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    UINT elems = 0, stride = 0;
    void release() {
        if (uav) { uav->Release(); uav = nullptr; }
        if (srv) { srv->Release(); srv = nullptr; }
        if (res) { res->Release(); res = nullptr; }
        elems = stride = 0;
    }
};

struct Conv {
    std::string name;
    UINT cin = 0, cout = 0, pad = 0, level = 0;
    Buf w, b;
};

class Net {
public:
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;

    // the model, read as opennr::Network::load reads it
    int cin0 = 16, base = 16, levels = 4, cout0 = 4;
    std::vector<Conv> convs, gconvs;
    bool glob = false;
    int gin = 0, ghid = 0, gout = 0;
    Buf fc, gbias;

    ID3D11ComputeShader *cs_stage = nullptr, *cs_conv = nullptr, *cs_pool = nullptr,
                        *cs_up = nullptr, *cs_glob = nullptr, *cs_pack = nullptr;

    UINT W = 0, H = 0;                 // the network resolution the buffers are sized for
    Buf chans, A, B, head, packed, skip[8];
    std::string err;

    bool make(Buf& b, UINT elems, UINT stride, const void* init, bool uav) {
        b.release();
        b.elems = elems;
        b.stride = stride;
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = elems * stride;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE | (uav ? D3D11_BIND_UNORDERED_ACCESS : 0);
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = stride;
        D3D11_SUBRESOURCE_DATA sd{init, 0, 0};
        if (FAILED(dev->CreateBuffer(&bd, init ? &sd : nullptr, &b.res))) return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC vd{};
        vd.Format = DXGI_FORMAT_UNKNOWN;
        vd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        vd.Buffer.NumElements = elems;
        if (FAILED(dev->CreateShaderResourceView(b.res, &vd, &b.srv))) return false;
        if (uav) {
            D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
            ud.Format = DXGI_FORMAT_UNKNOWN;
            ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            ud.Buffer.NumElements = elems;
            if (FAILED(dev->CreateUnorderedAccessView(b.res, &ud, &b.uav))) return false;
        }
        return true;
    }

    ID3D11ComputeShader* compile(const char* src, const char* name) {
        ID3DBlob *code = nullptr, *e = nullptr;
        if (FAILED(D3DCompile(src, std::strlen(src), name, nullptr, nullptr, "main", "cs_5_0", 0, 0,
                              &code, &e))) {
            err = std::string(name) + ": " + (e ? (const char*)e->GetBufferPointer() : "?");
            if (e) e->Release();
            return nullptr;
        }
        ID3D11ComputeShader* cs = nullptr;
        dev->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &cs);
        code->Release();
        if (e) e->Release();
        return cs;
    }

    void dispatch(ID3D11ComputeShader* cs, ID3D11ShaderResourceView* t0,
                  ID3D11ShaderResourceView* t1, ID3D11ShaderResourceView* t2,
                  ID3D11UnorderedAccessView* u0, const UINT (&args)[12], UINT gx, UINT gy) {
        // unbind first: D3D11 drops an SRV whose resource is still bound as the previous
        // dispatch's UAV, and the read would return zeros
        ID3D11UnorderedAccessView* nul_u = nullptr;
        ctx->CSSetUnorderedAccessViews(0, 1, &nul_u, nullptr);
        ID3D11ShaderResourceView* nul_s[3] = {nullptr, nullptr, nullptr};
        ctx->CSSetShaderResources(0, 3, nul_s);

        ctx->CSSetShader(cs, nullptr, 0);
        ctx->CSSetUnorderedAccessViews(0, 1, &u0, nullptr);
        ID3D11ShaderResourceView* srvs[3] = {t0, t1, t2};
        ctx->CSSetShaderResources(0, 3, srvs);

        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = sizeof(args);
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA sd{args, 0, 0};
        ID3D11Buffer* cb = nullptr;
        dev->CreateBuffer(&bd, &sd, &cb);
        ctx->CSSetConstantBuffers(0, 1, &cb);
        ctx->Dispatch(gx, gy, 1);
        cb->Release();
    }

    // model.txt, the weights beside it, and the shaders
    bool load(const std::string& dir) {
        std::FILE* f = std::fopen((dir + "/model.txt").c_str(), "r");
        if (!f) { err = "no " + dir + "/model.txt"; return false; }
        int n = 0;
        if (std::fscanf(f, "%d %d %d %d %d", &cin0, &base, &levels, &cout0, &n) != 5) {
            err = "bad header in model.txt";
            std::fclose(f);
            return false;
        }
        auto read_conv = [&](std::vector<Conv>& into) -> bool {
            char nm[64];
            int a = 0, b2 = 0, pad = 0, lvl = 0;
            if (std::fscanf(f, "%63s %d %d %d %d", nm, &a, &b2, &pad, &lvl) != 5) {
                err = "bad line in model.txt";
                return false;
            }
            Conv c;
            c.name = nm; c.cin = a; c.cout = b2; c.pad = pad; c.level = lvl;
            std::vector<uint8_t> wv(size_t(9) * c.cin * c.pad * 2), bv(size_t(c.pad) * 4);
            auto slurp = [&](const char* suf, std::vector<uint8_t>& v) {
                std::FILE* g = std::fopen((dir + "/" + c.name + suf).c_str(), "rb");
                const bool ok = g && std::fread(v.data(), 1, v.size(), g) == v.size();
                if (g) std::fclose(g);
                return ok;
            };
            if (!slurp(".w.bin", wv) || !slurp(".b.bin", bv)) {
                err = c.name + ": missing or short weight files";
                return false;
            }
            if (!make(c.w, UINT(9 * c.cin * c.pad / 4), 8, wv.data(), false) ||
                !make(c.b, UINT(c.pad / 4), 16, bv.data(), false)) {
                err = c.name + ": buffer creation failed";
                return false;
            }
            into.push_back(std::move(c));
            return true;
        };
        for (int i = 0; i < n; ++i)
            if (!read_conv(convs)) { std::fclose(f); return false; }
        char tag[16] = {};
        int gn = 0;
        glob = std::fscanf(f, "%15s %d %d %d %d", tag, &gn, &gin, &ghid, &gout) == 5 &&
               std::string(tag) == "global";
        if (glob) {
            for (int i = 0; i < gn; ++i)
                if (!read_conv(gconvs)) { std::fclose(f); return false; }
            const size_t nf = size_t(ghid) * gin + ghid + size_t(gout) * ghid + gout;
            std::vector<float> fv(nf);
            std::FILE* g = std::fopen((dir + "/fc.bin").c_str(), "rb");
            const bool ok = g && std::fread(fv.data(), 4, nf, g) == nf;
            if (g) std::fclose(g);
            if (!ok || !make(fc, UINT(nf), 4, fv.data(), false) ||
                !make(gbias, UINT(gout / 4), 16, nullptr, true)) {
                err = "fc.bin missing, short, or its buffers failed";
                std::fclose(f);
                return false;
            }
        }
        std::fclose(f);
        cs_stage = compile(kStageHLSL, "stage");
        cs_conv = compile(kConvHLSL, "conv");
        cs_pool = compile(kPoolHLSL, "pool");
        cs_up = compile(kUpCatHLSL, "upcat");
        cs_glob = compile(kGlobalHLSL, "global");
        cs_pack = compile(kPackHLSL, "pack");
        return cs_stage && cs_conv && cs_pool && cs_up && cs_glob && cs_pack;
    }

    // buffers for a w x h network resolution
    bool alloc(UINT w, UINT h) {
        W = w;
        H = h;
        const UINT npix = W * H;
        UINT big = 0;
        for (const std::vector<Conv>* list : {&convs, &gconvs})
            for (const Conv& c : *list)
                big = std::max<UINT>(big, (W >> c.level) * (H >> c.level) * std::max(c.cin, c.pad));
        if (glob) big = std::max<UINT>(big, (W / 2) * (H / 2) * 16);
        bool ok = make(chans, npix * 16 / 4, 8, nullptr, true) &&
                  make(A, big / 4, 8, nullptr, true) && make(B, big / 4, 8, nullptr, true) &&
                  make(head, npix * convs.back().pad / 4, 8, nullptr, true) &&
                  make(packed, npix, 8, nullptr, true);
        for (int i = 0; ok && i < levels; ++i)
            ok = make(skip[i], (W >> i) * (H >> i) * (base << i) / 4, 8, nullptr, true);
        if (!ok) err = "buffer allocation failed";
        return ok;
    }

    // the frame at the output resolution -> the 16 input channels at W x H
    void stage(ID3D11ShaderResourceView* src, const UINT (&stage_args)[12]) {
        dispatch(cs_stage, src, nullptr, nullptr, chans.uav, stage_args, (W + 7) / 8, (H + 7) / 8);
    }

    // the network, input channels -> output
    void run() {
        Buf* cur = &chans;
        Buf* a = &A;
        Buf* b = &B;
        UINT k = 0, ww = W, hh = H;
        auto conv = [&](const Conv& c, Buf* in, Buf* o, UINT W2, UINT H2, bool act,
                        bool zero_pad = false, ID3D11ShaderResourceView* bias = nullptr) {
            const UINT args[12] = {W2, H2, c.cin, c.pad, act ? 1u : 0u, zero_pad ? 1u : 0u,
                                   0, 0, 0, 0, 0, 0};
            const UINT threads = (W2 * H2 / 4) * (c.pad / 4);
            dispatch(cs_conv, in->srv, c.w.srv, bias ? bias : c.b.srv, o->uav, args,
                     (threads + 63) / 64, 1);
        };
        auto pool = [&](Buf* in, Buf* o, UINT W2, UINT H2, UINT C) {
            const UINT pa[12] = {W2, H2, C, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            const UINT nn = (W2 / 2) * (H2 / 2) * (C / 4);
            dispatch(cs_pool, in->srv, nullptr, nullptr, o->uav, pa, (nn + 63) / 64, 1);
        };
        if (glob) {                       // the whole-frame branch
            UINT gw = W, gh = H;
            for (int s = 0; s < 3; ++s) {
                pool(cur, b, gw, gh, 16);
                std::swap(a, b); cur = a; gw /= 2; gh /= 2;
            }
            for (size_t s = 0; s < gconvs.size(); ++s) {
                const Conv& c = gconvs[s];
                conv(c, cur, b, gw, gh, true, true);
                std::swap(a, b); cur = a;
                if (s + 1 < gconvs.size()) {
                    pool(cur, b, gw, gh, c.pad);
                    std::swap(a, b); cur = a; gw /= 2; gh /= 2;
                }
            }
            const UINT ga[12] = {gw * gh, UINT(gin), UINT(ghid), UINT(gout), 0, 0, 0, 0, 0, 0, 0, 0};
            dispatch(cs_glob, cur->srv, fc.srv, convs[2 * levels].b.srv, gbias.uav, ga, 1, 1);
            cur = &chans;
        }
        for (int i = 0; i < levels; ++i) {
            for (int j = 0; j < 2; ++j) {
                const Conv& c = convs[k++];
                const bool last = (j == 1);
                conv(c, cur, last ? &skip[i] : b, ww, hh, true);
                if (last) cur = &skip[i];
                else { std::swap(a, b); cur = a; }
            }
            if (i + 1 < levels) {
                pool(&skip[i], b, ww, hh, UINT(base << i));
                std::swap(a, b); cur = a; ww /= 2; hh /= 2;
            }
        }
        for (int j = 0; j < 2; ++j) {
            const Conv& c = convs[k++];
            conv(c, cur, b, ww, hh, true, false, (j == 0 && glob) ? gbias.srv : nullptr);
            std::swap(a, b); cur = a;
        }
        for (int i = levels - 1; i >= 0; --i) {
            const UINT sw = W >> i, sh = H >> i;
            const Conv& c0 = convs[k];
            const UINT cs2 = base << i, clo = c0.cin - cs2;
            const UINT scale = (i + 1 < levels) ? 2u : 1u;
            const UINT ua[12] = {sw, sh, clo, cs2, scale, 0, 0, 0, 0, 0, 0, 0};
            const UINT nn = sw * sh * (c0.cin / 4);
            dispatch(cs_up, cur->srv, skip[i].srv, nullptr, b->uav, ua, (nn + 63) / 64, 1);
            std::swap(a, b); cur = a; ww = sw; hh = sh;
            for (int j = 0; j < 2; ++j) {
                const Conv& c = convs[k++];
                conv(c, cur, b, ww, hh, true);
                std::swap(a, b); cur = a;
            }
        }
        conv(convs[k], cur, &head, W, H, false);
    }

    // head -> packed: output `group`'s four channels, one uint2 a pixel
    void pack(UINT group = 0) {
        const UINT pa[12] = {W * H, convs.back().pad / 4, group, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        dispatch(cs_pack, head.srv, nullptr, nullptr, packed.uav, pa, (W * H + 63) / 64, 1);
    }

    void release() {
        for (std::vector<Conv>* list : {&convs, &gconvs})
            for (Conv& c : *list) { c.w.release(); c.b.release(); }
        convs.clear();
        gconvs.clear();
        for (Buf* bb : {&fc, &gbias, &chans, &A, &B, &head, &packed}) bb->release();
        for (Buf& bb : skip) bb.release();
        for (ID3D11ComputeShader** cs : {&cs_stage, &cs_conv, &cs_pool, &cs_up, &cs_glob, &cs_pack})
            if (*cs) { (*cs)->Release(); *cs = nullptr; }
        glob = false;
    }
};

}  // namespace d3d
}  // namespace opennr
