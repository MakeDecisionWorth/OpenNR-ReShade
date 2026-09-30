// The network's weights, loaded onto the game's device through ReShade.
//
// A model is a folder: `model.txt` (the layer list) and one pair of files a layer, `.w.bin`
// (fp16 weights) and `.b.bin` (fp32 bias). The network is a small U-Net: four levels, 19 3x3
// convolutions, and optionally a whole-frame branch that adds to the middle block's bias.
//
// Copyright (c) 2026 MakeDecisionWorth. MIT licence, see LICENSE.

#pragma once

#include <reshade.hpp>

#include "opennr_shaders.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace opennr {

using namespace reshade::api;

// ------------------------------------------------------------------ buffers on Vulkan
//
// On D3D a structured buffer is bound through a view. On Vulkan it is a storage buffer, bound by
// the buffer itself, so there `Network::buffer` creates no view and hands out the buffer's handle
// in its place. This list tells such a stand-in from a real view: it is bound as a storage
// buffer, barriers go on the buffer, and it is never passed to `destroy_resource_view`. One entry
// a stand-in, so a buffer handed out as both "SRV" and "UAV" is listed twice. Empty on D3D.
inline std::vector<uint64_t>& buffer_stand_ins() {
    static std::vector<uint64_t> v;
    return v;
}
inline bool is_buffer_stand_in(resource_view v) {
    if (!v.handle) return false;
    for (uint64_t h : buffer_stand_ins())
        if (h == v.handle) return true;
    return false;
}
inline void destroy_view(device* d, resource_view& v) {
    if (!v.handle) return;
    std::vector<uint64_t>& s = buffer_stand_ins();
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i] == v.handle) {
            s[i] = s.back();
            s.pop_back();
            v = {};
            return;
        }
    d->destroy_resource_view(v);
    v = {};
}

struct Conv {
    std::string name;
    // `cout` is what the layer computes, `pad` how wide the buffer it writes is; they differ on
    // the output layer only, which is padded up to a multiple of 16
    uint32_t cin = 0, cout = 0, pad = 0, level = 0;
    resource w = {}, b = {};
    resource_view w_srv = {}, b_srv = {};
};

struct Network {
    uint32_t cin = 16, base = 16, levels = 4, cout = 4;
    std::vector<Conv> convs;
    bool ok = false;

    // the optional whole-frame branch
    bool glob = false;
    uint32_t g_in = 0, g_hidden = 0, g_out = 0;
    std::vector<Conv> gconvs;
    resource fc = {};                 // fc1 W, b1, fc2 W, b2, fp32, in the order kGlobalHLSL reads
    resource_view fc_srv = {};

    // A structured buffer of `count` elements of `stride` bytes: 8 for four packed halves, 16 for
    // a float4, 4 for a float. `step` names the call that failed.
    static bool buffer(device* d, uint64_t count, uint32_t stride, const void* init, resource* r,
                       resource_view* srv, resource_view* uav, const char** step = nullptr) {
        const char* dummy = nullptr;
        if (!step) step = &dummy;
        // on Vulkan every buffer is a storage buffer, read-only ones included
        const bool vk = d->get_api() == device_api::vulkan;
        // a copy source as well, so Save views can read the network's output back
        resource_desc rd(count * stride, memory_heap::gpu_only,
                         resource_usage::shader_resource | resource_usage::copy_source |
                             ((uav || vk) ? resource_usage::unordered_access
                                          : resource_usage::undefined));
        rd.buffer.structured.stride = stride;
        subresource_data sd{const_cast<void*>(init), 0, 0};
        *step = "create_resource";
        if (!d->create_resource(rd, init ? &sd : nullptr,
                                uav ? resource_usage::unordered_access
                                    : resource_usage::shader_resource, r))
            return false;
        if (vk) {
            if (srv) { *srv = resource_view{r->handle}; buffer_stand_ins().push_back(r->handle); }
            if (uav) { *uav = resource_view{r->handle}; buffer_stand_ins().push_back(r->handle); }
            *step = nullptr;
            return true;
        }
        resource_view_desc vd{};
        vd.type = resource_view_type::buffer;
        vd.format = format::unknown;
        vd.buffer.offset = 0;
        vd.buffer.structured.count = uint32_t(count);
        vd.buffer.structured.stride = stride;
        *step = "srv";
        if (srv && !d->create_resource_view(*r, resource_usage::shader_resource, vd, srv))
            return false;
        *step = "uav";
        if (uav && !d->create_resource_view(*r, resource_usage::unordered_access, vd, uav))
            return false;
        *step = nullptr;
        return true;
    }

    bool load(device* d, const std::string& dir, std::string& err) {
        std::FILE* f = std::fopen((dir + "/model.txt").c_str(), "r");
        if (!f) { err = "no model.txt in " + dir; return false; }
        int n = 0, ci = 0, bs = 0, lv = 0, co = 0;
        if (std::fscanf(f, "%d %d %d %d %d", &ci, &bs, &lv, &co, &n) != 5) {
            err = "bad header in model.txt";
            std::fclose(f);
            return false;
        }
        cin = ci; base = bs; levels = lv; cout = co;
        // one layer line and its two files, into `into`
        auto read_conv = [&](bool last, std::vector<Conv>& into) -> bool {
            char nm[64];
            int a = 0, b2 = 0, pad = 0, lvl = 0;
            if (std::fscanf(f, "%63s %d %d %d %d", nm, &a, &b2, &pad, &lvl) != 5) {
                err = "bad line in model.txt";
                return false;
            }
            if (!last && b2 != pad) {
                err = std::string(nm) + ": width " + std::to_string(b2) + " padded to " +
                      std::to_string(pad) + " -- this model is not supported";
                return false;
            }
            Conv c;
            c.name = nm;
            c.cin = uint32_t(a);
            c.cout = uint32_t(b2);
            c.pad = uint32_t(pad);
            c.level = uint32_t(lvl);
            auto read = [&](const char* suf, size_t bytes, std::vector<uint8_t>& into) {
                std::FILE* g = std::fopen((dir + "/" + c.name + suf).c_str(), "rb");
                if (!g) return false;
                into.resize(bytes);
                const bool okr = std::fread(into.data(), 1, bytes, g) == bytes;
                std::fclose(g);
                return okr;
            };
            std::vector<uint8_t> wv, bv;
            if (!read(".w.bin", size_t(9) * c.cin * c.pad * 2, wv) ||    // fp16
                !read(".b.bin", size_t(c.pad) * 4, bv)) {                // fp32
                err = c.name + ": missing or short weight files";
                return false;
            }
            const char* step = nullptr;
            char why[160];
            if (!buffer(d, size_t(9) * c.cin * c.pad / 4, 8, wv.data(),
                        &c.w, &c.w_srv, nullptr, &step)) {
                std::snprintf(why, sizeof why, "%s: weights, %s failed", c.name.c_str(), step);
                err = why;
                return false;
            }
            if (!buffer(d, size_t(c.pad) / 4, 16, bv.data(), &c.b, &c.b_srv, nullptr, &step)) {
                std::snprintf(why, sizeof why, "%s: bias, %s failed", c.name.c_str(), step);
                err = why;
                return false;
            }
            into.push_back(c);
            return true;
        };
        for (int i = 0; i < n; ++i)
            if (!read_conv(i + 1 == n, convs)) { std::fclose(f); return false; }

        // the optional whole-frame section: "global <n> <in> <hidden> <out>", n layers, fc.bin
        char tag[16] = {};
        int gn = 0, gi = 0, gh = 0, go = 0;
        if (std::fscanf(f, "%15s %d %d %d %d", tag, &gn, &gi, &gh, &go) == 5 &&
            std::strcmp(tag, "global") == 0) {
            for (int i = 0; i < gn; ++i)
                if (!read_conv(false, gconvs)) { std::fclose(f); return false; }
            const uint32_t mid = base << (levels - 1);
            if (gn < 1 || uint32_t(gi) != gconvs.back().pad || uint32_t(go) != mid ||
                gi > 64 || gh > 64 || go % 4) {
                err = "the model's whole-frame section does not fit its network";
                std::fclose(f);
                return false;
            }
            const size_t nf = size_t(gh) * gi + gh + size_t(go) * gh + go;
            std::vector<float> fv(nf);
            std::FILE* g = std::fopen((dir + "/fc.bin").c_str(), "rb");
            const bool okf = g && std::fread(fv.data(), 4, nf, g) == nf;
            if (g) std::fclose(g);
            if (!okf || !buffer(d, nf, 4, fv.data(), &fc, &fc_srv, nullptr)) {
                err = "fc.bin missing, short, or its buffer failed";
                std::fclose(f);
                return false;
            }
            g_in = uint32_t(gi); g_hidden = uint32_t(gh); g_out = uint32_t(go);
            glob = true;
        }
        std::fclose(f);
        ok = true;
        return true;
    }

    void destroy(device* d) {
        for (std::vector<Conv>* list : {&convs, &gconvs})
            for (Conv& c : *list) {
                for (resource_view* v : {&c.w_srv, &c.b_srv}) destroy_view(d, *v);
                for (resource* r : {&c.w, &c.b})
                    if (r->handle) { d->destroy_resource(*r); *r = {}; }
            }
        destroy_view(d, fc_srv);
        if (fc.handle) { d->destroy_resource(fc); fc = {}; }
        convs.clear();
        gconvs.clear();
        glob = false;
        ok = false;
    }

    // the widest intermediate, in halves, so the caller can size the ping-pong pair once
    uint64_t scratch_floats(uint32_t W, uint32_t H) const {
        uint64_t big = 0;
        for (const std::vector<Conv>* list : {&convs, &gconvs})
            for (const Conv& c : *list) {
                const uint64_t px = uint64_t(W >> c.level) * (H >> c.level);
                big = std::max(big, px * std::max(c.cin, c.pad));
            }
        // the whole-frame branch pools the 16 input channels down from the full extent
        if (glob) big = std::max(big, uint64_t(W / 2) * (H / 2) * cin);
        return big;
    }
};

}  // namespace opennr
