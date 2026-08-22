#include "ESKVCache.h"
#include "ESDecodeAttn.h"

#include <array>
#include <stdexcept>

namespace es {

ESKVCache::ESKVCache(int numLayers)
    : k_(numLayers), v_(numLayers), slots_(numLayers),
      kq_(numLayers), ks_(numLayers), kb_(numLayers),
      vq_(numLayers), vs_(numLayers), vb_(numLayers), qslots_(numLayers),
      rq8slots_(numLayers) {}

static int roundUpChunk(int n) {
    return ((n + ESKVCache::kGrowChunk - 1) / ESKVCache::kGrowChunk) * ESKVCache::kGrowChunk;
}

std::pair<mx::array, mx::array> ESKVCache::update(int layer, const mx::array & kNew, const mx::array & vNew,
                                                  int maxKeep, bool prealloc) {
    if (stepMode_) {
        // Compiled-step append: scatter the new position into the fixed-capacity buffer at a
        // position carried as DATA (an int32 [1] array input of the compiled graph), so one
        // recorded graph serves every token. Returns the full-capacity buffers; the step's
        // additive mask kills the unwritten/expired slots (their softmax weight is exactly 0).
        Slot & s = slots_[layer];
        const int kvH = kNew.shape(0), hd = kNew.shape(2);
        const mx::array & idx = (maxKeep > 0) ? *stepSlidingIdx_ : *stepGlobalIdx_;
        mx::array kUpd = mx::reshape(kNew, {1, kvH, 1, hd});  // scatter updates: [nIdx, kvH, 1, hd]
        mx::array vUpd = mx::reshape(vNew, {1, kvH, 1, hd});
        s.k = mx::scatter(*s.k, idx, kUpd, /*axis=*/1);
        s.v = mx::scatter(*s.v, idx, vUpd, /*axis=*/1);
        return {*s.k, *s.v};
    }
    if (!prealloc) {
        // ── Legacy mode: concat-grow (+ slice eviction). Kept as the bit-exact reference and
        // for the --cache-verify A/B; the prealloc mode below is the default runtime path.
        if (!k_[layer].has_value()) {
            k_[layer] = kNew;
            v_[layer] = vNew;
        } else {
            k_[layer] = mx::concatenate({*k_[layer], kNew}, 1);  // append along seq axis
            v_[layer] = mx::concatenate({*v_[layer], vNew}, 1);
        }
        // Sliding-window eviction: keep only the last `maxKeep` keys along the seq axis. The caller
        // only requests this for sliding layers on a single-token (decode) append, where the dropped
        // keys were all masked to -1e30 (softmax weight == 0) — so retaining the tail is bit-exact.
        if (maxKeep > 0 && k_[layer]->shape(1) > maxKeep) {
            const int len = k_[layer]->shape(1), hd = k_[layer]->shape(2), nh = k_[layer]->shape(0);
            k_[layer] = mx::slice(*k_[layer], {0, len - maxKeep, 0}, {nh, len, hd});
            const int vhd = v_[layer]->shape(2), vnh = v_[layer]->shape(0), vlen = v_[layer]->shape(1);
            v_[layer] = mx::slice(*v_[layer], {0, vlen - maxKeep, 0}, {vnh, vlen, vhd});
        }
        return {*k_[layer], *v_[layer]};
    }

    // Mode-mismatch guard: raw content (pre-norm, k-only or q8) cannot back the composite
    // path — the K/V it needs are post-norm. Restore with the matching RawMode or re-prime.
    if (rq8slots_[layer].q.has_value() ||
        (slots_[layer].k.has_value() && !slots_[layer].v.has_value()))
        throw std::runtime_error("ESKVCache::update: slot holds raw-K content — not usable "
                                 "by the composite path; re-prime the cache");

    // ── Prealloc mode: in-place slice_update append into a chunk-grown buffer; the live range
    // [start, len) is what the legacy mode would have stored, and the returned views slice it.
    Slot & s = slots_[layer];
    const int kvH = kNew.shape(0), nNew = kNew.shape(1), hd = kNew.shape(2);

    if (!s.k.has_value()) {
        // First append (prefill): store directly, capacity == content — identical to legacy's
        // first store. The first capacity growth below re-homes it into a chunked buffer.
        s.k = kNew; s.v = vNew;
        s.len = nNew; s.start = 0;
    } else {
        const int cap = s.k->shape(1);
        if (s.len + nNew > cap) {
            // Compact the live range to the front of a fresh chunk-rounded buffer. This is the
            // only copy in this mode, and it runs once per ~kGrowChunk tokens (sliding layers:
            // content is capped at the window, so the buffer size repeats and MLX's buffer
            // cache recycles it; global layers: capacity steps up by kGrowChunk).
            const int content = s.len - s.start;
            const int newCap = roundUpChunk(content + nNew + kGrowChunk);
            mx::array nk = mx::zeros({kvH, newCap, hd}, kNew.dtype());
            mx::array nv = mx::zeros({kvH, newCap, hd}, vNew.dtype());
            if (content > 0) {
                nk = mx::slice_update(nk, mx::slice(*s.k, {0, s.start, 0}, {kvH, s.len, hd}),
                                      {0, 0, 0}, {kvH, content, hd});
                nv = mx::slice_update(nv, mx::slice(*s.v, {0, s.start, 0}, {kvH, s.len, hd}),
                                      {0, 0, 0}, {kvH, content, hd});
            }
            s.k = nk; s.v = nv;
            if (s.start > 0) s.lost = true;   // rows [0, start) discarded — see Slot::lost
            s.len = content; s.start = 0;
        }
        // In-place append: the stored buffer is the only live reference by eval time (the views
        // returned last step died with that step's graph), so slice_update donates — no copy.
        s.k = mx::slice_update(*s.k, kNew, {0, s.len, 0}, {kvH, s.len + nNew, hd});
        s.v = mx::slice_update(*s.v, vNew, {0, s.len, 0}, {kvH, s.len + nNew, hd});
        s.len += nNew;
    }

    // Sliding-window eviction: advance the logical start instead of trimming storage (the
    // dropped keys were masked to -1e30 — softmax weight exactly 0 — so this is bit-exact).
    if (maxKeep > 0 && s.len - s.start > maxKeep) s.start = s.len - maxKeep;

    if (s.start == 0 && s.len == s.k->shape(1)) return {*s.k, *s.v};  // full buffer, no slice needed
    mx::array K = mx::slice(*s.k, {0, s.start, 0}, {kvH, s.len, hd});
    mx::array V = mx::slice(*s.v, {0, s.start, 0}, {kvH, s.len, hd});
    return {K, V};
}

std::pair<mx::array, mx::array> ESKVCache::current(int layer, bool prealloc) const {
    if (!prealloc) return {*k_[layer], *v_[layer]};
    const Slot & s = slots_[layer];
    const int kvH = s.k->shape(0), hd = s.k->shape(2);
    if (s.start == 0 && s.len == s.k->shape(1)) return {*s.k, *s.v};
    return {mx::slice(*s.k, {0, s.start, 0}, {kvH, s.len, hd}),
            mx::slice(*s.v, {0, s.start, 0}, {kvH, s.len, hd})};
}

static const char * kQuantNames[6] = {"kq_", "ks_", "kb_", "vq_", "vs_", "vb_"};

void ESKVCache::markPrefix(int pos, const std::string & fingerprint) {
    // Markers below the sliding window + prefill chunk are ambiguous (an evicting layer
    // still holds its full prefix there and cannot be told apart from a never-evicting
    // one). Every real marker (persona boundary, depth waypoints) sits far above this.
    if (pos < 4096)
        throw std::invalid_argument("ESKVCache::markPrefix: marker positions below 4096 unsupported");
    Marker m; m.pos = pos; m.fp = fingerprint;
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot & s = slots_[i];
        if (!s.k.has_value() || !s.v.has_value()) continue;   // raw/q8 layers: prefix-sliceable
        const int live = s.len - s.start;
        if (live >= pos) continue;                            // never evicted: prefix-sliceable
        const int kvH = s.k->shape(0), hd = s.k->shape(2);
        // Deep-copy the live windows NOW — later appends mutate these buffers in place.
        mx::array k = mx::contiguous(mx::slice(*s.k, {0, s.start, 0}, {kvH, s.len, hd}));
        mx::array v = mx::contiguous(mx::slice(*s.v, {0, s.start, 0}, {kvH, s.len, hd}));
        mx::eval(k, v);
        m.slidingKV.push_back({(int) i, {k, v}});
    }
    markers_.push_back(std::move(m));
}

bool ESKVCache::markRewindPoint(int pos) {
    // Same copy rule as markPrefix (live < pos: the layer evicts, and its pre-window rows
    // will not survive a later compaction), but into the single non-persisted mark. Layers
    // whose buffer still holds the full absolute prefix are left to cursor truncation,
    // guarded by Slot::lost at rewind time — so this works at any position, with no 4096
    // floor: a layer that starts evicting after the mark either keeps its prefix rows
    // (truncation stays valid) or trips `lost` and rewindToMark reports -1.
    if (pos <= 0) return false;
    for (const auto & a : k_) if (a.has_value()) return false;   // legacy storage: no rewind
    bool prealloc = false;
    for (const Slot & s : slots_) if (s.k.has_value()) { prealloc = true; break; }
    if (!prealloc) for (const RQ8Slot & s : rq8slots_) if (s.q.has_value()) { prealloc = true; break; }
    if (!prealloc) for (const QSlot & s : qslots_) if (s.t[0].has_value()) { prealloc = true; break; }
    if (!prealloc) return false;
    Marker m; m.pos = pos;
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot & s = slots_[i];
        if (!s.k.has_value() || !s.v.has_value()) continue;   // raw/q8/quant: prefix-sliceable
        const int live = s.len - s.start;
        if (live >= pos && !s.lost) continue;                 // full prefix present: truncate later
        const int kvH = s.k->shape(0), hd = s.k->shape(2);
        // Deep-copy the live window NOW — later appends mutate these buffers in place.
        mx::array k = mx::contiguous(mx::slice(*s.k, {0, s.start, 0}, {kvH, s.len, hd}));
        mx::array v = mx::contiguous(mx::slice(*s.v, {0, s.start, 0}, {kvH, s.len, hd}));
        mx::eval(k, v);
        m.slidingKV.push_back({(int) i, {k, v}});
    }
    rewindMark_ = std::move(m);
    return true;
}

int ESKVCache::rewindToMark() {
    if (!rewindMark_) return -1;
    const int P = rewindMark_->pos;
    std::vector<const std::pair<mx::array, mx::array> *> copyFor(slots_.size(), nullptr);
    for (const auto & lkv : rewindMark_->slidingKV) copyFor[lkv.first] = &lkv.second;
    // Validate BEFORE mutating: a refused rewind leaves the cache exactly as generated
    // (the caller keeps the unstripped turn — still a correct cache, just not excised).
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot & s = slots_[i];
        if (s.k.has_value() && !copyFor[i] && (s.lost || s.len < P)) { rewindMark_.reset(); return -1; }
        if (rq8slots_[i].q.has_value() && rq8slots_[i].len < P) { rewindMark_.reset(); return -1; }
        if (qslots_[i].t[0].has_value() && qslots_[i].len < P)  { rewindMark_.reset(); return -1; }
    }
    for (size_t i = 0; i < slots_.size(); ++i) {
        Slot & s = slots_[i];
        if (copyFor[i]) {
            // Evicting layer: install the mark's window (identical to the marker-restore
            // layout — content at the front, start 0; eviction re-applies on next append).
            const mx::array & k = copyFor[i]->first;
            const mx::array & v = copyFor[i]->second;
            const int kvH = k.shape(0), content = k.shape(1), hd = k.shape(2);
            const int cap = roundUpChunk(content + kGrowChunk);
            s.k = mx::slice_update(mx::zeros({kvH, cap, hd}, k.dtype()), k,
                                   {0, 0, 0}, {kvH, content, hd});
            s.v = mx::slice_update(mx::zeros({kvH, cap, hd}, v.dtype()), v,
                                   {0, 0, 0}, {kvH, content, hd});
            mx::eval(*s.k, *s.v);
            s.len = content; s.start = 0;
            s.lost = content < P;
        } else if (s.k.has_value()) {
            s.len = P; s.start = 0;   // full-prefix buffer: pure cursor truncation
        }
        if (rq8slots_[i].q.has_value()) rq8slots_[i].len = P;
        if (qslots_[i].t[0].has_value()) qslots_[i].len = P;
    }
    rewindMark_.reset();
    return P;
}

bool ESKVCache::saveSnapshot(const std::string & path, const std::string & fingerprint, int pos) const {
    std::unordered_map<std::string, mx::array> tensors;
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].k.has_value() && !slots_[i].v.has_value()) {
            // Raw-K layer (config.rawKV): single tensor holds the pre-norm k_proj cache.
            const Slot & s = slots_[i];
            const int kvH = s.k->shape(0), hd = s.k->shape(2);
            tensors.emplace("k_" + std::to_string(i),
                            (s.len == s.k->shape(1) && s.start == 0)
                                ? *s.k
                                : mx::slice(*s.k, {0, s.start, 0}, {kvH, s.len, hd}));
        } else if (slots_[i].k.has_value()) {
            auto kv = current((int) i, /*prealloc=*/true);
            tensors.emplace("k_" + std::to_string(i), kv.first);
            tensors.emplace("v_" + std::to_string(i), kv.second);
        } else if (rq8slots_[i].q.has_value()) {
            // q8 raw layer (config.rawKVQ8): persist packed rows + per-row scale/bias.
            const RQ8Slot & s = rq8slots_[i];
            const mx::array * bufs[3] = {&*s.q, &*s.sc, &*s.bs};
            const char * names[3] = {"rq_", "rs_", "rb_"};
            for (int t = 0; t < 3; ++t) {
                const int kvH = bufs[t]->shape(0), w = bufs[t]->shape(2);
                tensors.emplace(names[t] + std::to_string(i),
                                s.len == bufs[t]->shape(1)
                                    ? *bufs[t]
                                    : mx::slice(*bufs[t], {0, 0, 0}, {kvH, s.len, w}));
            }
        } else if (qslots_[i].t[0].has_value()) {
            // Hybrid quant-KV layer: persist the six packed/scales/biases live ranges.
            const QSlot & s = qslots_[i];
            for (int t = 0; t < 6; ++t) {
                const int kvH = s.t[t]->shape(0), w = s.t[t]->shape(2);
                tensors.emplace(kQuantNames[t] + std::to_string(i),
                                s.len == s.t[t]->shape(1)
                                    ? *s.t[t]
                                    : mx::slice(*s.t[t], {0, 0, 0}, {kvH, s.len, w}));
            }
        } else {
            return false;   // snapshot requires a fully-primed cache
        }
    }
    std::unordered_map<std::string, std::string> meta = {
        {"fingerprint", fingerprint},
        {"pos", std::to_string(pos)},
        {"layers", std::to_string(slots_.size())},
    };
    // Prefix markers: evicting-layer windows as m<idx>_k/v_<layer>, position and prefix
    // fingerprint in the metadata. Full-length layers need nothing — restore slices them.
    meta.emplace("marker_count", std::to_string(markers_.size()));
    for (size_t mi = 0; mi < markers_.size(); ++mi) {
        const Marker & m = markers_[mi];
        meta.emplace("marker_" + std::to_string(mi) + "_pos", std::to_string(m.pos));
        meta.emplace("marker_" + std::to_string(mi) + "_fp", m.fp);
        for (const auto & lkv : m.slidingKV) {
            tensors.emplace("m" + std::to_string(mi) + "_k_" + std::to_string(lkv.first), lkv.second.first);
            tensors.emplace("m" + std::to_string(mi) + "_v_" + std::to_string(lkv.first), lkv.second.second);
        }
    }
    try {
        mx::save_safetensors(path, tensors, meta);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

int ESKVCache::restoreSnapshot(const std::string & path, const std::string & fingerprint,
                               RawMode mode, const std::function<std::string(int)> & prefixFp) {
    try {
        auto loaded = mx::load_safetensors(path);
        auto & tensors = loaded.first;
        auto & meta = loaded.second;
        auto fp = meta.find("fingerprint");
        auto posIt = meta.find("pos");
        auto layersIt = meta.find("layers");
        if (fp == meta.end() || posIt == meta.end() || layersIt == meta.end()) return -1;
        if ((size_t) std::stoul(layersIt->second) != slots_.size()) return -1;

        // Full-fingerprint match restores everything. On a miss, fall back to the LONGEST
        // prefix marker whose stored fingerprint the caller can reproduce; full-length
        // layers get truncated to the marker position, evicting layers take the marker's
        // saved windows, and the caller prefills the remaining tail.
        const int fullPos = std::stoi(posIt->second);
        int targetPos = fullPos, markerIdx = -1;
        if (fp->second != fingerprint) {
            const int nMark = meta.count("marker_count") ? std::stoi(meta.at("marker_count")) : 0;
            if (!prefixFp || nMark == 0) return -1;
            int bestPos = -1;
            for (int mi = 0; mi < nMark; ++mi) {
                const std::string tag = "marker_" + std::to_string(mi);
                if (!meta.count(tag + "_pos") || !meta.count(tag + "_fp")) continue;
                const int mpos = std::stoi(meta.at(tag + "_pos"));
                if (mpos <= bestPos) continue;
                const std::string cand = prefixFp(mpos);
                if (!cand.empty() && cand == meta.at(tag + "_fp")) { bestPos = mpos; markerIdx = mi; }
            }
            if (markerIdx < 0) return -1;
            targetPos = bestPos;
        }

        // Validate completeness BEFORE mutating anything — a malformed file must not leave the
        // cache partially restored. Each layer is bf16 (k_i/v_i), raw (k_i only), or quant
        // (all six quant tensors).
        std::vector<bool> isQuant(slots_.size(), false), isRaw(slots_.size(), false),
                          isRQ8(slots_.size(), false);
        for (size_t i = 0; i < slots_.size(); ++i) {
            const bool hasK = tensors.count("k_" + std::to_string(i));
            if (hasK && tensors.count("v_" + std::to_string(i))) continue;
            if (hasK) { isRaw[i] = true; continue; }
            if (tensors.count("rq_" + std::to_string(i))) {
                if (!tensors.count("rs_" + std::to_string(i)) ||
                    !tensors.count("rb_" + std::to_string(i))) return -1;
                isRQ8[i] = true; continue;
            }
            bool q = true;
            for (int t = 0; t < 6; ++t) q = q && tensors.count(kQuantNames[t] + std::to_string(i));
            if (!q) return -1;
            isQuant[i] = true;
        }

        // Mode gate + conversion plan (see the header): the stored raw mode is inferred
        // from the tensor names. Raw <-> q8 converts during rehoming below; composite <->
        // raw is not derivable — mismatches fall back to a normal prime.
        bool anyRaw = false, anyRQ8 = false;
        for (size_t i = 0; i < slots_.size(); ++i) { anyRaw |= (bool) isRaw[i]; anyRQ8 |= (bool) isRQ8[i]; }
        if (mode != RawMode::asStored) {
            const bool storedRawish = anyRaw || anyRQ8;
            if (storedRawish && mode == RawMode::composite) return -1;
            if (!storedRawish && mode != RawMode::composite) return -1;
        }
        const bool toQ8  = mode == RawMode::rawQ8;
        const bool toRaw = mode == RawMode::raw;

        // Re-home each live range into a fresh chunk-aligned buffer (identical to the
        // compaction layout: content at the front, start 0). Values are byte-identical to
        // what was saved, so continuation from here matches a fresh prefill exactly.
        auto rehome = [](const mx::array & a, int cap) {
            const int kvH = a.shape(0), content = a.shape(1), w = a.shape(2);
            return mx::slice_update(mx::zeros({kvH, cap, w}, a.dtype()), a,
                                    {0, 0, 0}, {kvH, content, w});
        };
        // Marker restores truncate full-length layers to targetPos before rehoming (and
        // before any mode conversion — convert only what survives the cut).
        auto trunc = [targetPos](const mx::array & a) {
            const int kvH = a.shape(0), rows = a.shape(1), w = a.shape(2);
            return rows <= targetPos ? a : mx::slice(a, {0, 0, 0}, {kvH, targetPos, w});
        };
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (markerIdx >= 0) {
                // Evicting layers take the marker's saved windows verbatim.
                const std::string mk = "m" + std::to_string(markerIdx) + "_k_" + std::to_string(i);
                const std::string mv = "m" + std::to_string(markerIdx) + "_v_" + std::to_string(i);
                if (tensors.count(mk)) {
                    if (!tensors.count(mv)) return -1;
                    const mx::array & k = tensors.at(mk);
                    const int content = k.shape(1);
                    const int cap = roundUpChunk(content + kGrowChunk);
                    Slot & s = slots_[i];
                    s.k = rehome(k, cap);
                    s.v = rehome(tensors.at(mv), cap);
                    s.len = content; s.start = 0;
                    s.lost = content < targetPos;   // window rows only — no absolute prefix
                    rq8slots_[i] = RQ8Slot{};
                    continue;
                }
            }
            if (isRQ8[i]) {
                mx::array q = trunc(tensors.at("rq_" + std::to_string(i)));
                const int content = q.shape(1);
                const int cap = roundUpChunk(content + kGrowChunk);
                if (toRaw) {
                    // Rehydration: dequantize the stored q8 rows into a bf16 raw slot.
                    // The values are exactly what the q8 path was computing on.
                    mx::array dq = mx::astype(
                        mx::add(mx::multiply(mx::astype(q, mx::float32),
                                             trunc(tensors.at("rs_" + std::to_string(i)))),
                                trunc(tensors.at("rb_" + std::to_string(i)))),
                        mx::bfloat16);
                    Slot & s = slots_[i];
                    s.k = rehome(dq, cap); s.v.reset();
                    s.len = content; s.start = 0;
                    s.lost = content < targetPos;
                    rq8slots_[i] = RQ8Slot{};
                    continue;
                }
                RQ8Slot & s = rq8slots_[i];
                s.q  = rehome(q, cap);
                s.sc = rehome(trunc(tensors.at("rs_" + std::to_string(i))), cap);
                s.bs = rehome(trunc(tensors.at("rb_" + std::to_string(i))), cap);
                s.len = content;
                slots_[i].k.reset(); slots_[i].v.reset();
                slots_[i].len = 0; slots_[i].start = 0;
                continue;
            }
            if (isRaw[i] && toQ8) {
                // Stored bf16 raw, running q8: quantize during restore (the lazy first-
                // append migration would otherwise hitch the first user-visible token).
                mx::array k = trunc(tensors.at("k_" + std::to_string(i)));
                const int content = k.shape(1);
                const int cap = roundUpChunk(content + kGrowChunk);
                auto qsb = quantizeRawRows(k);
                RQ8Slot & s = rq8slots_[i];
                s.q  = rehome(qsb[0], cap);
                s.sc = rehome(qsb[1], cap);
                s.bs = rehome(qsb[2], cap);
                s.len = content;
                slots_[i].k.reset(); slots_[i].v.reset();
                slots_[i].len = 0; slots_[i].start = 0;
                continue;
            }
            if (isQuant[i]) {
                QSlot & s = qslots_[i];
                mx::array first = trunc(tensors.at(kQuantNames[0] + std::to_string(i)));
                const int content = first.shape(1);
                const int cap = roundUpChunk(content + kGrowChunk);
                for (int t = 0; t < 6; ++t)
                    s.t[t] = rehome(trunc(tensors.at(kQuantNames[t] + std::to_string(i))), cap);
                s.len = content;
                continue;
            }
            mx::array k = trunc(tensors.at("k_" + std::to_string(i)));
            const int content = k.shape(1);
            const int cap = roundUpChunk(content + kGrowChunk);
            Slot & s = slots_[i];
            s.k = rehome(k, cap);
            if (!isRaw[i]) s.v = rehome(trunc(tensors.at("v_" + std::to_string(i))), cap);
            else s.v.reset();
            s.len = content; s.start = 0;
            s.lost = content < targetPos;   // a stored sliding window is not a full prefix
        }
        // Eval by what each slot actually holds (conversions above may have changed it).
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (rq8slots_[i].q.has_value())   { mx::eval(*rq8slots_[i].q, *rq8slots_[i].sc, *rq8slots_[i].bs); }
            else if (isQuant[i])              { for (auto & t : qslots_[i].t) mx::eval(*t); }
            else if (slots_[i].v.has_value()) { mx::eval(*slots_[i].k, *slots_[i].v); }
            else                              { mx::eval(*slots_[i].k); }
        }
        return targetPos;
    } catch (const std::exception &) {
        return -1;
    }
}

ESKVCache::Raw ESKVCache::updateRaw(int layer, const mx::array & kNew, bool prealloc) {
    const int kvH = kNew.shape(0), nNew = kNew.shape(1), hd = kNew.shape(2);
    if (!prealloc) {
        // Legacy concat-grow reference path (A/B only).
        if (!k_[layer].has_value()) k_[layer] = kNew;
        else k_[layer] = mx::concatenate({*k_[layer], kNew}, 1);
        const int len = k_[layer]->shape(1);
        return {*k_[layer], *k_[layer], len, len};
    }
    Slot & s = slots_[layer];
    if (s.k.has_value() && s.v.has_value())
        throw std::runtime_error("ESKVCache::updateRaw: slot holds composite K/V content "
                                 "(post-norm) — not derivable as raw; re-prime the cache");
    // Lazy rehydration backstop (a q8 cache reached without the mode-aware restore):
    // dequantize the live range into a bf16 raw slot, retire the q8 slot.
    if (!s.k.has_value() && rq8slots_[layer].q.has_value()) {
        RQ8Slot & r = rq8slots_[layer];
        mx::array live = (r.len == r.q->shape(1)) ? *r.q : mx::slice(*r.q, {0, 0, 0}, {kvH, r.len, hd});
        mx::array sc = (r.len == r.sc->shape(1)) ? *r.sc : mx::slice(*r.sc, {0, 0, 0}, {kvH, r.len, 1});
        mx::array bs = (r.len == r.bs->shape(1)) ? *r.bs : mx::slice(*r.bs, {0, 0, 0}, {kvH, r.len, 1});
        mx::array dq = mx::astype(
            mx::add(mx::multiply(mx::astype(live, mx::float32), sc), bs), kNew.dtype());
        const int cap = roundUpChunk(r.len + kGrowChunk);
        s.k = mx::slice_update(mx::zeros({kvH, cap, hd}, kNew.dtype()), dq,
                               {0, 0, 0}, {kvH, r.len, hd});
        s.len = r.len; s.start = 0;
        rq8slots_[layer] = RQ8Slot{};
    }
    if (!s.k.has_value()) {
        s.k = kNew; s.len = nNew; s.start = 0;
    } else {
        const int cap = s.k->shape(1);
        if (s.len + nNew > cap) {
            const int newCap = roundUpChunk(s.len + nNew + kGrowChunk);
            mx::array nk = mx::zeros({kvH, newCap, hd}, kNew.dtype());
            nk = mx::slice_update(nk, mx::slice(*s.k, {0, 0, 0}, {kvH, s.len, hd}),
                                  {0, 0, 0}, {kvH, s.len, hd});
            s.k = nk;
        }
        s.k = mx::slice_update(*s.k, kNew, {0, s.len, 0}, {kvH, s.len + nNew, hd});
        s.len += nNew;
    }
    mx::array view = (s.len == s.k->shape(1))
        ? *s.k
        : mx::slice(*s.k, {0, 0, 0}, {kvH, s.len, hd});
    return {view, *s.k, s.len, s.k->shape(1)};
}

// Per-row affine u8: q = round((x - min) / scale), scale = (max - min)/255 floored away
// from zero so constant rows stay finite. Dequant (q * scale + min) is what BOTH consumers
// see — the decode kernel in registers and the prefill ops path — so the two stay
// value-identical to each other, exactly the rawKV determinism argument. One fused
// dispatch (esQuantizeRawRows): the op-level equivalent (~6 dispatches) measured
// ~3 ms/token of pure dispatch overhead across the 10 global layers at decode.
std::array<mx::array, 3> ESKVCache::quantizeRawRows(const mx::array & kNew) {
    return esQuantizeRawRows(kNew);
}

ESKVCache::RawQ8 ESKVCache::updateRawQ8(int layer, const mx::array & kNew) {
    const int kvH = kNew.shape(0), nNew = kNew.shape(1), hd = kNew.shape(2);
    RQ8Slot & s = rq8slots_[layer];

    if (slots_[layer].k.has_value() && slots_[layer].v.has_value())
        throw std::runtime_error("ESKVCache::updateRawQ8: slot holds composite K/V content "
                                 "(post-norm) — not derivable as raw; re-prime the cache");

    // Lazy migration from a bf16 raw slot (a restored kv snapshot): quantize the live
    // range once, retire the bf16 buffer, continue as q8.
    if (!s.q.has_value() && slots_[layer].k.has_value()) {
        Slot & b = slots_[layer];
        mx::array live = (b.len == b.k->shape(1) && b.start == 0)
            ? *b.k
            : mx::slice(*b.k, {0, b.start, 0}, {kvH, b.len, hd});
        auto qsb = quantizeRawRows(live);
        const int cap = roundUpChunk(b.len + kGrowChunk);
        s.q  = mx::slice_update(mx::zeros({kvH, cap, hd}, mx::uint8),   qsb[0], {0, 0, 0}, {kvH, b.len, hd});
        s.sc = mx::slice_update(mx::zeros({kvH, cap, 1},  mx::float32), qsb[1], {0, 0, 0}, {kvH, b.len, 1});
        s.bs = mx::slice_update(mx::zeros({kvH, cap, 1},  mx::float32), qsb[2], {0, 0, 0}, {kvH, b.len, 1});
        s.len = b.len;
        b.k.reset(); b.len = 0; b.start = 0;
    }

    auto qsb = quantizeRawRows(kNew);
    if (!s.q.has_value()) {
        s.q = qsb[0]; s.sc = qsb[1]; s.bs = qsb[2]; s.len = nNew;
    } else {
        const int cap = s.q->shape(1);
        if (s.len + nNew > cap) {
            const int newCap = roundUpChunk(s.len + nNew + kGrowChunk);
            mx::array * bufs[3] = {&*s.q, &*s.sc, &*s.bs};
            for (auto * b : bufs) {
                const int w = b->shape(2);
                *b = mx::slice_update(mx::zeros({kvH, newCap, w}, b->dtype()),
                                      mx::slice(*b, {0, 0, 0}, {kvH, s.len, w}),
                                      {0, 0, 0}, {kvH, s.len, w});
            }
        }
        s.q  = mx::slice_update(*s.q,  qsb[0], {0, s.len, 0}, {kvH, s.len + nNew, hd});
        s.sc = mx::slice_update(*s.sc, qsb[1], {0, s.len, 0}, {kvH, s.len + nNew, 1});
        s.bs = mx::slice_update(*s.bs, qsb[2], {0, s.len, 0}, {kvH, s.len + nNew, 1});
        s.len += nNew;
    }

    const int pitch = s.q->shape(1);
    auto view = [&](const mx::array & b, int w) {
        return (s.len == pitch) ? b : mx::slice(b, {0, 0, 0}, {kvH, s.len, w});
    };
    return {view(*s.q, hd), view(*s.sc, 1), view(*s.bs, 1), *s.q, *s.sc, *s.bs, s.len, pitch};
}

ESKVCache::QKV ESKVCache::updateQuant(int layer, const mx::array & kNew, const mx::array & vNew,
                                      int groupSize, int bits, bool prealloc) {
    // Quantize the new tokens along the head dim (last axis). Each seq position is independently
    // quantized, so appending along the seq axis just stacks packed rows — valid.
    auto kq = mx::quantize(kNew, groupSize, bits);  // {packed, scales, biases}
    auto vq = mx::quantize(vNew, groupSize, bits);

    if (prealloc) {
        // ── Prealloc mode: slice_update appends into chunk-grown buffers, same donation and
        // amortization arguments as update(). Six tensors share one cursor; widths differ per
        // tensor (packed hd*bits/32, scales/biases hd/groupSize) but the seq axis is common.
        QSlot & s = qslots_[layer];
        mx::array parts[6] = {kq[0], kq[1], kq[2], vq[0], vq[1], vq[2]};
        const int nNew = kNew.shape(1);
        if (!s.t[0].has_value()) {
            for (int i = 0; i < 6; ++i) s.t[i] = parts[i];
            s.len = nNew;
        } else {
            const int kvH = s.t[0]->shape(0);
            if (s.len + nNew > s.t[0]->shape(1)) {
                const int newCap = roundUpChunk(s.len + nNew + kGrowChunk);
                for (int i = 0; i < 6; ++i) {
                    const int w = s.t[i]->shape(2);
                    mx::array nb = mx::zeros({kvH, newCap, w}, s.t[i]->dtype());
                    s.t[i] = mx::slice_update(nb, mx::slice(*s.t[i], {0, 0, 0}, {kvH, s.len, w}),
                                              {0, 0, 0}, {kvH, s.len, w});
                }
            }
            for (int i = 0; i < 6; ++i) {
                const int w = parts[i].shape(2);
                s.t[i] = mx::slice_update(*s.t[i], parts[i], {0, s.len, 0}, {kvH, s.len + nNew, w});
            }
            s.len += nNew;
        }
        auto view = [&](int i) {
            const int kvH = s.t[i]->shape(0), w = s.t[i]->shape(2);
            if (s.len == s.t[i]->shape(1)) return *s.t[i];
            return mx::slice(*s.t[i], {0, 0, 0}, {kvH, s.len, w});
        };
        return {view(0), view(1), view(2), view(3), view(4), view(5)};
    }

    if (!kq_[layer].has_value()) {
        kq_[layer] = kq[0]; ks_[layer] = kq[1]; kb_[layer] = kq[2];
        vq_[layer] = vq[0]; vs_[layer] = vq[1]; vb_[layer] = vq[2];
    } else {
        kq_[layer] = mx::concatenate({*kq_[layer], kq[0]}, 1);
        ks_[layer] = mx::concatenate({*ks_[layer], kq[1]}, 1);
        kb_[layer] = mx::concatenate({*kb_[layer], kq[2]}, 1);
        vq_[layer] = mx::concatenate({*vq_[layer], vq[0]}, 1);
        vs_[layer] = mx::concatenate({*vs_[layer], vq[1]}, 1);
        vb_[layer] = mx::concatenate({*vb_[layer], vq[2]}, 1);
    }
    return {*kq_[layer], *ks_[layer], *kb_[layer], *vq_[layer], *vs_[layer], *vb_[layer]};
}

void ESKVCache::reset() {
    for (auto & a : k_) a.reset();
    for (auto & a : v_) a.reset();
    for (auto & s : slots_) { s.k.reset(); s.v.reset(); s.len = 0; s.start = 0; s.lost = false; }
    for (auto & s : qslots_) { for (auto & t : s.t) t.reset(); s.len = 0; }
    for (auto & s : rq8slots_) { s.q.reset(); s.sc.reset(); s.bs.reset(); s.len = 0; }
    markers_.clear();
    rewindMark_.reset();
    for (auto & a : kq_) a.reset();
    for (auto & a : ks_) a.reset();
    for (auto & a : kb_) a.reset();
    for (auto & a : vq_) a.reset();
    for (auto & a : vs_) a.reset();
    for (auto & a : vb_) a.reset();
    seqLen_ = 0;
}

}  // namespace es
