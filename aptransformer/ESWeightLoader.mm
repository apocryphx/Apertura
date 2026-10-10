#include "ESWeightLoader.h"
#include <map>

#import <Foundation/Foundation.h>
#include <set>
#include <stdexcept>
#include <vector>

namespace es {

static const std::string kTextPrefix = "model.language_model.";

ESWeightLoader::ESWeightLoader(const std::string & modelDir, const ESModelConfig & config) {
    @autoreleasepool {
        NSString * mdir = [NSString stringWithUTF8String:modelDir.c_str()];
        NSData * mdata = [NSData dataWithContentsOfFile:[mdir stringByAppendingPathComponent:@"manifest.json"]];
        if (mdata) {
            NSDictionary * m = [NSJSONSerialization JSONObjectWithData:mdata options:0 error:nil];
            if ([[m objectForKey:@"kind"] isEqual:@"apertura-model"]) { loadBundle(modelDir, config); return; }
        }
    }
    loadHF(modelDir, config);
}

void ESWeightLoader::loadHF(const std::string & modelDir, const ESModelConfig & config) {
    @autoreleasepool {
        NSString * dir = [NSString stringWithUTF8String:modelDir.c_str()];
        NSString * indexPath = [dir stringByAppendingPathComponent:@"model.safetensors.index.json"];
        NSData * idxData = [NSData dataWithContentsOfFile:indexPath];

        // Collect the set of shard files we need (those holding text-decoder weights).
        std::set<std::string> shards;
        if (idxData) {
            NSDictionary * idx = [NSJSONSerialization JSONObjectWithData:idxData options:0 error:nil];
            NSDictionary * wmap = idx[@"weight_map"];
            for (NSString * wname in wmap) {
                if ([wname hasPrefix:@(kTextPrefix.c_str())]) {
                    shards.insert([wmap[wname] UTF8String]);
                }
            }
        } else {
            // Single-file fallback.
            shards.insert("model.safetensors");
        }

        for (const std::string & shard : shards) {
            NSString * shardPath = [dir stringByAppendingPathComponent:@(shard.c_str())];
            auto loaded = mx::load_safetensors([shardPath UTF8String]);
            for (auto & kv : loaded.first) {
                const std::string & name = kv.first;
                if (name.rfind(kTextPrefix, 0) != 0) continue;  // skip vision/audio/embed_vision
                std::string key = name.substr(kTextPrefix.size());
                weights_.emplace(std::move(key), mx::astype(kv.second, config.computeDtype));
            }
        }

        if (weights_.find("embed_tokens.weight") == weights_.end()) {
            throw std::runtime_error("ESWeightLoader: embed_tokens.weight not found under " +
                                     kTextPrefix + " in " + modelDir);
        }
    }
}

const mx::array & ESWeightLoader::get(const std::string & name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("ESWeightLoader: missing tensor '" + name + "'");
    }
    return it->second;
}

const mx::array & ESWeightLoader::layer(int idx, const std::string & suffix) const {
    return get("layers." + std::to_string(idx) + "." + suffix);
}

#pragma mark - Bundle (.apml) reload

void ESWeightLoader::loadBundle(const std::string & packageDir, const ESModelConfig & config) {
    (void) config;  // bundle tensors are stored verbatim — no compute-dtype cast
    @autoreleasepool {
        isBundle_ = true;
        NSString * dir = [NSString stringWithUTF8String:packageDir.c_str()];
        NSData * md = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
        NSDictionary * manifest = md ? [NSJSONSerialization JSONObjectWithData:md options:0 error:nil] : nil;
        if (!manifest)
            throw std::runtime_error("ESWeightLoader: missing/invalid manifest.json in bundle " + packageDir);

        NSString * defId = manifest[@"default_variant"];
        NSDictionary * variant = nil;
        for (NSDictionary * v in manifest[@"variants"]) {
            if ([v[@"id"] isEqual:defId]) { variant = v; break; }
        }
        if (!variant) throw std::runtime_error("ESWeightLoader: default_variant not found in manifest");

        NSDictionary * q = variant[@"quantization"];
        bundleBits_      = [q[@"bits"] intValue];
        bundleGroupSize_ = [q[@"group_size"] intValue];
        bundleEmbedBits_ = [q[@"embed_bits"] intValue];
        bundlePleBits_   = [q[@"ple_bits"] intValue];   // absent in pre-ple bundles -> 0 (bf16 table)
        bundleLattice_   = q[@"lattice"] != nil;         // lattice-exact QAT recipe (2026-10-09)

        NSString * vpath = variant[@"path"];  // e.g. "weights/mlx-q4"
        for (NSString * f in variant[@"files"]) {
            NSString * stPath = [[dir stringByAppendingPathComponent:vpath] stringByAppendingPathComponent:f];
            auto loaded = mx::load_safetensors([stPath UTF8String]);
            for (auto & kv : loaded.first) {
                weights_.emplace(kv.first, kv.second);  // verbatim: packed u32 stays u32
            }
        }
        if (weights_.find("embed_tokens.weight") == weights_.end())
            throw std::runtime_error("ESWeightLoader: embed_tokens.weight missing in bundle " + packageDir);
    }
}

ESWeightLoader::QuantTriple ESWeightLoader::quantized(const std::string & name) const {
    return { get(name), get(name + ".scales"), get(name + ".biases") };
}

#pragma mark - Layer factories

ESLinear esMakeLinear(const ESWeightLoader & w, const std::string & name, int quantBits, int groupSize) {
    if (w.hasQuantized(name)) {
        auto q = w.quantized(name);
        return ESLinear(q.weight, q.scales, q.biases, w.bundleBits(), w.bundleGroupSize());
    }
    return ESLinear(w.get(name), quantBits, groupSize);
}

ESEmbedding esMakeEmbedding(const ESWeightLoader & w, const std::string & name, int quantEmbedBits, int groupSize) {
    if (w.hasQuantized(name)) {
        auto q = w.quantized(name);
        if (quantEmbedBits > 0 && quantEmbedBits != w.bundleEmbedBits() && !w.bundleLattice()) {
            // Q4-head mode (roadmap P4) — never on a lattice-exact bundle (its head is exact already): re-quantize the bundle's packed embedding/LM head at the
            // requested width (--quant-embed N on a bundle). Dequant->requant from Q8 loses ~nothing
            // vs quantizing from bf16 (Q8's error is tiny against a Q4 bin). The head GEMV reads
            // ~1.50 GB (Q8) vs ~0.79 GB (Q4) per decode token -> ~1.5 ms/token (~+3% decode) for a
            // small top-1 cost — measured via --head-verify. Default (no flag) keeps the bundle's
            // head verbatim; quality-first stays Q8.
            mx::array full = mx::dequantize(q.weight, q.scales, q.biases,
                                            w.bundleGroupSize(), w.bundleEmbedBits());
            return ESEmbedding(full, quantEmbedBits, w.bundleGroupSize());
        }
        return ESEmbedding(q.weight, q.scales, q.biases, w.bundleEmbedBits(), w.bundleGroupSize());
    }
    return ESEmbedding(w.get(name), quantEmbedBits, groupSize);
}

ESEmbedding esMakePleTable(const ESWeightLoader & w, const std::string & name, int quantPleBits, int groupSize) {
    if (w.hasQuantized(name)) {
        auto q = w.quantized(name);
        if (quantPleBits > 0 && quantPleBits != w.bundlePleBits() && !w.bundleLattice()) {
            // Re-quantize the bundle's packed table (never on a lattice-exact bundle) at the requested width (--quant-ple N on a
            // bundle), same dequant->requant argument as the Q4-head path above.
            mx::array full = mx::dequantize(q.weight, q.scales, q.biases,
                                            w.bundleGroupSize(), w.bundlePleBits());
            return ESEmbedding(full, quantPleBits, w.bundleGroupSize());
        }
        return ESEmbedding(q.weight, q.scales, q.biases, w.bundlePleBits(), w.bundleGroupSize());
    }
    return ESEmbedding(w.get(name), quantPleBits, groupSize);
}

ESExperts esMakeExperts(const ESWeightLoader & w, const std::string & gateUpName,
                        const std::string & downName, int quantBits, int groupSize) {
    if (w.hasQuantized(gateUpName)) {
        auto g = w.quantized(gateUpName);
        auto d = w.quantized(downName);
        return ESExperts(g.weight, g.scales, g.biases, d.weight, d.scales, d.biases,
                         w.bundleBits(), w.bundleGroupSize());
    }
    return ESExperts(w.get(gateUpName), w.get(downName), quantBits, groupSize);
}

#pragma mark - Quantized bundle export

static bool octEndsWith(const std::string & s, const std::string & suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// The projections the runtime quantizes (must mirror the layer constructors:
// ESAttention q/k/v/o, ESMLPBlock gate/up/down, ESExperts gate_up/down). The
// round-trip conformance test guards against drift from this list.
static bool octIsLayerProjQuant(const std::string & name) {
    static const char * kSfx[] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "experts.gate_up_proj", "experts.down_proj",
    };
    for (const char * s : kSfx) if (octEndsWith(name, s)) return true;
    return false;
}

static void octCopyIfPresent(NSFileManager * fm, NSString * srcDir, NSString * dstDir, NSString * file) {
    NSString * src = [srcDir stringByAppendingPathComponent:file];
    if ([fm fileExistsAtPath:src]) {
        // Copy bytes (not the link): HF cache snapshots symlink into ../../blobs, and a
        // preserved relative symlink breaks once the package is moved out of the cache.
        // dataWithContentsOfFile follows the symlink, so the package holds a real file.
        NSData * data = [NSData dataWithContentsOfFile:src];
        if (data) [data writeToFile:[dstDir stringByAppendingPathComponent:file] atomically:YES];
    }
}

#pragma mark - QAT lattice-exact quantization (int4 g32, learned step)

// Checkpoint structure (measured on google/gemma-4-31B-it-qat-q4_0-unquantized, 2026-10-09):
// every quantized tensor is, per 32-element block along the input dim, bf16(k * d) with integer
// codes k in [-8, 7] and a per-block step d that is NOT a function of the block's absmax -- it is
// the QAT-learned scale, so the extreme code present is 8/7 in ~60%/~38% of blocks and smaller in
// the rest. d is recovered from the structure: the smallest K in 1..8 for which absmax/K makes
// every element a near-integer multiple with codes in range (the absmax codes to +-K; a larger K
// would need an all-even block, which merely picks a finer step that is still exact). The step
// is then refined by least squares and snapped to the bf16 value that reproduces the most stored
// bf16 weights (the stored values are bf16 roundings of k*d, so no single bf16 step hits all of
// them: ~90.5% bit-exact, 100% within one bf16 ulp is the ceiling for this checkpoint).
std::vector<mx::array> quantizeQ4Lattice(const mx::array & wIn, ESLatticeFit * fit) {
    constexpr int kGroup = 32, kBits = 4, kPerWord = 32 / kBits;  // 8 nibbles per uint32
    constexpr float kTol = 0.06f;  // |w/d - round| noise bound is ~0.03 (bf16 on w and on absmax)
    const int in = wIn.shape(-1);
    if (in % kGroup != 0)
        throw std::runtime_error("quantizeQ4Lattice: last dim " + std::to_string(in) + " not a multiple of 32");
    mx::Shape lead(wIn.shape().begin(), wIn.shape().end() - 1);  // [..., ] without `in`

    // Blocks [..., in/32, 32] in f32 (the arithmetic below must not round in bf16).
    mx::Shape blk = lead; blk.push_back(in / kGroup); blk.push_back(kGroup);
    mx::array wb  = mx::reshape(wIn, blk);
    mx::array w32 = mx::astype(wb, mx::float32);
    const mx::array zero(0.0f), one(1.0f);

    // 1. Recover the step: smallest K with absmax/K a consistent int4 lattice.
    mx::array am = mx::max(mx::abs(w32), -1, /*keepdims=*/true);           // [..., in/32, 1]
    am = mx::where(mx::equal(am, zero), one, am);                             // all-zero block
    mx::array chosen = mx::zeros_like(am), d = mx::zeros_like(am);
    for (int K = 1; K <= 8; ++K) {
        mx::array dK = mx::divide(am, mx::array((float) K));
        mx::array r  = mx::divide(w32, dK);
        mx::array k  = mx::round(r);
        mx::array ok = mx::logical_and(
            mx::all(mx::less(mx::abs(mx::subtract(r, k)), mx::array(kTol)), -1, true),
            mx::logical_and(mx::all(mx::greater_equal(k, mx::array(-8.0f)), -1, true),
                            mx::all(mx::less_equal(k, mx::array(7.0f)), -1, true)));
        ok = mx::logical_and(ok, mx::equal(chosen, zero));
        chosen = mx::where(ok, mx::array((float) K), chosen);
        d      = mx::where(ok, dK, d);
    }
    d = mx::where(mx::equal(d, zero), mx::divide(am, mx::array(8.0f)), d);  // unresolved: best effort

    // 2. Codes on the recovered step; least-squares refinement of d from all codes.
    mx::array k = mx::clip(mx::round(mx::divide(w32, d)), mx::array(-8.0f), mx::array(7.0f));
    mx::array kk = mx::maximum(mx::sum(mx::multiply(k, k), -1, true), one);
    mx::array dls = mx::divide(mx::sum(mx::multiply(w32, k), -1, true), kk);
    dls = mx::where(mx::equal(dls, zero), d, dls);

    // 3. Snap to the bf16 step (+-4 ulps around the LS estimate) that reproduces the most stored
    //    weights. The scale is stored in the weight dtype; the kernel applies it in float.
    mx::array baseBits = mx::astype(mx::view(mx::astype(dls, mx::bfloat16), mx::uint16), mx::int32);
    mx::array bestHit = mx::zeros_like(am), bestS = mx::astype(dls, mx::bfloat16);
    for (int off = -4; off <= 4; ++off) {
        mx::array sb  = mx::view(mx::astype(mx::add(baseBits, mx::array(off)), mx::uint16), mx::bfloat16);
        mx::array s32 = mx::astype(sb, mx::float32);
        s32 = mx::where(mx::equal(s32, zero), one, s32);
        mx::array rec = mx::astype(mx::multiply(k, s32), wIn.dtype());
        mx::array hit = mx::astype(mx::sum(mx::astype(mx::equal(rec, wb), mx::int32), -1, true), mx::float32);
        mx::array better = mx::greater(hit, bestHit);
        bestHit = mx::where(better, hit, bestHit);
        bestS   = mx::where(better, sb, bestS);
    }
    mx::array s32 = mx::astype(bestS, mx::float32);

    // 4. Pack 8 nibbles per uint32, element i in bits [4i, 4i+4) (MLX affine layout; the unit test
    //    gates this against mx::dequantize). Nibbles are disjoint, so sum == bitwise-or.
    mx::array q = mx::astype(mx::add(k, mx::array(8.0f)), mx::uint32);     // [..., in/32, 32] in [0,15]
    mx::Shape pk = lead; pk.push_back(in / kPerWord); pk.push_back(kPerWord);
    const uint32_t shiftsV[kPerWord] = {0, 4, 8, 12, 16, 20, 24, 28};
    mx::array shifts(shiftsV, {kPerWord}, mx::uint32);
    mx::array packed = mx::sum(mx::left_shift(mx::reshape(q, pk), shifts), -1);  // [..., in/8] uint32

    // scale = s, bias = -8 s  ->  (k+8) s - 8 s = k s, exact in f32 (both products are exact).
    mx::Shape sc = lead; sc.push_back(in / kGroup);
    mx::array scales = mx::astype(mx::reshape(bestS, sc), wIn.dtype());
    mx::array biases = mx::astype(mx::reshape(mx::multiply(s32, mx::array(-8.0f)), sc), wIn.dtype());

    if (fit) {
        mx::array recon = mx::multiply(k, s32);                              // what the kernel sees
        mx::array exact = mx::equal(mx::astype(recon, wIn.dtype()), wb);     // same bf16 bits
        mx::array err   = mx::abs(mx::subtract(recon, w32));
        mx::array near  = mx::less_equal(err, mx::multiply(mx::abs(w32), mx::array(1.0f / 128.0f)));
        mx::array nExact = mx::sum(mx::astype(exact, mx::int64));
        mx::array nNear  = mx::sum(mx::astype(near,  mx::int64));
        mx::array eMax   = mx::max(err);
        mx::eval(nExact, nNear, eMax);
        fit->total     += (uint64_t) w32.size();
        fit->exact     += (uint64_t) nExact.item<int64_t>();
        fit->near      += (uint64_t) nNear.item<int64_t>();
        fit->maxAbsErr  = std::max(fit->maxAbsErr, eMax.item<float>());
    }
    return {packed, scales, biases};
}

// Tensor class for the scan / export report: the suffix after the last "layers.N." (or the
// whole name for embeddings), so 60 q_proj tensors roll up into one row.
static std::string octTensorClass(const std::string & name) {
    size_t p = name.find("layers.");
    if (p == std::string::npos) return name;
    size_t dot = name.find('.', p + 7);           // skip "layers.N"
    return dot == std::string::npos ? name : name.substr(dot + 1);
}

bool scanQ4Lattice(const std::string & modelDir, std::string * error) {
    try {
        ESModelConfig cfg;  // bf16
        ESWeightLoader loader(modelDir, cfg);
        std::map<std::string, ESLatticeFit> byClass;
        std::map<std::string, int> count;
        ESLatticeFit all;
        std::printf("== QAT int4 lattice scan (g32, codes [-8,7], learned step) ==\n  model : %s\n", modelDir.c_str());
        for (const auto & kv : loader.all()) {
            const std::string & name = kv.first;
            bool cand = name == "embed_tokens.weight" || name == "embed_tokens_per_layer.weight"
                        || octIsLayerProjQuant(name);
            if (!cand) continue;
            ESLatticeFit f;
            (void) quantizeQ4Lattice(kv.second, &f);
            std::string cls = octTensorClass(name);
            ESLatticeFit & c = byClass[cls];
            c.total += f.total; c.exact += f.exact; c.near += f.near;
            c.maxAbsErr = std::max(c.maxAbsErr, f.maxAbsErr);
            count[cls]++;
            all.total += f.total; all.exact += f.exact; all.near += f.near;
            all.maxAbsErr = std::max(all.maxAbsErr, f.maxAbsErr);
        }
        std::printf("  %-36s %7s %14s %10s %10s %12s\n", "tensor class", "tensors", "weights", "exact%", "≤1ulp%", "max|err|");
        for (const auto & kv : byClass) {
            const ESLatticeFit & f = kv.second;
            std::printf("  %-36s %7d %14llu %9.4f%% %9.4f%% %12.3e\n", kv.first.c_str(), count[kv.first],
                        (unsigned long long) f.total, 100.0 * f.exactFrac(), 100.0 * f.nearFrac(), f.maxAbsErr);
        }
        std::printf("  %-36s %7s %14llu %9.4f%% %9.4f%% %12.3e\n", "ALL", "",
                    (unsigned long long) all.total, 100.0 * all.exactFrac(), 100.0 * all.nearFrac(), all.maxAbsErr);
        std::printf("  verdict: %s\n", all.nearFrac() >= 0.999
                    ? "ON the QAT int4 lattice -> export with --export-lattice"
                    : "NOT on the QAT int4 lattice (plain checkpoint) -> export with the affine recipe");
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool verifyLatticeBundle(const std::string & modelDir, const std::string & apml, std::string * error) {
    try {
        ESModelConfig cfg;
        ESWeightLoader src(modelDir, cfg);
        ESWeightLoader bun(apml, cfg);
        if (!bun.isBundle()) throw std::runtime_error("not an .apml bundle: " + apml);
        const int gs = bun.bundleGroupSize();
        std::printf("== verify-lattice (bundle dequantized vs source bf16, weight by weight) ==\n"
                    "  bundle : %s\n  source : %s\n  recipe : bits=%d embed_bits=%d ple_bits=%d group=%d\n",
                    apml.c_str(), modelDir.c_str(), bun.bundleBits(), bun.bundleEmbedBits(), bun.bundlePleBits(), gs);
        std::map<std::string, ESLatticeFit> byClass; std::map<std::string, int> count; ESLatticeFit all;
        for (const auto & kv : src.all()) {
            const std::string & name = kv.first;
            if (!bun.hasQuantized(name)) continue;
            int bits = name == "embed_tokens.weight" ? bun.bundleEmbedBits()
                     : name == "embed_tokens_per_layer.weight" ? bun.bundlePleBits() : bun.bundleBits();
            auto q = bun.quantized(name);
            mx::array w32 = mx::astype(kv.second, mx::float32);
            mx::array rec = mx::dequantize(q.weight, q.scales, q.biases, gs, bits);  // f32 math on bf16 inputs
            mx::array rec32 = mx::astype(rec, mx::float32);
            mx::array exact = mx::equal(mx::astype(rec32, kv.second.dtype()), kv.second);
            mx::array err   = mx::abs(mx::subtract(rec32, w32));
            mx::array near  = mx::less_equal(err, mx::multiply(mx::abs(w32), mx::array(1.0f / 128.0f)));
            mx::array nE = mx::sum(mx::astype(exact, mx::int64)), nN = mx::sum(mx::astype(near, mx::int64)), eM = mx::max(err);
            mx::eval(nE, nN, eM);
            ESLatticeFit f; f.total = (uint64_t) w32.size(); f.exact = (uint64_t) nE.item<int64_t>();
            f.near = (uint64_t) nN.item<int64_t>(); f.maxAbsErr = eM.item<float>();
            std::string cls = octTensorClass(name);
            ESLatticeFit & c = byClass[cls];
            c.total += f.total; c.exact += f.exact; c.near += f.near; c.maxAbsErr = std::max(c.maxAbsErr, f.maxAbsErr);
            count[cls]++;
            all.total += f.total; all.exact += f.exact; all.near += f.near; all.maxAbsErr = std::max(all.maxAbsErr, f.maxAbsErr);
        }
        std::printf("  %-36s %7s %14s %10s %10s %12s\n", "tensor class", "tensors", "weights", "exact%", "≤1ulp%", "max|err|");
        for (const auto & kv : byClass) {
            const ESLatticeFit & f = kv.second;
            std::printf("  %-36s %7d %14llu %9.4f%% %9.4f%% %12.3e\n", kv.first.c_str(), count[kv.first],
                        (unsigned long long) f.total, 100.0 * f.exactFrac(), 100.0 * f.nearFrac(), f.maxAbsErr);
        }
        std::printf("  %-36s %7s %14llu %9.4f%% %9.4f%% %12.3e\n", "ALL", "",
                    (unsigned long long) all.total, 100.0 * all.exactFrac(), 100.0 * all.nearFrac(), all.maxAbsErr);
        bool ok = all.nearFrac() >= 0.999;
        std::printf("  %s (gate: >= 99.9%% of weights within one bf16 ulp of the source)\n", ok ? "PASS" : "FAIL");
        return ok;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool exportQuantizedBundle(const std::string & modelDir,
                           const std::string & outPackagePath,
                           const ESBundleExportOptions & opts,
                           std::string * error) {
    auto fail = [&](const std::string & msg) { if (error) *error = msg; return false; };

    @autoreleasepool {
        NSFileManager * fm = [NSFileManager defaultManager];
        NSString * dir = [NSString stringWithUTF8String:modelDir.c_str()];

        // config.json must exist; we read model_type for the manifest and copy it verbatim.
        NSString * configPath = [dir stringByAppendingPathComponent:@"config.json"];
        NSData * configData = [NSData dataWithContentsOfFile:configPath];
        if (!configData) return fail("config.json not found in " + modelDir);
        NSString * architecture = @"gemma4";
        if (NSDictionary * cfg = [NSJSONSerialization JSONObjectWithData:configData options:0 error:nil]) {
            if (NSString * mt = cfg[@"model_type"]) architecture = mt;
            else if (NSDictionary * tc = cfg[@"text_config"]) if (NSString * mt2 = tc[@"model_type"]) architecture = mt2;
        }

        // Lattice mode is defined only at 4 bits / group 32 (the q4_0 grid).
        const int groupSize = opts.lattice ? 32 : opts.groupSize;
        const int bits      = opts.lattice ? 4  : opts.bits;
        int embedBitsOut = opts.embedBits, pleBitsOut = opts.pleBits;  // what we actually write
        constexpr double kLatticeAccept = 0.999;                         // near/total to trust the fit

        // Load bf16 weights. The loader only needs computeDtype; quant fields are irrelevant here.
        ESModelConfig cfg;  // defaults: computeDtype == bfloat16
        std::unordered_map<std::string, mx::array> out;
        std::vector<mx::array> toEval;
        ESLatticeFit latticeAll;
        int latticeExactTensors = 0, latticeFallbackTensors = 0;
        std::vector<std::string> fallbackNames;
        try {
            ESWeightLoader loader(modelDir, cfg);
            for (const auto & kv : loader.all()) {
                const std::string & name = kv.first;
                const mx::array & w = kv.second;
                int b = 0;
                bool isEmbed = name == "embed_tokens.weight";
                bool isPle   = name == "embed_tokens_per_layer.weight";  // elastic PLE table
                if (isEmbed)      b = opts.embedBits;
                else if (isPle)   b = opts.pleBits;
                else if (octIsLayerProjQuant(name)) b = bits;

                if (b > 0 && opts.lattice) {
                    // Lattice-exact when the tensor is on the grid; otherwise the affine recipe at
                    // the requested bits (group 32). Evaluated per tensor to bound f32 temporaries.
                    ESLatticeFit f;
                    std::vector<mx::array> parts = quantizeQ4Lattice(w, &f);
                    if (f.nearFrac() >= kLatticeAccept) {
                        latticeExactTensors++;
                        if (isEmbed) embedBitsOut = 4;
                        if (isPle)   pleBitsOut   = 4;
                    } else {
                        latticeFallbackTensors++;
                        fallbackNames.push_back(name);
                        parts = mx::quantize(w, groupSize, b);
                    }
                    latticeAll.total += f.total; latticeAll.exact += f.exact; latticeAll.near += f.near;
                    latticeAll.maxAbsErr = std::max(latticeAll.maxAbsErr, f.maxAbsErr);
                    mx::eval(parts);
                    out.emplace(name, parts[0]);
                    out.emplace(name + ".scales", parts[1]);
                    out.emplace(name + ".biases", parts[2]);
                } else if (b > 0) {
                    std::vector<mx::array> parts = mx::quantize(w, groupSize, b);  // {w_q, scales, biases}
                    out.emplace(name, parts[0]);
                    out.emplace(name + ".scales", parts[1]);
                    out.emplace(name + ".biases", parts[2]);
                    toEval.push_back(parts[0]); toEval.push_back(parts[1]); toEval.push_back(parts[2]);
                } else {
                    out.emplace(name, w);
                    toEval.push_back(w);
                }
            }
            mx::eval(toEval);
        } catch (const std::exception & e) {
            return fail(std::string("weight load/quantize failed: ") + e.what());
        }
        if (opts.lattice) {
            std::printf("  lattice: %d tensors exact, %d fallback (affine g32); weights on-lattice "
                        "exact %.4f%%, within 1 ulp %.4f%%, max|err| %.3e\n",
                        latticeExactTensors, latticeFallbackTensors,
                        100.0 * latticeAll.exactFrac(), 100.0 * latticeAll.nearFrac(), latticeAll.maxAbsErr);
            for (const std::string & n : fallbackNames) std::printf("    fallback: %s\n", n.c_str());
            if (embedBitsOut != opts.embedBits)
                std::printf("  lattice: embed_tokens on the grid -> stored exact at 4 bits (embed_bits=4)\n");
            if (pleBitsOut != opts.pleBits)
                std::printf("  lattice: PLE table on the grid -> stored exact at 4 bits (ple_bits=4)\n");
        }

        // Assemble the package in a temp dir, then move it into place atomically.
        NSString * tmpRoot = [NSTemporaryDirectory() stringByAppendingPathComponent:
                              [@"apml-" stringByAppendingString:[[NSUUID UUID] UUIDString]]];
        NSString * variant = [NSString stringWithUTF8String:opts.variantId.c_str()];
        NSString * variantDir = [[tmpRoot stringByAppendingPathComponent:@"weights"]
                                 stringByAppendingPathComponent:variant];
        NSError * ferr = nil;
        if (![fm createDirectoryAtPath:variantDir withIntermediateDirectories:YES attributes:nil error:&ferr])
            return fail(std::string("mkdir temp package failed: ") + ferr.localizedDescription.UTF8String);

        // Weights.
        std::unordered_map<std::string, std::string> meta = {
            {"apertura.kind", "apertura-model"},
            {"apertura.bits", std::to_string(bits)},
            {"apertura.group_size", std::to_string(groupSize)},
            {"apertura.embed_bits", std::to_string(embedBitsOut)},
            {"apertura.ple_bits", std::to_string(pleBitsOut)},
        };
        if (opts.lattice) meta.emplace("apertura.lattice", "qat-int4-g32");
        std::string stPath = [[variantDir stringByAppendingPathComponent:@"model.safetensors"] UTF8String];
        try {
            mx::save_safetensors(stPath, out, meta);
        } catch (const std::exception & e) {
            return fail(std::string("save_safetensors failed: ") + e.what());
        }

        // quantization.json (alongside the weights).
        NSMutableDictionary * quant = [@{ @"scheme": @"mlx-affine",
                                          @"bits": @(bits),
                                          @"group_size": @(groupSize),
                                          @"embed_bits": @(embedBitsOut),
                                          @"ple_bits": @(pleBitsOut) } mutableCopy];
        if (opts.lattice) {
            // Provenance of the exact recipe: the scales ARE the trained q4_0 steps. Loaders
            // ignore these keys (the tensors are ordinary mlx-affine g32).
            quant[@"lattice"] = @"qat-int4-g32";
            quant[@"lattice_exact_tensors"]    = @(latticeExactTensors);
            quant[@"lattice_fallback_tensors"] = @(latticeFallbackTensors);
            quant[@"lattice_exact_weights_pct"] = @(100.0 * latticeAll.exactFrac());
            quant[@"lattice_near_weights_pct"]  = @(100.0 * latticeAll.nearFrac());
        }
        [[NSJSONSerialization dataWithJSONObject:quant options:NSJSONWritingPrettyPrinted error:nil]
            writeToFile:[variantDir stringByAppendingPathComponent:@"quantization.json"] atomically:YES];

        // manifest.json (the self-describing trust anchor).
        NSMutableDictionary * variantEntry = [@{
            @"id": variant, @"runtime": @"mlx",
            @"path": [@"weights/" stringByAppendingString:variant],
            @"precision": [NSString stringWithFormat:@"q%d", bits],
            @"quantization": quant,
            @"files": @[@"model.safetensors"],
        } mutableCopy];
        NSMutableDictionary * manifest = [@{
            @"format_version": @1,
            @"kind": @"apertura-model",
            @"architecture": architecture,
            @"config": @"config.json",
            @"tokenizer": @{ @"file": @"tokenizer.json", @"kind": @"huggingface-tokenizers" },
            @"source": @{ @"model_id": [NSString stringWithUTF8String:opts.sourceModelId.c_str()],
                          @"revision": [NSString stringWithUTF8String:opts.sourceRevision.c_str()] },
            @"default_variant": variant,
            @"variants": @[variantEntry],
        } mutableCopy];
        if ([fm fileExistsAtPath:[dir stringByAppendingPathComponent:@"chat_template.jinja"]])
            manifest[@"chat_template"] = @"chat_template.jinja";
        NSData * manifestData = [NSJSONSerialization dataWithJSONObject:manifest
                                                              options:NSJSONWritingPrettyPrinted error:&ferr];
        if (!manifestData) return fail("manifest serialization failed");
        [manifestData writeToFile:[tmpRoot stringByAppendingPathComponent:@"manifest.json"] atomically:YES];

        // Copy the natural-format auxiliary members.
        octCopyIfPresent(fm, dir, tmpRoot, @"config.json");
        octCopyIfPresent(fm, dir, tmpRoot, @"tokenizer.json");
        octCopyIfPresent(fm, dir, tmpRoot, @"chat_template.jinja");
        octCopyIfPresent(fm, dir, tmpRoot, @"generation_config.json");

        // Move into place atomically (replace if a package already exists there).
        NSString * outPath = [NSString stringWithUTF8String:outPackagePath.c_str()];
        NSURL * tmpURL = [NSURL fileURLWithPath:tmpRoot];
        NSURL * outURL = [NSURL fileURLWithPath:outPath];
        [fm createDirectoryAtPath:[outPath stringByDeletingLastPathComponent]
            withIntermediateDirectories:YES attributes:nil error:nil];
        if ([fm fileExistsAtPath:outPath]) {
            if (![fm replaceItemAtURL:outURL withItemAtURL:tmpURL backupItemName:nil
                              options:0 resultingItemURL:nil error:&ferr])
                return fail(std::string("atomic replace failed: ") + ferr.localizedDescription.UTF8String);
        } else if (![fm moveItemAtURL:tmpURL toURL:outURL error:&ferr]) {
            return fail(std::string("move into place failed: ") + ferr.localizedDescription.UTF8String);
        }
        return true;
    }
}

}  // namespace es
