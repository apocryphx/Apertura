#include "ESWeightLoader.h"

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
        return ESEmbedding(q.weight, q.scales, q.biases, w.bundleEmbedBits(), w.bundleGroupSize());
    }
    return ESEmbedding(w.get(name), quantEmbedBits, groupSize);
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
        [fm copyItemAtPath:src toPath:[dstDir stringByAppendingPathComponent:file] error:nil];
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

        // Load bf16 weights. The loader only needs computeDtype; quant fields are irrelevant here.
        ESModelConfig cfg;  // defaults: computeDtype == bfloat16
        std::unordered_map<std::string, mx::array> out;
        std::vector<mx::array> toEval;
        try {
            ESWeightLoader loader(modelDir, cfg);
            for (const auto & kv : loader.all()) {
                const std::string & name = kv.first;
                const mx::array & w = kv.second;
                int b = 0;
                if (name == "embed_tokens.weight") b = opts.embedBits;
                else if (octIsLayerProjQuant(name)) b = opts.bits;

                if (b > 0) {
                    std::vector<mx::array> parts = mx::quantize(w, opts.groupSize, b);  // {w_q, scales, biases}
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
            {"apertura.bits", std::to_string(opts.bits)},
            {"apertura.group_size", std::to_string(opts.groupSize)},
            {"apertura.embed_bits", std::to_string(opts.embedBits)},
        };
        std::string stPath = [[variantDir stringByAppendingPathComponent:@"model.safetensors"] UTF8String];
        try {
            mx::save_safetensors(stPath, out, meta);
        } catch (const std::exception & e) {
            return fail(std::string("save_safetensors failed: ") + e.what());
        }

        // quantization.json (alongside the weights).
        NSDictionary * quant = @{ @"scheme": @"mlx-affine",
                                  @"bits": @(opts.bits),
                                  @"group_size": @(opts.groupSize),
                                  @"embed_bits": @(opts.embedBits) };
        [[NSJSONSerialization dataWithJSONObject:quant options:NSJSONWritingPrettyPrinted error:nil]
            writeToFile:[variantDir stringByAppendingPathComponent:@"quantization.json"] atomically:YES];

        // manifest.json (the self-describing trust anchor).
        NSMutableDictionary * variantEntry = [@{
            @"id": variant, @"runtime": @"mlx",
            @"path": [@"weights/" stringByAppendingString:variant],
            @"precision": [NSString stringWithFormat:@"q%d", opts.bits],
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
