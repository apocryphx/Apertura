#include "ESWeightLoader.h"

#import <Foundation/Foundation.h>
#include <set>
#include <stdexcept>

namespace es {

static const std::string kTextPrefix = "model.language_model.";

ESWeightLoader::ESWeightLoader(const std::string & modelDir, const ESModelConfig & config) {
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

}  // namespace es
