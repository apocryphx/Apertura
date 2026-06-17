#include "ESModelConfig.h"

#import <Foundation/Foundation.h>
#include <cmath>
#include <stdexcept>

namespace es {

static double numForKey(NSDictionary * d, NSString * k, double dflt) {
    id v = d[k];
    return [v isKindOfClass:[NSNumber class]] ? [v doubleValue] : dflt;
}

ESModelConfig ESModelConfig::fromConfigJSON(const std::string & configJsonPath) {
    @autoreleasepool {
        NSString * path = [NSString stringWithUTF8String:configJsonPath.c_str()];
        NSData * data = [NSData dataWithContentsOfFile:path];
        if (!data) {
            throw std::runtime_error("ESModelConfig: cannot read " + configJsonPath);
        }
        NSError * err = nil;
        NSDictionary * root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
        if (!root || err) {
            throw std::runtime_error("ESModelConfig: bad JSON in " + configJsonPath);
        }
        NSDictionary * tc = root[@"text_config"];
        if (![tc isKindOfClass:[NSDictionary class]]) {
            throw std::runtime_error("ESModelConfig: missing text_config");
        }

        ESModelConfig c;
        c.hiddenSize        = (int) numForKey(tc, @"hidden_size", c.hiddenSize);
        c.numHiddenLayers   = (int) numForKey(tc, @"num_hidden_layers", c.numHiddenLayers);
        c.numAttentionHeads = (int) numForKey(tc, @"num_attention_heads", c.numAttentionHeads);
        c.numKeyValueHeads  = (int) numForKey(tc, @"num_key_value_heads", c.numKeyValueHeads);
        c.numGlobalKVHeads  = (int) numForKey(tc, @"num_global_key_value_heads", c.numGlobalKVHeads);
        c.headDim           = (int) numForKey(tc, @"head_dim", c.headDim);
        c.globalHeadDim     = (int) numForKey(tc, @"global_head_dim", c.globalHeadDim);
        c.intermediateSize  = (int) numForKey(tc, @"intermediate_size", c.intermediateSize);
        c.slidingWindow     = (int) numForKey(tc, @"sliding_window", c.slidingWindow);
        c.vocabSize         = (int) numForKey(tc, @"vocab_size", c.vocabSize);
        c.maxPositionEmbeddings = (int) numForKey(tc, @"max_position_embeddings", c.maxPositionEmbeddings);
        c.rmsNormEps            = (float) numForKey(tc, @"rms_norm_eps", c.rmsNormEps);
        c.finalLogitSoftcapping = (float) numForKey(tc, @"final_logit_softcapping", c.finalLogitSoftcapping);
        c.attentionKEqV     = numForKey(tc, @"attention_k_eq_v", 1.0) != 0.0;
        c.enableMoeBlock      = numForKey(tc, @"enable_moe_block", 0.0) != 0.0;
        c.numExperts          = (int) numForKey(tc, @"num_experts", 0);
        c.topKExperts         = (int) numForKey(tc, @"top_k_experts", 0);
        c.moeIntermediateSize = (int) numForKey(tc, @"moe_intermediate_size", 0);
        c.hiddenSizePerLayerInput = (int) numForKey(tc, @"hidden_size_per_layer_input", 0);
        c.vocabSizePerLayerInput  = (int) numForKey(tc, @"vocab_size_per_layer_input", 0);
        c.numKvSharedLayers       = (int) numForKey(tc, @"num_kv_shared_layers", 0);

        id tie = root[@"tie_word_embeddings"];
        if ([tie isKindOfClass:[NSNumber class]]) c.tieWordEmbeddings = [tie boolValue];

        // rope_parameters: {sliding_attention:{rope_theta}, full_attention:{rope_theta, partial_rotary_factor}}
        NSDictionary * rp = tc[@"rope_parameters"];
        if ([rp isKindOfClass:[NSDictionary class]]) {
            NSDictionary * sl = rp[@"sliding_attention"];
            NSDictionary * fl = rp[@"full_attention"];
            if ([sl isKindOfClass:[NSDictionary class]])
                c.ropeThetaLocal = (float) numForKey(sl, @"rope_theta", c.ropeThetaLocal);
            if ([fl isKindOfClass:[NSDictionary class]]) {
                c.ropeThetaGlobal = (float) numForKey(fl, @"rope_theta", c.ropeThetaGlobal);
                c.globalPartialRotaryFactor =
                    (float) numForKey(fl, @"partial_rotary_factor", c.globalPartialRotaryFactor);
            }
        }

        // layer_types: array of "sliding_attention" / "full_attention"
        NSArray * lt = tc[@"layer_types"];
        c.layerIsSliding.clear();
        if ([lt isKindOfClass:[NSArray class]]) {
            for (id e in lt) {
                bool sliding = [e isKindOfClass:[NSString class]] &&
                               [(NSString *) e isEqualToString:@"sliding_attention"];
                c.layerIsSliding.push_back(sliding);
            }
        }
        if ((int) c.layerIsSliding.size() != c.numHiddenLayers) {
            // Fallback: derive 5:1 sliding:full (full at idx 5,11,...).
            c.layerIsSliding.assign(c.numHiddenLayers, true);
            for (int i = 5; i < c.numHiddenLayers; i += 6) c.layerIsSliding[i] = false;
            c.layerIsSliding[c.numHiddenLayers - 1] = false;  // final always global
        }
        return c;
    }
}

float ESModelConfig::embedScale() const { return std::sqrt((float) hiddenSize); }

}  // namespace es
