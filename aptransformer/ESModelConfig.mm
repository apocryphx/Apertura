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
        c.padTokenId        = (int) numForKey(tc, @"pad_token_id", c.padTokenId);
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
        id bidir = tc[@"use_bidirectional_attention"];
        c.useBidirectionalVisionAttention = [bidir isKindOfClass:[NSString class]] && [bidir isEqualToString:@"vision"];

        // vision_config (+ root-level placeholder ids) — Gemma4ForConditionalGeneration checkpoints.
        NSDictionary * vc = root[@"vision_config"];
        if ([vc isKindOfClass:[NSDictionary class]]) {
            ESVisionConfig & v = c.vision;
            v.present               = true;
            NSString * vmt = vc[@"model_type"];
            v.kind = ([vmt isKindOfClass:[NSString class]] && [vmt isEqualToString:@"gemma4_unified_vision"])
                         ? ESVisionKind::unified : ESVisionKind::tower;
            v.mmEmbedDim            = (int) numForKey(vc, @"mm_embed_dim", v.mmEmbedDim);
            v.mmPosembSize          = (int) numForKey(vc, @"mm_posemb_size", v.mmPosembSize);
            v.outputProjDims        = (int) numForKey(vc, @"output_proj_dims", v.outputProjDims);
            v.hiddenSize            = (int) numForKey(vc, @"hidden_size", v.hiddenSize);
            v.numHiddenLayers       = (int) numForKey(vc, @"num_hidden_layers", v.numHiddenLayers);
            v.numAttentionHeads     = (int) numForKey(vc, @"num_attention_heads", v.numAttentionHeads);
            v.numKeyValueHeads      = (int) numForKey(vc, @"num_key_value_heads", v.numKeyValueHeads);
            v.headDim               = (int) numForKey(vc, @"head_dim", v.headDim);
            v.intermediateSize      = (int) numForKey(vc, @"intermediate_size", v.intermediateSize);
            v.patchSize             = (int) numForKey(vc, @"patch_size", v.patchSize);
            v.poolingKernelSize     = (int) numForKey(vc, @"pooling_kernel_size", v.poolingKernelSize);
            v.positionEmbeddingSize = (int) numForKey(vc, @"position_embedding_size", v.positionEmbeddingSize);
            v.maxSoftTokens         = (int) numForKey(vc, @"default_output_length", v.maxSoftTokens);
            v.rmsNormEps            = (float) numForKey(vc, @"rms_norm_eps", v.rmsNormEps);
            v.useClippedLinears     = numForKey(vc, @"use_clipped_linears", 1.0) != 0.0;
            v.standardize           = numForKey(vc, @"standardize", 0.0) != 0.0;
            NSDictionary * vrp = vc[@"rope_parameters"];
            if ([vrp isKindOfClass:[NSDictionary class]])
                v.ropeTheta = (float) numForKey(vrp, @"rope_theta", v.ropeTheta);
            v.imageTokenId = (int) numForKey(root, @"image_token_id", v.imageTokenId);
            v.boiTokenId   = (int) numForKey(root, @"boi_token_id", v.boiTokenId);
            v.eoiTokenId   = (int) numForKey(root, @"eoi_token_id", v.eoiTokenId);
            v.maxSoftTokens = (int) numForKey(root, @"vision_soft_tokens_per_image", v.maxSoftTokens);
        }
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
        c.profile = ESExecutionProfile::forConfig(c);
        c.prefillChunk = c.profile.prefillChunk;
        return c;
    }
}

float ESModelConfig::embedScale() const { return std::sqrt((float) hiddenSize); }

// Family detection from the architecture fields, then that family's execution policy. The
// numbers and their measurements are documented field by field in ESExecutionProfile.h.
ESExecutionProfile ESExecutionProfile::forConfig(const ESModelConfig & c) {
    ESExecutionProfile p;
    if (c.numKvSharedLayers > 0) {
        p.family = ESModelFamily::elastic;
        p.prefillChunk = 1024;
        p.sharedKVPrefillSkip = true;
    } else if (c.enableMoeBlock || c.numExperts > 0) {
        p.family = ESModelFamily::moe;
        p.moeSortMinPairs = 64;
    }
    return p;
}

}  // namespace es
