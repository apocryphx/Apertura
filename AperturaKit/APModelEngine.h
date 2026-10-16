//  APModelEngine — PROJECT header (never public). Objective-C++ ONLY: the engine SPI
//  behind APModel for the facade's .mm files — the es:: objects, the engine thread, and
//  the identity bytes cache fingerprints hash. Pure-ObjC files never see this header
//  (APResponseInternal.h and APInternal.h carry the C++-free SPI).
#pragma once
#import "APModel.h"

#ifndef __cplusplus
#error "APModelEngine.h is Objective-C++ only: include it from .mm files"
#endif
#include "ESGemma4TextForCausalLM.h"
#include "ESChatTemplate.h"
#include "ESTokenizer.h"
#include "ESGemma4VisionModel.h"

NS_ASSUME_NONNULL_BEGIN

@interface APModel (Engine)

/// Enqueue work on the model's dedicated engine thread (MLX streams are per-thread;
/// ALL engine calls must run here). One runner per model — cross-session serialization.
- (void)performOnEngine:(void (^)(void))block;

/// Same, blocking the caller until `block` has run — the synchronous facade entry points
/// (load, tokenization, session reset). Never call it FROM the engine thread.
- (void)performOnEngineAndWait:(void (^)(void))block;

- (APModelConfiguration *)internalConfiguration;
- (unsigned long long)internalWeightBytes;
/// The bytes that identify this loaded model for cache fingerprints: identifier (UTF-8),
/// weight byte count (8 bytes) and head precision (8 bytes). Persona snapshots and
/// session checkpoints both hash exactly these, so one definition keeps them in step.
- (NSData *)identityData;
- (es::ESGemma4TextForCausalLM *)internalLM;
/// The vision tower + multimodal embedder; NULL when the weights carry none (hasVisionTower).
- (nullable es::ESGemma4VisionModel *)internalVision;
- (es::ESTokenizer *)internalTokenizer;
- (es::ESChatTemplate *)internalTemplate;
- (const es::ESModelConfig *)internalConfig;

@end

NS_ASSUME_NONNULL_END
