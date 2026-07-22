//  APInternal — PROJECT header (never public). Objective-C++ only: exposes the engine
//  objects to facade .mm files and private initializers/testing SPI to the gate.
#pragma once
#import "APModel.h"
#import "APResponse.h"
#import "APSession.h"

#ifdef __cplusplus
#include "ESGemma4TextForCausalLM.h"
#include "ESChatTemplate.h"
#include "ESTokenizer.h"

@interface APModel (Internal)
/// Enqueue work on the model's dedicated engine thread (MLX streams are per-thread;
/// ALL engine calls must run here). One runner per model — cross-session serialization.
- (void)performOnEngine:(void (^)(void))block;
- (APModelConfiguration *)internalConfiguration;
- (es::ESGemma4TextForCausalLM *)internalLM;
- (es::ESTokenizer *)internalTokenizer;
- (es::ESChatTemplate *)internalTemplate;
- (const es::ESModelConfig *)internalConfig;
@end
#endif

@interface APResponseDelta (Internal)
- (instancetype)initWithText:(NSString *)text tokenCount:(NSInteger)tokenCount;
@end

@interface APResponseStats (Internal)
- (instancetype)initWithPromptTokens:(NSInteger)prompt responseTokens:(NSInteger)response
                    timeToFirstToken:(NSTimeInterval)ttft
                           prefillTPS:(double)prefillTPS decodeTPS:(double)decodeTPS;
@end

@interface APResponse (Internal)
- (instancetype)initWithMessage:(APMessage *)message
                   finishReason:(APFinishReason)reason
                          stats:(APResponseStats *)stats;
@end

@interface APResponseTask (Internal)
- (instancetype)initWithProgress:(NSProgress *)progress;
- (BOOL)isCancelled;
@end

@interface APSession (Testing)
/// Raw sampled token ids of the most recent completed response — SPI for the
/// --facade-verify byte-identity gate. Not product API.
@property (readonly, nullable) NSArray<NSNumber *> *lastResponseTokenIDsForTesting;
@end
