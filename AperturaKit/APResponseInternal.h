//  APResponseInternal — PROJECT header (never public). Pure Objective-C: the private
//  initializers of the response value types, for the backends that build them
//  (APLocalSession.mm, APGoogleSession.m) and for APResponse.m itself.
#pragma once
#import "APResponse.h"

NS_ASSUME_NONNULL_BEGIN

@interface APResponseDelta (Internal)
- (instancetype)initWithText:(NSString *)text tokenCount:(NSInteger)tokenCount
                   isThought:(BOOL)isThought;
@end

@interface APResponseStats (Internal)
- (instancetype)initWithPromptTokens:(NSInteger)prompt responseTokens:(NSInteger)response
                    timeToFirstToken:(NSTimeInterval)ttft
                           prefillTPS:(double)prefillTPS decodeTPS:(double)decodeTPS;
@end

@interface APResponse (Internal)
- (instancetype)initWithMessage:(APMessage *)message
                   finishReason:(APFinishReason)reason
                          stats:(APResponseStats *)stats
                      reasoning:(nullable NSString *)reasoning
              executedToolNames:(nullable NSArray<NSString *> *)executedToolNames;
@end

@interface APResponseTask (Internal)
- (instancetype)initWithProgress:(NSProgress *)progress;
/// Polled per token (local) / per part (remote); set by -cancel.
- (BOOL)isCancelled;
@end

NS_ASSUME_NONNULL_END
