//  APInternal — PROJECT header (never public). Pure Objective-C: the testing SPI the
//  gates and AperturaKitTests reach into (byte-identity checks need the raw token ids
//  and the budget arithmetic). Not product API.
//  Siblings: APModelEngine.h (ObjC++ engine SPI), APResponseInternal.h (private
//  initializers), APSessionSubclass.h (the protected backend surface).
#pragma once
#import "APLocalSession.h"

NS_ASSUME_NONNULL_BEGIN

#ifdef __cplusplus
extern "C" {
#endif
/// The response token budget for one turn: what fits between `position + turnTokenCount`
/// and `limit` (less the closing markers), capped by `requested` when it is > 0.
FOUNDATION_EXPORT NSInteger APLocalSessionResponseTokenBudget(NSInteger limit,
                                                              NSInteger position,
                                                              NSInteger turnTokenCount,
                                                              NSInteger requested);
#ifdef __cplusplus
}
#endif

@interface APLocalSession (Testing)
/// Raw sampled token ids of the most recent completed response — SPI for the
/// --facade-verify byte-identity gate.
@property (readonly, nullable) NSArray<NSNumber *> *lastResponseTokenIDsForTesting;
@end

NS_ASSUME_NONNULL_END
