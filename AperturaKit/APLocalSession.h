//  APLocalSession — the on-device backend: one conversation over one persistent KV cache.
//
//  The engine's prefix cache (ESSession semantics underneath) is the DEFAULT behavior:
//  the persona/context is prefilled once at prime and every turn appends only its delta
//  (the measured 33.7x multi-turn win). The streaming loop mirrors the gated CLI
//  session path token-for-token; byte-identity is enforced by the --facade-verify gate.
//  primeWithMessages:cacheURL: honors the persistent KV-snapshot fast path.
//
//  Concurrency: engine work runs on the model's dedicated engine thread. Multiple
//  sessions may share one APModel (weights are shared; generation interleaves at token
//  granularity).
#import <AperturaKit/APSession.h>
#import <AperturaKit/APModel.h>

NS_ASSUME_NONNULL_BEGIN

@interface APLocalSession : APSession

- (instancetype)initWithModel:(APModel *)model;

@end

NS_ASSUME_NONNULL_END
