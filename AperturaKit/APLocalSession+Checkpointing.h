//  APLocalSession+Checkpointing — PROJECT header (never public), Objective-C++ only.
//
//  The checkpoint category (declared publicly in APLocalSession.h, implemented in
//  APLocalSession+Checkpointing.mm) shares the session's engine-side state with the main
//  implementation; this class extension is that shared state. Rules: the KV cache is
//  touched ONLY on the engine thread; the scalars that public getters read are guarded by
//  @synchronized(self). Nothing else imports this header.
#pragma once
#import "APLocalSession.h"
#import "APSessionSubclass.h"
#import "APInternal.h"

#include "ESKVCache.h"
#include <memory>

NS_ASSUME_NONNULL_BEGIN

@interface APLocalSession () {
  @package
    APModel * _model;
    std::unique_ptr<es::ESKVCache> _cache;   // touched ONLY on the engine thread
    int  _pos;
    int  _turnCount;
    BOOL _openModelTurn;     // last response did not close its turn (cancel/max-tokens)
    BOOL _warnedNearFull;
    NSArray<NSNumber *> * _Nullable _lastIds;
}
@end

/// The running config's global-cache mode, for mode-aware snapshot restore: snapshots
/// persisted under the OTHER raw mode convert during restore (q8 -> bf16 dequantizes,
/// bf16 -> q8 quantizes), so flipping APModelConfiguration.globalKVCacheMode never
/// strands a snapshot or checkpoint; composite <-> raw mismatches re-prime instead.
static inline es::ESKVCache::RawMode APLocalSessionRawMode(APModel * model) {
    const es::ESModelConfig * c = [model internalConfig];
    if (!c->rawKV) return es::ESKVCache::RawMode::composite;
    return c->rawKVQ8 ? es::ESKVCache::RawMode::rawQ8 : es::ESKVCache::RawMode::raw;
}

NS_ASSUME_NONNULL_END
