//  APSessionSubclass — PROJECT header (never public): the protected base-class surface
//  concrete backends build on. Everything here is implemented once in APSession.m so
//  every backend shares one transcript, tool registry, and delivery discipline.
#pragma once
#import "APSession.h"

NS_ASSUME_NONNULL_BEGIN

@interface APSession (Subclass)

/// Deliver a callback block on `callbackQueue` (falls back to the main queue).
- (void)deliver:(dispatch_block_t)block;

/// Transcript mutation (thread-safe; the readonly `transcript` property snapshots).
- (void)appendTranscriptMessages:(NSArray<APMessage *> *)messages;
- (void)clearTranscript;

/// Registered tools sorted by name — the stable order every backend advertises
/// (local: stable prime ids and snapshot fingerprints; remote: stable request bodies).
- (NSArray<id<APTool>> *)toolsSortedByName;

/// Record whether the most recent prime restored from a KV snapshot (local backend;
/// backends without snapshots leave the default NO).
- (void)noteLastPrimeRestoredFromSnapshot:(BOOL)restored;

@end

NS_ASSUME_NONNULL_END
