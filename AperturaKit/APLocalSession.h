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

#pragma mark - Session checkpointing (one per device)

//  The device checkpoint persists a LIVE session's KV cache across launches, so a deep
//  conversation resumes in seconds instead of re-prefilling its whole history. Policy is
//  aggressively simple: exactly ONE checkpoint per device (in Application Support), keyed
//  by model identity + the owning session UUID; starting a new conversation deletes it.
//  The safetensors file holds the cache; a sidecar plist holds the session scalars
//  (pos/turnCount/openModelTurn/reasoning) that the fingerprint also binds.

/// The single device checkpoint file (…/Application Support/<bundle id>/session-checkpoint.safetensors).
+ (NSURL *)deviceCheckpointURL;

/// Delete the device checkpoint (both files). Call when the user starts a new conversation.
+ (void)removeDeviceCheckpoint;

/// The session UUID the stored checkpoint belongs to, iff its sidecar matches `model` —
/// a cheap peek (no engine work) for deciding whether a resume fast path exists.
+ (nullable NSUUID *)checkpointedSessionIDForModel:(APModel *)model;

/// Write the checkpoint for this live session (blocking; runs on the engine thread and
/// waits — a 60K-context cache is several GB, expect seconds). Requires at least one
/// primed/generated token. Returns NO (and leaves no partial checkpoint) on any failure.
- (BOOL)saveCheckpointForSessionID:(NSUUID *)sessionID;

/// Restore the device checkpoint into this FRESH session (pos must be 0) instead of
/// priming. `messages` seeds the transcript (persona + stored turns — what a full
/// re-prime would have appended) so later archiving sees the complete conversation.
/// The sidecar must match `sessionID`, the model, and this session's reasoning flag.
- (void)restoreCheckpointForSessionID:(NSUUID *)sessionID
                             messages:(NSArray<APMessage *> *)messages
                           completion:(void (^)(NSError *_Nullable error))completion;

@end

NS_ASSUME_NONNULL_END
