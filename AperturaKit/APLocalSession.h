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

#pragma mark - Session checkpointing

//  A checkpoint persists a LIVE session's KV cache across launches, so a deep
//  conversation resumes in seconds instead of re-prefilling its whole history. The
//  general form is URL-parameterized — keep any number of checkpoint files (e.g. one
//  per conversation at checkpoints/<sessionUUID>.safetensors); the historical single
//  device checkpoint remains as a convenience default via the device* methods.
//  Identity is model + session UUID + session scalars, never the file location, so
//  checkpoint files can be moved or renamed freely. The safetensors file holds the
//  cache; a sidecar plist (same path, .plist) holds the session scalars
//  (pos/turnCount/openModelTurn/reasoning) that the fingerprint also binds.
//  Checkpoint URLs must end in .safetensors (the engine appends the extension on save
//  but not on load).

/// Write the checkpoint for this live session to `checkpointURL` (async; the save runs
/// on the engine thread — a 60K-context cache is several GB, expect seconds —
/// `completion(saved)` is delivered on the callback queue when the files are on disk).
/// Requires at least one primed/generated token. On any failure the checkpoint at
/// `checkpointURL` is removed — never a partial checkpoint.
- (void)saveCheckpointToURL:(NSURL *)checkpointURL
                  sessionID:(NSUUID *)sessionID
                 completion:(void (^)(BOOL saved))completion;

/// Blocking variant of the URL save (waits on the engine-thread work).
- (BOOL)saveCheckpointToURL:(NSURL *)checkpointURL sessionID:(NSUUID *)sessionID;

/// Restore the checkpoint at `checkpointURL` into this FRESH session (pos must be 0)
/// instead of priming. `messages` seeds the transcript (persona + stored turns — what a
/// full re-prime would have appended) so later archiving sees the complete conversation.
/// The sidecar must match `sessionID`, the model, and this session's reasoning flag.
- (void)restoreCheckpointFromURL:(NSURL *)checkpointURL
                       sessionID:(NSUUID *)sessionID
                        messages:(NSArray<APMessage *> *)messages
                      completion:(void (^)(NSError *_Nullable error))completion;

/// The session UUID the checkpoint at `checkpointURL` belongs to, iff its sidecar
/// matches `model` — a cheap peek (no engine work) for deciding whether a resume fast
/// path exists.
+ (nullable NSUUID *)checkpointedSessionIDAtURL:(NSURL *)checkpointURL
                                       forModel:(APModel *)model;

/// Delete the checkpoint at `checkpointURL` (both files).
+ (void)removeCheckpointAtURL:(NSURL *)checkpointURL;

#pragma mark - Device checkpoint (convenience default: one fixed location)

/// The single device checkpoint file (…/Application Support/<bundle id>/session-checkpoint.safetensors).
+ (NSURL *)deviceCheckpointURL;

/// Delete the device checkpoint (both files).
+ (void)removeDeviceCheckpoint;

/// Peek at the device checkpoint — forwards to the URL form.
+ (nullable NSUUID *)checkpointedSessionIDForModel:(APModel *)model;

/// Save to the device checkpoint location (blocking) — forwards to the URL form.
- (BOOL)saveCheckpointForSessionID:(NSUUID *)sessionID;

/// Save to the device checkpoint location (async) — forwards to the URL form.
- (void)saveCheckpointForSessionID:(NSUUID *)sessionID
                        completion:(void (^)(BOOL saved))completion;

/// Restore from the device checkpoint location — forwards to the URL form.
- (void)restoreCheckpointForSessionID:(NSUUID *)sessionID
                             messages:(NSArray<APMessage *> *)messages
                           completion:(void (^)(NSError *_Nullable error))completion;

@end

NS_ASSUME_NONNULL_END
