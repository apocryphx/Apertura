//  APCheckpointStore — identity, files, and policy for the device session checkpoint.
//
//  INTERNAL header (not in the framework's public set). The store owns everything about
//  the checkpoint EXCEPT the engine work: where the one-per-device files live
//  (Application Support), the sidecar plist codec, the SHA-256 fingerprint that binds
//  sidecar scalars to the safetensors content, and the peek/remove policy operations.
//  APLocalSession keeps the two operations that must touch the engine thread and the
//  KV cache (save/restore) and delegates the rest here; its public checkpoint class
//  methods forward to this store. Stateless — class methods only.
#import <Foundation/Foundation.h>
#import <AperturaKit/APModel.h>

NS_ASSUME_NONNULL_BEGIN

@interface APCheckpointStore : NSObject

/// The single device checkpoint file. NOTE: must end in .safetensors —
/// mx::save_safetensors silently appends the extension on save while load does not.
+ (NSURL *)deviceCheckpointURL;

/// The sidecar plist holding {sessionID, modelIdentifier, pos, turnCount,
/// openModelTurn, reasoningEnabled} — every value re-bound by the fingerprint.
+ (NSURL *)sidecarURL;

/// Delete both files. Also the failure path: never leave a partial checkpoint.
+ (void)removeDeviceCheckpoint;

/// The session UUID the stored checkpoint belongs to, iff its sidecar matches `model`
/// and the safetensors file exists — a cheap peek, no engine work.
+ (nullable NSUUID *)checkpointedSessionIDForModel:(APModel *)model;

/// Checkpoint validity key: format version, model identity, session UUID, and the
/// session scalars. Restore recomputes it FROM sidecar values, so a stale or edited
/// sidecar simply fails the safetensors fingerprint match — the sidecar is never
/// trusted on its own. Hex SHA-256.
+ (NSString *)fingerprintForModel:(APModel *)model
                        sessionID:(NSUUID *)sessionID
                              pos:(int)pos
                        turnCount:(int)turnCount
                    openModelTurn:(BOOL)openModelTurn
                        reasoning:(BOOL)reasoning;

/// Write the sidecar after a successful safetensors save. Returns NO on I/O failure.
+ (BOOL)writeSidecarForSessionID:(NSUUID *)sessionID
                           model:(APModel *)model
                             pos:(int)pos
                       turnCount:(int)turnCount
                   openModelTurn:(BOOL)openModelTurn
                       reasoning:(BOOL)reasoning;

/// The sidecar, iff it matches `sessionID` + `model` + `reasoning`; nil otherwise.
/// Keys: pos, turnCount, openModelTurn (NSNumber).
+ (nullable NSDictionary *)sidecarMatchingSessionID:(NSUUID *)sessionID
                                              model:(APModel *)model
                                          reasoning:(BOOL)reasoning;

@end

NS_ASSUME_NONNULL_END
