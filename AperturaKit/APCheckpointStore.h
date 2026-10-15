//  APCheckpointStore — identity, files, and policy for session KV checkpoints.
//
//  INTERNAL header (not in the framework's public set). The store owns everything about
//  a checkpoint EXCEPT the engine work: the sidecar plist codec, the SHA-256 fingerprint
//  that binds sidecar scalars to the safetensors content, and the peek/remove policy
//  operations. Checkpoints are URL-parameterized — a caller may keep any number of
//  checkpoint files (e.g. one per conversation); the historical one-per-device file is
//  just `deviceCheckpointURL` (APLocalSession's device* methods are the URL forms at
//  that location). The fingerprint deliberately excludes the file location, so
//  checkpoint files can be moved or renamed freely. APLocalSession(Checkpointing) keeps
//  the two operations that must touch the engine thread and the KV cache (save/restore)
//  and delegates the rest here. Stateless — class methods only.
#import <Foundation/Foundation.h>
#import <AperturaKit/APModel.h>

NS_ASSUME_NONNULL_BEGIN

@interface APCheckpointStore : NSObject

#pragma mark - URL-parameterized checkpoints (the general form)

/// The sidecar plist for a checkpoint file: same path with the extension replaced by
/// .plist. NOTE: the checkpoint URL itself must end in .safetensors —
/// mx::save_safetensors silently appends the extension on save while load does not.
+ (NSURL *)sidecarURLForCheckpointURL:(NSURL *)checkpointURL;

/// Delete a checkpoint (both files). Also the failure path: never leave a partial
/// checkpoint.
+ (void)removeCheckpointAtURL:(NSURL *)checkpointURL;

/// The session UUID the checkpoint at `checkpointURL` belongs to, iff its sidecar
/// matches `model` and the safetensors file exists — a cheap peek, no engine work.
+ (nullable NSUUID *)checkpointedSessionIDAtURL:(NSURL *)checkpointURL
                                       forModel:(APModel *)model;

/// Write the sidecar after a successful safetensors save. Returns NO on I/O failure.
+ (BOOL)writeSidecarForCheckpointURL:(NSURL *)checkpointURL
                           sessionID:(NSUUID *)sessionID
                               model:(APModel *)model
                                 pos:(int)pos
                           turnCount:(int)turnCount
                       openModelTurn:(BOOL)openModelTurn
                           reasoning:(BOOL)reasoning;

/// The sidecar at `checkpointURL`, iff it matches `sessionID` + `model` + `reasoning`;
/// nil otherwise. Keys: pos, turnCount, openModelTurn (NSNumber).
+ (nullable NSDictionary *)sidecarAtCheckpointURL:(NSURL *)checkpointURL
                                matchingSessionID:(NSUUID *)sessionID
                                            model:(APModel *)model
                                        reasoning:(BOOL)reasoning;

/// Checkpoint validity key: format version, the model's identity bytes (-[APModel
/// identityData], shared with persona-snapshot fingerprints), session UUID, and the
/// session scalars. Restore recomputes it FROM sidecar values, so a stale or edited
/// sidecar simply fails the safetensors fingerprint match — the sidecar is never
/// trusted on its own. Hex SHA-256. Location-independent by design.
+ (NSString *)fingerprintForModel:(APModel *)model
                        sessionID:(NSUUID *)sessionID
                              pos:(int)pos
                        turnCount:(int)turnCount
                    openModelTurn:(BOOL)openModelTurn
                        reasoning:(BOOL)reasoning;

#pragma mark - Device checkpoint (convenience default: one fixed location)

/// The historical single device checkpoint file
/// (…/Application Support/<bundle id>/session-checkpoint.safetensors).
+ (NSURL *)deviceCheckpointURL;

@end

NS_ASSUME_NONNULL_END
