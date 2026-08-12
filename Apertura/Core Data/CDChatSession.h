//
//  CDChatSession.h
//  Apertura
//
//  Created by Kolja Wawrowsky on 8/6/26.
//
//  One saved conversation. The chat itself lives in `transcriptJSON` as a VERSIONED JSON
//  DOCUMENT rather than a keyed archive: a transcript stays readable with `sqlite3` and
//  `jq`, survives class renames, and can be handed to anything that speaks JSON. Roles
//  and content kinds travel as strings for the same reason — APRole's integer values are
//  an implementation detail, and a transcript written today must still read correctly if
//  the enum ever gains a case in the middle.
//
//  The transcript is the app-persistable source of truth (see APSession.transcript):
//  capture it here, and a relaunch can replay prime + turns into a fresh session.
//
//  Attributes come from the model via CDChatSession+CoreDataProperties.h (imported at the
//  bottom — the standard Category/Extension codegen shape). This class adds the JSON
//  codec, the APSession bridge, and the identifier/timestamp bookkeeping.

#import <Foundation/Foundation.h>
#import <CoreData/CoreData.h>

@class APMessage, APModel, APSession;

NS_ASSUME_NONNULL_BEGIN

/// Values for `backend` — the same two strings the app persists under "AperturaBackend",
/// so the column and the user default never drift apart.
FOUNDATION_EXPORT NSString * const CDChatSessionBackendLocal;    // @"local"
FOUNDATION_EXPORT NSString * const CDChatSessionBackendGoogle;   // @"google"

/// The envelope version written to `transcriptSchemaVersion` (and to the document's
/// "version" key). 0 means nothing has been written yet; a stored value GREATER than this
/// constant means the row came from a newer build — its transcript still decodes, but
/// content kinds this build cannot represent are dropped on re-save (see `messages`).
FOUNDATION_EXPORT const int16_t CDChatSessionCurrentSchemaVersion;

@interface CDChatSession : NSManagedObject

#pragma mark - Lifecycle

/// Insert an empty session. `identifier`, `dateCreated`, and `dateModified` are seeded.
+ (instancetype)insertInContext:(NSManagedObjectContext *)context;

/// All sessions, most recently modified first — for a sessions list.
+ (NSFetchRequest<CDChatSession *> *)recentSessionsFetchRequest;

/// The session with this `identifier`, or nil if there is none (or on error).
+ (nullable instancetype)sessionWithIdentifier:(NSUUID *)identifier
                                     inContext:(NSManagedObjectContext *)context
                                         error:(NSError **)error;

#pragma mark - The chat

/// The conversation. Decoded from `transcriptJSON` on first access and cached until the
/// data changes; setting it re-encodes and updates `messageCount` and
/// `transcriptSchemaVersion`. Never nil — an unwritten transcript reads as empty.
///
/// Writing `transcriptJSON` directly is supported too (import, repair): the decoded cache
/// drops, and the two denormalized columns are recomputed from the document itself — so
/// `messageCount` and `transcriptSchemaVersion` always describe what is actually stored.
///
/// v1 carries text parts only. APContent reserves image and audio kinds but cannot yet
/// construct them, so parts of those kinds decode to nothing; writing back a transcript
/// that contained them therefore loses them. Check `transcriptSchemaVersion` against
/// CDChatSessionCurrentSchemaVersion before mutating a row a newer build wrote.
@property (nonatomic, copy) NSArray<APMessage *> *messages;

/// Append one turn (encodes immediately; nil is ignored).
- (void)appendMessage:(APMessage *)message;

/// The stored document as text — exactly the bytes on disk, nil when empty. For export,
/// diffing, and seeing what actually got saved.
@property (nullable, nonatomic, readonly, copy) NSString *transcriptJSONString;

#pragma mark - Capture

/// Snapshot a live conversation: transcript, context token count, reasoning flag, and the
/// backend it ran on (plus `modelIdentifier` for the remote backend, which knows its own
/// model name). Fills in `title` from the first user turn if the session has none yet.
///
/// SYSTEM messages are deliberately left out of `messages`: a session's transcript includes
/// its primed standing prefix, and copying that here would duplicate the persona already in
/// `personaText` into every row. To resume, prime with `personaText` followed by `messages`.
/// (Setting `messages` yourself stores exactly what you pass — the filtering is capture's.)
- (void)captureSession:(APSession *)session;

/// Same, plus the model identifier — the local backend does not expose its APModel, so
/// pass the one you loaded. A non-nil `model` wins over the backend's own name.
- (void)captureSession:(APSession *)session model:(nullable APModel *)model;

/// Record the persona that was primed into this session, hashing it into `personaSHA256`
/// so a later replay can tell whether the persona file changed underneath it.
- (void)recordPersonaText:(nullable NSString *)text atPath:(nullable NSURL *)path;

@end

NS_ASSUME_NONNULL_END

#import "CDChatSession+CoreDataProperties.h"
