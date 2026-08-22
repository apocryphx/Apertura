//
//  CDPersona.h
//  Apertura
//
//  One custom GPT: a name and a single `body` — the full standing prefix, sections
//  separated by the "\n\n---\n\n" convention (the same joint the KV-snapshot prompt
//  families use; there is deliberately no "part" entity). `sha256` fingerprints the
//  body so snapshot/replay code can tell when it changed.
//
//  Version history is a bidirectional linked list ON THIS ENTITY: `previousVersion` /
//  `nextVersion` (inverses). The CURRENT persona is the head — `nextVersion == nil` —
//  and keeps its stable `identifier` forever, so sessions and KV snapshots stay pointed
//  at it across edits. Editing snapshots the old state into a new history row spliced
//  in behind the head (`snapshotBeforeEditWithNote:author:`), then mutates the head.
//  Deleting a head cascades down its whole chain.
//
//  Attributes come from the model via CDPersona+CoreDataProperties.h (imported at the
//  bottom — the standard Category/Extension codegen shape).

#import <Foundation/Foundation.h>
#import <CoreData/CoreData.h>

@class CDChatSession;   // relationship type in the generated properties header

NS_ASSUME_NONNULL_BEGIN

/// The section joint inside `body` — one persona, many documents, one string.
FOUNDATION_EXPORT NSString * const CDPersonaSectionSeparator;   // @"\n\n---\n\n"

@interface CDPersona : NSManagedObject

#pragma mark - Lifecycle

/// Insert an empty persona. `identifier`, `dateCreated`, and `dateModified` are seeded.
+ (instancetype)insertInContext:(NSManagedObjectContext *)context;

/// The CURRENT personas (list heads only — history rows are excluded), sorted by name.
+ (NSFetchRequest<CDPersona *> *)currentPersonasFetchRequest;

/// The persona with this `identifier`, or nil if there is none (or on error).
+ (nullable instancetype)personaWithIdentifier:(NSUUID *)identifier
                                     inContext:(NSManagedObjectContext *)context
                                         error:(NSError **)error;

/// Create a persona whose body is the given files joined IN ORDER with the section
/// separator — the one-step import for a decomposed persona (identity doc, archives,
/// exemplars…). nil on any read failure.
+ (nullable instancetype)importFromFileURLs:(NSArray<NSURL *> *)fileURLs
                                       name:(NSString *)name
                                  inContext:(NSManagedObjectContext *)context
                                      error:(NSError **)error;

#pragma mark - Editing

/// Archive-first discipline: freeze the CURRENT state into a new history row spliced in
/// behind this head, then return it. Call before every mutation of `body` — the head
/// keeps its identity, the history row keeps the old text. `note`/`author` describe the
/// coming edit (author: "user", "revise_persona", "record_legend", "mcp") and land in
/// the history row's `notes`.
- (CDPersona *)snapshotBeforeEditWithNote:(nullable NSString *)note
                                   author:(nullable NSString *)author;

/// Set `body`, recompute `sha256`, touch `dateModified` — the one mutation path.
- (void)updateBody:(nullable NSString *)body;

/// This version and every older one, newest first (self included). For history UI.
- (NSArray<CDPersona *> *)versionChain;

/// YES when this row is the current head (nextVersion == nil).
@property (nonatomic, readonly) BOOL isCurrentVersion;

@end

NS_ASSUME_NONNULL_END

#import "CDPersona+CoreDataProperties.h"
