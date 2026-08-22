//
//  APPersistence.h
//  Apertura
//
//  The one Core Data stack, shared by every process that opens the store — the app and
//  apertura-mcp compile this same file. Plain NSPersistentContainer (the CloudKit
//  container never had entitlements, so sync never worked; the store's leftover CloudKit
//  metadata is inert), with three load-bearing options:
//
//   • NSPersistentHistoryTrackingKey = YES — MANDATORY, one-way door: the store was
//     first opened by NSPersistentCloudKitContainer with history tracking on, and a
//     container that omits the flag fails to open it. It is also what makes
//     cross-process change observation work.
//   • Remote-change notifications — the app refreshes its lists when the MCP process
//     writes, and vice versa.
//   • viewContext.automaticallyMergesChangesFromParent = YES.
//
//  The store stays at NSPersistentContainer's default location for the "Apertura"
//  model — no relocation, no data risk; both processes resolve the same path.

#import <Foundation/Foundation.h>
#import <CoreData/CoreData.h>

NS_ASSUME_NONNULL_BEGIN

@interface APPersistence : NSObject

/// The shared container. Created on first use; the store loads synchronously enough to
/// use immediately (errors are logged, and the container is returned regardless — a
/// chat app must never take the conversation down with a persistence error).
+ (NSPersistentContainer *)sharedContainer;

/// The resolved on-disk store URL (for logging and CLI tooling).
+ (NSURL *)storeURL;

@end

NS_ASSUME_NONNULL_END
