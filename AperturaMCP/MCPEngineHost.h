//
//  MCPEngineHost.h
//  apertura-mcp
//
//  The standalone engine host: its own APModel (never connected to a running app — two
//  hosts on one machine means two weight loads; quit the app when benchmarking the
//  31B), a map of live APLocalSessions, and the SAME Core Data store the app uses
//  (personas, sessions, transcripts) through APPersistence. All Core Data work runs on
//  a background context (the process's main thread is the stdio loop — the view
//  context would deadlock). Session callbacks land on a private serial queue for the
//  same reason; tool handlers bridge async engine completions with semaphores waited
//  on the stdio thread, never on the engine thread.
//
//  CAVEAT (also in the plan): never run the app and apertura-mcp against the same
//  session's CHECKPOINT concurrently — files are last-writer-wins.

#import <Foundation/Foundation.h>
#import "MCPToolRegistry.h"

NS_ASSUME_NONNULL_BEGIN

@interface MCPEngineHost : NSObject

/// Register every tool into the registry. Tools lazy-load the model on first need.
- (void)registerToolsInto:(MCPToolRegistry *)registry;

/// Best-effort persistence flush on shutdown (EOF on stdin).
- (void)shutdown;

@end

NS_ASSUME_NONNULL_END
