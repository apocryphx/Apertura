//  APLocalSession+Checkpointing — the session's checkpoint surface (see APLocalSession.h).
//
//  Two operations touch the engine thread and the KV cache (save/restore); everything
//  else — identity, sidecar codec, peek/remove policy — is APCheckpointStore. The URL
//  forms are the general case; the device methods are the same operations at the store's
//  one fixed location (the app and the MCP server use both).
#import "APLocalSession+Checkpointing.h"
#import "APCheckpointStore.h"
#import "APError.h"

#include <string>

static NSError * apSessionError(APErrorCode code, NSString * message) {
    return [NSError errorWithDomain:APErrorDomain code:code
                           userInfo:@{ NSLocalizedDescriptionKey : message }];
}

@implementation APLocalSession (Checkpointing)

- (void)saveCheckpointToURL:(NSURL *)checkpointURL
                  sessionID:(NSUUID *)sessionID
                 completion:(void (^)(BOOL saved))completion {
    [_model performOnEngine:^{
        BOOL ok = NO;
        int pos, turns; BOOL open;
        @synchronized(self) { pos = self->_pos; turns = self->_turnCount; open = self->_openModelTurn; }
        if (pos > 0) {
            NSString * fp = [APCheckpointStore fingerprintForModel:self->_model sessionID:sessionID
                                                               pos:pos turnCount:turns
                                                     openModelTurn:open
                                                         reasoning:self.reasoningEnabled];
            if (self->_cache->saveSnapshot(
                    std::string(checkpointURL.path.UTF8String),
                    std::string(fp.UTF8String), pos)) {
                ok = [APCheckpointStore writeSidecarForCheckpointURL:checkpointURL
                                                           sessionID:sessionID model:self->_model
                                                                 pos:pos turnCount:turns
                                                       openModelTurn:open
                                                           reasoning:self.reasoningEnabled];
            }
        }
        if (!ok) [APCheckpointStore removeCheckpointAtURL:checkpointURL];   // never partial
        [self deliver:^{ completion(ok); }];
    }];
}

- (BOOL)saveCheckpointToURL:(NSURL *)checkpointURL sessionID:(NSUUID *)sessionID {
    __block BOOL result = NO;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [self saveCheckpointToURL:checkpointURL sessionID:sessionID completion:^(BOOL saved) {
        result = saved;
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    return result;
}

- (void)restoreCheckpointFromURL:(NSURL *)checkpointURL
                       sessionID:(NSUUID *)sessionID
                        messages:(NSArray<APMessage *> *)messages
                      completion:(void (^)(NSError *_Nullable))completion {
    [_model performOnEngine:^{
        if (self->_pos != 0) {
            [self deliver:^{ completion(apSessionError(APErrorInvalidMessage,
                @"checkpoint restore requires a fresh session")); }];
            return;
        }
        NSDictionary * side = [APCheckpointStore sidecarAtCheckpointURL:checkpointURL
                                                      matchingSessionID:sessionID
                                                                  model:self->_model
                                                              reasoning:self.reasoningEnabled];
        if (!side) {
            [self deliver:^{ completion(apSessionError(APErrorInvalidMessage,
                @"no matching checkpoint")); }];
            return;
        }
        const int pos = [side[@"pos"] intValue], turns = [side[@"turnCount"] intValue];
        const BOOL open = [side[@"openModelTurn"] boolValue];
        NSString * fp = [APCheckpointStore fingerprintForModel:self->_model sessionID:sessionID
                                                           pos:pos turnCount:turns
                                                 openModelTurn:open
                                                     reasoning:self.reasoningEnabled];
        int got = self->_cache->restoreSnapshot(
            std::string(checkpointURL.path.UTF8String),
            std::string(fp.UTF8String), APLocalSessionRawMode(self->_model));
        if (got != pos || pos <= 0) {
            self->_cache->reset();
            [self deliver:^{ completion(apSessionError(APErrorEngineFailure,
                @"checkpoint restore failed (stale or corrupt)")); }];
            return;
        }
        @synchronized(self) {
            self->_pos = pos; self->_turnCount = turns; self->_openModelTurn = open;
        }
        [self appendTranscriptMessages:messages];
        [self deliver:^{ completion(nil); }];
    }];
}

+ (NSUUID *)checkpointedSessionIDAtURL:(NSURL *)checkpointURL forModel:(APModel *)model {
    return [APCheckpointStore checkpointedSessionIDAtURL:checkpointURL forModel:model];
}

+ (void)removeCheckpointAtURL:(NSURL *)checkpointURL {
    [APCheckpointStore removeCheckpointAtURL:checkpointURL];
}

#pragma mark - Device checkpoint (convenience default: the URL forms at one fixed location)

+ (NSURL *)deviceCheckpointURL { return APCheckpointStore.deviceCheckpointURL; }
+ (void)removeDeviceCheckpoint { [self removeCheckpointAtURL:APCheckpointStore.deviceCheckpointURL]; }
+ (NSUUID *)checkpointedSessionIDForModel:(APModel *)model {
    return [self checkpointedSessionIDAtURL:APCheckpointStore.deviceCheckpointURL forModel:model];
}

- (BOOL)saveCheckpointForSessionID:(NSUUID *)sessionID {
    return [self saveCheckpointToURL:APCheckpointStore.deviceCheckpointURL sessionID:sessionID];
}

- (void)saveCheckpointForSessionID:(NSUUID *)sessionID
                        completion:(void (^)(BOOL saved))completion {
    [self saveCheckpointToURL:APCheckpointStore.deviceCheckpointURL sessionID:sessionID
                   completion:completion];
}

- (void)restoreCheckpointForSessionID:(NSUUID *)sessionID
                             messages:(NSArray<APMessage *> *)messages
                           completion:(void (^)(NSError *_Nullable))completion {
    [self restoreCheckpointFromURL:APCheckpointStore.deviceCheckpointURL sessionID:sessionID
                          messages:messages completion:completion];
}

@end
