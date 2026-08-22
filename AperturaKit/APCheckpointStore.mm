//  APCheckpointStore — see header. ObjC++ (.mm) only because APInternal.h gates the
//  APModel(Internal) category behind __cplusplus; nothing here touches the engine, MLX,
//  or the KV cache, so the file/crypto/policy layer stays testable without a model.
#import "APCheckpointStore.h"
#import "APInternal.h"
#import <CommonCrypto/CommonDigest.h>

static NSURL * apCheckpointDirURL(void) {
    NSURL * base = [NSFileManager.defaultManager URLForDirectory:NSApplicationSupportDirectory
                                                        inDomain:NSUserDomainMask
                                               appropriateForURL:nil create:YES error:nil];
    NSString * bundle = NSBundle.mainBundle.bundleIdentifier ?: @"Apertura";
    NSURL * dir = [base URLByAppendingPathComponent:bundle isDirectory:YES];
    [NSFileManager.defaultManager createDirectoryAtURL:dir withIntermediateDirectories:YES
                                            attributes:nil error:nil];
    return dir;
}

@implementation APCheckpointStore

#pragma mark - URL-parameterized checkpoints

+ (NSURL *)sidecarURLForCheckpointURL:(NSURL *)checkpointURL {
    return [[checkpointURL URLByDeletingPathExtension] URLByAppendingPathExtension:@"plist"];
}

+ (void)removeCheckpointAtURL:(NSURL *)checkpointURL {
    [NSFileManager.defaultManager removeItemAtURL:checkpointURL error:nil];
    [NSFileManager.defaultManager removeItemAtURL:[self sidecarURLForCheckpointURL:checkpointURL]
                                            error:nil];
}

+ (NSUUID *)checkpointedSessionIDAtURL:(NSURL *)checkpointURL forModel:(APModel *)model {
    NSDictionary * side = [NSDictionary dictionaryWithContentsOfURL:
                              [self sidecarURLForCheckpointURL:checkpointURL]];
    if (![side isKindOfClass:NSDictionary.class]) return nil;
    if (![side[@"modelIdentifier"] isEqual:(model.modelIdentifier ?: @"")]) return nil;
    if (![NSFileManager.defaultManager fileExistsAtPath:checkpointURL.path]) return nil;
    NSString * sid = side[@"sessionID"];
    return [sid isKindOfClass:NSString.class] ? [[NSUUID alloc] initWithUUIDString:sid] : nil;
}

+ (BOOL)writeSidecarForCheckpointURL:(NSURL *)checkpointURL
                           sessionID:(NSUUID *)sessionID
                               model:(APModel *)model
                                 pos:(int)pos
                           turnCount:(int)turnCount
                       openModelTurn:(BOOL)openModelTurn
                           reasoning:(BOOL)reasoning {
    NSDictionary * side = @{
        @"version"          : @2,
        @"sessionID"        : sessionID.UUIDString,
        @"modelIdentifier"  : model.modelIdentifier ?: @"",
        @"pos"              : @(pos),
        @"turnCount"        : @(turnCount),
        @"openModelTurn"    : @(openModelTurn),
        @"reasoningEnabled" : @(reasoning),
    };
    return [side writeToURL:[self sidecarURLForCheckpointURL:checkpointURL] error:nil];
}

+ (NSDictionary *)sidecarAtCheckpointURL:(NSURL *)checkpointURL
                       matchingSessionID:(NSUUID *)sessionID
                                   model:(APModel *)model
                               reasoning:(BOOL)reasoning {
    NSDictionary * side = [NSDictionary dictionaryWithContentsOfURL:
                              [self sidecarURLForCheckpointURL:checkpointURL]];
    if (![side isKindOfClass:NSDictionary.class]) return nil;
    NSString * sid = side[@"sessionID"];
    BOOL match = [sid isKindOfClass:NSString.class]
        && [sid isEqualToString:sessionID.UUIDString]
        && [side[@"modelIdentifier"] isEqual:(model.modelIdentifier ?: @"")]
        && [side[@"reasoningEnabled"] boolValue] == reasoning;
    return match ? side : nil;
}

+ (NSString *)fingerprintForModel:(APModel *)model
                        sessionID:(NSUUID *)sessionID
                              pos:(int)pos
                        turnCount:(int)turnCount
                    openModelTurn:(BOOL)openModelTurn
                        reasoning:(BOOL)reasoning {
    NSMutableData * blob = [NSMutableData data];
    uint32_t version = 2;
    [blob appendBytes:&version length:sizeof(version)];
    [blob appendData:[model.modelIdentifier dataUsingEncoding:NSUTF8StringEncoding]];
    unsigned long long wb = [model internalWeightBytes];
    [blob appendBytes:&wb length:sizeof(wb)];
    int64_t head = [model internalConfiguration].headBits;
    [blob appendBytes:&head length:sizeof(head)];
    [blob appendData:[sessionID.UUIDString dataUsingEncoding:NSUTF8StringEncoding]];
    int32_t scalars[3] = {pos, turnCount, (openModelTurn ? 1 : 0) | (reasoning ? 2 : 0)};
    [blob appendBytes:scalars length:sizeof(scalars)];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(blob.bytes, (CC_LONG) blob.length, digest);
    NSMutableString * hex = [NSMutableString stringWithCapacity:2 * CC_SHA256_DIGEST_LENGTH];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) [hex appendFormat:@"%02x", digest[i]];
    return hex;
}

#pragma mark - Device checkpoint (convenience default)

+ (NSURL *)deviceCheckpointURL {
    return [apCheckpointDirURL() URLByAppendingPathComponent:@"session-checkpoint.safetensors"];
}

+ (NSURL *)sidecarURL {
    return [self sidecarURLForCheckpointURL:self.deviceCheckpointURL];
}

+ (void)removeDeviceCheckpoint {
    [self removeCheckpointAtURL:self.deviceCheckpointURL];
}

+ (NSUUID *)checkpointedSessionIDForModel:(APModel *)model {
    return [self checkpointedSessionIDAtURL:self.deviceCheckpointURL forModel:model];
}

+ (BOOL)writeSidecarForSessionID:(NSUUID *)sessionID
                           model:(APModel *)model
                             pos:(int)pos
                       turnCount:(int)turnCount
                   openModelTurn:(BOOL)openModelTurn
                       reasoning:(BOOL)reasoning {
    return [self writeSidecarForCheckpointURL:self.deviceCheckpointURL sessionID:sessionID
                                        model:model pos:pos turnCount:turnCount
                                openModelTurn:openModelTurn reasoning:reasoning];
}

+ (NSDictionary *)sidecarMatchingSessionID:(NSUUID *)sessionID
                                     model:(APModel *)model
                                 reasoning:(BOOL)reasoning {
    return [self sidecarAtCheckpointURL:self.deviceCheckpointURL matchingSessionID:sessionID
                                  model:model reasoning:reasoning];
}

@end
