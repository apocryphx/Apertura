//
//  APModelRegistry.m
//  Apertura
//

#import "APModelRegistry.h"

static NSString * const kSelectedModelKey = @"AperturaSelectedModel";
static NSString * const kModelConfigsKey  = @"AperturaModelConfigs";
static NSString * const kLegacyModelPathKey = @"AperturaModelPath";

// Config dictionary keys (per model).
static NSString * const kCfgHeadBits   = @"headBits";
static NSString * const kCfgCacheMode  = @"cacheMode";
static NSString * const kCfgChunk      = @"prefillChunkLength";
static NSString * const kCfgMaxContext = @"maximumContextLength";

@implementation APInstalledModel
@end

@implementation APModelRegistry

#pragma mark - Directories

+ (NSURL *)supportDirectory {
    NSURL * base = [NSFileManager.defaultManager URLForDirectory:NSApplicationSupportDirectory
                                                        inDomain:NSUserDomainMask
                                               appropriateForURL:nil create:YES error:nil];
    return [base URLByAppendingPathComponent:@"Apertura" isDirectory:YES];
}

+ (NSURL *)modelsDirectory {
    NSURL * dir = [[self supportDirectory] URLByAppendingPathComponent:@"Models" isDirectory:YES];
    [NSFileManager.defaultManager createDirectoryAtURL:dir withIntermediateDirectories:YES
                                            attributes:nil error:nil];
    return dir;
}

+ (NSURL *)checkpointsDirectory {
    NSURL * dir = [[self supportDirectory] URLByAppendingPathComponent:@"Checkpoints" isDirectory:YES];
    [NSFileManager.defaultManager createDirectoryAtURL:dir withIntermediateDirectories:YES
                                            attributes:nil error:nil];
    return dir;
}

#pragma mark - Scan + selection

+ (unsigned long long)sizeOfDirectoryAtURL:(NSURL *)url {
    unsigned long long total = 0;
    NSDirectoryEnumerator * e =
        [NSFileManager.defaultManager enumeratorAtURL:url
                           includingPropertiesForKeys:@[ NSURLTotalFileAllocatedSizeKey,
                                                         NSURLFileSizeKey ]
                                              options:0 errorHandler:nil];
    for (NSURL * f in e) {
        NSNumber * size = nil;
        [f getResourceValue:&size forKey:NSURLFileSizeKey error:nil];
        total += size.unsignedLongLongValue;
    }
    return total;
}

+ (NSArray<APInstalledModel *> *)installedModels {
    NSMutableArray<APInstalledModel *> * models = [NSMutableArray array];
    NSArray<NSURL *> * entries =
        [NSFileManager.defaultManager contentsOfDirectoryAtURL:[self modelsDirectory]
                                    includingPropertiesForKeys:@[ NSURLIsSymbolicLinkKey ]
                                                       options:NSDirectoryEnumerationSkipsHiddenFiles
                                                         error:nil];
    for (NSURL * entry in entries) {
        if (![entry.pathExtension isEqualToString:@"apml"]) continue;
        NSNumber * link = nil;
        [entry getResourceValue:&link forKey:NSURLIsSymbolicLinkKey error:nil];
        NSURL * resolved = link.boolValue ? entry.URLByResolvingSymlinksInPath : entry;
        BOOL isDir = NO;
        if (![NSFileManager.defaultManager fileExistsAtPath:resolved.path isDirectory:&isDir]
            || !isDir) continue;   // dangling symlink or stray file
        APInstalledModel * m = [[APInstalledModel alloc] init];
        m.name = entry.lastPathComponent;
        m.url = resolved;
        m.isSymlink = link.boolValue;
        m.sizeBytes = [self sizeOfDirectoryAtURL:resolved];
        [models addObject:m];
    }
    [models sortUsingComparator:^NSComparisonResult(APInstalledModel * a, APInstalledModel * b) {
        return [a.name localizedStandardCompare:b.name];
    }];
    return models;
}

+ (NSString *)selectedModelName {
    return [NSUserDefaults.standardUserDefaults stringForKey:kSelectedModelKey];
}

+ (void)setSelectedModelName:(NSString *)name {
    if (name) [NSUserDefaults.standardUserDefaults setObject:name forKey:kSelectedModelKey];
    else      [NSUserDefaults.standardUserDefaults removeObjectForKey:kSelectedModelKey];
}

+ (NSURL *)resolvedModelURL {
    NSArray<APInstalledModel *> * installed = [self installedModels];
    NSString * selected = [self selectedModelName];
    for (APInstalledModel * m in installed)
        if ([m.name isEqualToString:selected]) return m.url;
    if (installed.count == 1) return installed.firstObject.url;
    // Pre-registry setups: the old defaults override keeps working.
    NSString * legacy = [NSUserDefaults.standardUserDefaults stringForKey:kLegacyModelPathKey];
    if (legacy.length && [NSFileManager.defaultManager fileExistsAtPath:legacy])
        return [NSURL fileURLWithPath:legacy];
    return nil;
}

#pragma mark - Per-model load configuration

+ (APModelConfiguration *)configurationForModelNamed:(NSString *)name {
    APModelConfiguration * config = [[APModelConfiguration alloc] init];
    NSDictionary * all = [NSUserDefaults.standardUserDefaults dictionaryForKey:kModelConfigsKey];
    NSDictionary * stored = name ? all[name] : nil;
    if ([stored isKindOfClass:NSDictionary.class]) {
        if (stored[kCfgHeadBits])   config.headBits = [stored[kCfgHeadBits] integerValue];
        if (stored[kCfgCacheMode])  config.globalKVCacheMode = [stored[kCfgCacheMode] integerValue];
        if (stored[kCfgChunk])      config.prefillChunkLength = [stored[kCfgChunk] integerValue];
        if (stored[kCfgMaxContext]) config.maximumContextLength = [stored[kCfgMaxContext] integerValue];
    }
    return config;
}

+ (void)setConfiguration:(APModelConfiguration *)configuration forModelNamed:(NSString *)name {
    if (!name.length) return;
    NSMutableDictionary * all =
        [[NSUserDefaults.standardUserDefaults dictionaryForKey:kModelConfigsKey] mutableCopy]
            ?: [NSMutableDictionary dictionary];
    all[name] = @{ kCfgHeadBits   : @(configuration.headBits),
                   kCfgCacheMode  : @(configuration.globalKVCacheMode),
                   kCfgChunk      : @(configuration.prefillChunkLength),
                   kCfgMaxContext : @(configuration.maximumContextLength) };
    [NSUserDefaults.standardUserDefaults setObject:all forKey:kModelConfigsKey];
}

+ (APModelConfiguration *)configurationForResolvedModel {
    NSURL * url = [self resolvedModelURL];
    return [self configurationForModelNamed:url.lastPathComponent];
}

#pragma mark - Import / remove

+ (void)importModelAtURL:(NSURL *)url
                    copy:(BOOL)copy
              completion:(void (^)(APInstalledModel *, NSError *))completion {
    void (^finish)(APInstalledModel *, NSError *) = ^(APInstalledModel * m, NSError * e) {
        dispatch_async(dispatch_get_main_queue(), ^{ completion(m, e); });
    };
    // Pre-flight BEFORE any copying: a broken bundle should fail in milliseconds.
    APModelAvailability avail = [APModel availabilityOfModelAtURL:url configuration:nil];
    if (avail != APModelAvailable) {
        finish(nil, [NSError errorWithDomain:@"com.elarity.Apertura" code:avail
                                    userInfo:@{ NSLocalizedDescriptionKey :
            [NSString stringWithFormat:@"Not a usable model bundle (availability %ld): %@",
             (long)avail, url.path] }]);
        return;
    }
    NSURL * dest = [[self modelsDirectory] URLByAppendingPathComponent:url.lastPathComponent];
    if ([NSFileManager.defaultManager fileExistsAtPath:dest.path]) {
        finish(nil, [NSError errorWithDomain:@"com.elarity.Apertura" code:100
                                    userInfo:@{ NSLocalizedDescriptionKey :
            [NSString stringWithFormat:@"A model named %@ is already registered.",
             url.lastPathComponent] }]);
        return;
    }
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        NSError * error = nil;
        BOOL ok = copy
            ? [NSFileManager.defaultManager copyItemAtURL:url toURL:dest error:&error]
            : [NSFileManager.defaultManager createSymbolicLinkAtURL:dest
                                                 withDestinationURL:url error:&error];
        if (!ok) { finish(nil, error); return; }
        APInstalledModel * m = [[APInstalledModel alloc] init];
        m.name = dest.lastPathComponent;
        m.url = copy ? dest : url;
        m.isSymlink = !copy;
        m.sizeBytes = [self sizeOfDirectoryAtURL:m.url];
        finish(m, nil);
    });
}

+ (BOOL)removeModelNamed:(NSString *)name deleteFiles:(BOOL)deleteFiles {
    if (!name.length) return NO;
    NSURL * entry = [[self modelsDirectory] URLByAppendingPathComponent:name];
    NSNumber * link = nil;
    [entry getResourceValue:&link forKey:NSURLIsSymbolicLinkKey error:nil];
    // A symlink entry is always just unlinked — the original is never ours to delete.
    // A copied bundle is deleted only on request; otherwise it is left in place but
    // that means it stays registered, so refuse the half-measure.
    if (!link.boolValue && !deleteFiles) return NO;
    BOOL ok = [NSFileManager.defaultManager removeItemAtURL:entry error:nil];
    if (ok && [[self selectedModelName] isEqualToString:name]) [self setSelectedModelName:nil];
    return ok;
}

@end
