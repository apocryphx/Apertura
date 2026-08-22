//
//  APModelRegistry.h
//  Apertura
//
//  Models are a folder, not a database: ~/Library/Application Support/Apertura/Models/
//  holds .apml bundles (or symlinks to them — a 16 GB bundle should not have to be
//  duplicated to be registered), and the registry is a scan of that folder. Per-model
//  load-time configuration (headBits, cache mode, prefill chunk, max context) lives in
//  one NSUserDefaults dictionary keyed by folder name; the selected model is a folder
//  name in another default. Compiled into both the app and apertura-mcp.
//
//  The HF-downloader seam: a future downloader produces a local bundle and funnels it
//  through importModelAtURL:copy:completion: like any other import.

#import <Foundation/Foundation.h>
#import <AperturaKit/AperturaKit.h>

NS_ASSUME_NONNULL_BEGIN

/// One scanned bundle.
@interface APInstalledModel : NSObject
@property (nonatomic, copy) NSString * name;          // folder name, the registry key
@property (nonatomic) NSURL * url;                    // resolved (through symlinks)
@property (nonatomic) unsigned long long sizeBytes;   // weights on disk (resolved target)
@property (nonatomic) BOOL isSymlink;
@end

@interface APModelRegistry : NSObject

#pragma mark - The managed-state home

+ (NSURL *)modelsDirectory;        // …/Application Support/Apertura/Models
+ (NSURL *)checkpointsDirectory;   // …/Application Support/Apertura/Checkpoints

#pragma mark - Scan + selection

/// Every .apml bundle (or symlink to one) in the models directory, by name.
+ (NSArray<APInstalledModel *> *)installedModels;

/// The selected model's folder name (`AperturaSelectedModel`), nil when unset.
+ (nullable NSString *)selectedModelName;
+ (void)setSelectedModelName:(nullable NSString *)name;

/// The URL to load: the selected registry model when it exists, else the single
/// installed model when there is exactly one, else the legacy `AperturaModelPath`
/// default (pre-registry setups keep working), else nil.
+ (nullable NSURL *)resolvedModelURL;

#pragma mark - Per-model load configuration

/// The stored load-time configuration for a model (defaults when none stored).
/// Keyed by folder name in the `AperturaModelConfigs` defaults dictionary.
+ (APModelConfiguration *)configurationForModelNamed:(NSString *)name;
+ (void)setConfiguration:(APModelConfiguration *)configuration forModelNamed:(NSString *)name;

/// Configuration for whatever resolvedModelURL points at (legacy paths get defaults).
+ (APModelConfiguration *)configurationForResolvedModel;

#pragma mark - Import / remove

/// Register a bundle: pre-flight with APModel availability, then either COPY it into the
/// models directory (background queue; multi-GB) or SYMLINK it in place (instant — the
/// registry follows symlinks). Completion on the main queue.
+ (void)importModelAtURL:(NSURL *)url
                    copy:(BOOL)copy
              completion:(void (^)(APInstalledModel *_Nullable model,
                                   NSError *_Nullable error))completion;

/// Remove a registry entry. Deletes the symlink, or the copied bundle when
/// `deleteFiles` (a symlinked original is never touched).
+ (BOOL)removeModelNamed:(NSString *)name deleteFiles:(BOOL)deleteFiles;

@end

NS_ASSUME_NONNULL_END
