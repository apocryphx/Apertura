//
//  APPersistence.m
//  Apertura
//

#import "APPersistence.h"

@implementation APPersistence

+ (NSManagedObjectModel *)managedObjectModel {
    // App: the momd is in the main bundle. apertura-mcp: the momd is copied beside the
    // executable (its resources); mergedModelFromBundles finds both cases. Last resort:
    // a momd sitting next to the binary itself.
    NSManagedObjectModel * model = [NSManagedObjectModel mergedModelFromBundles:nil];
    if (model.entities.count > 0) return model;
    NSURL * beside = [[NSBundle.mainBundle.executableURL URLByDeletingLastPathComponent]
                         URLByAppendingPathComponent:@"Apertura.momd"];
    return [[NSManagedObjectModel alloc] initWithContentsOfURL:beside];
}

+ (NSPersistentContainer *)sharedContainer {
    static NSPersistentContainer * container;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        container = [[NSPersistentContainer alloc] initWithName:@"Apertura"
                                             managedObjectModel:[self managedObjectModel]];
        // Pin the store path EXPLICITLY: NSPersistentContainer's default directory is
        // process-dependent (app vs CLI resolve differently), and the whole point of
        // this stack is that apertura-mcp opens the SAME file the app does. The pinned
        // path is exactly where the app's container has always put it.
        NSPersistentStoreDescription * store =
            [NSPersistentStoreDescription persistentStoreDescriptionWithURL:[self storeURL]];
        container.persistentStoreDescriptions = @[ store ];
        // See the header: history tracking is a one-way door the CloudKit era opened.
        [store setOption:@YES forKey:NSPersistentHistoryTrackingKey];
        [store setOption:@YES forKey:NSPersistentStoreRemoteChangeNotificationPostOptionKey];
        // v1 -> v2 is additive-only by design; lightweight migration handles it.
        store.shouldMigrateStoreAutomatically = YES;
        store.shouldInferMappingModelAutomatically = YES;

        [container loadPersistentStoresWithCompletionHandler:^(NSPersistentStoreDescription * desc,
                                                               NSError * error) {
            if (error) {
                // Persistence must never take the conversation down with it: log, run on —
                // archiving fails quietly until the store is fixed.
                NSLog(@"Apertura: persistent store failed to load — %@, %@", error, error.userInfo);
            } else {
                NSLog(@"Apertura: store at %@", desc.URL.path);
            }
        }];
        container.viewContext.automaticallyMergesChangesFromParent = YES;
    });
    return container;
}

+ (NSURL *)storeURL {
    // Process-independent: ~/Library/Application Support/Apertura/Apertura.sqlite —
    // the location the app's container has used since v1. An APERTURA_STORE override
    // exists for tooling (cdverify2 migrates a COPY, never the live store).
    NSString * override = NSProcessInfo.processInfo.environment[@"APERTURA_STORE"];
    if (override.length) return [NSURL fileURLWithPath:override];
    NSURL * base = [NSFileManager.defaultManager URLForDirectory:NSApplicationSupportDirectory
                                                        inDomain:NSUserDomainMask
                                               appropriateForURL:nil create:YES error:nil];
    NSURL * dir = [base URLByAppendingPathComponent:@"Apertura" isDirectory:YES];
    [NSFileManager.defaultManager createDirectoryAtURL:dir withIntermediateDirectories:YES
                                            attributes:nil error:nil];
    return [dir URLByAppendingPathComponent:@"Apertura.sqlite"];
}

@end
