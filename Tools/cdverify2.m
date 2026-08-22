//  cdverify2 — the Core Data v2 gate: lightweight migration of a REAL v1 store copy,
//  CDPersona version-chain semantics, and the CDChatSession v2 additions. Nothing here
//  ships. Companion to cdverify.m (which still gates the v1 surface — run both).
//
//  Build (after an app build; DD = the workspace build's DerivedData .../Build,
//  GEN = "$DD/Intermediates.noindex/Apertura.build/Debug/Apertura.build/DerivedSources/CoreDataGenerated/Apertura"):
//    clang -fobjc-arc -fmodules -Wno-nonnull \
//      -F "$DD/Products/Debug" -rpath "$DD/Products/Debug" \
//      -I "<repo>/Apertura/Core Data" -I "$GEN" \
//      Tools/cdverify2.m "<repo>/Apertura/Core Data/CDChatSession.m" \
//      "<repo>/Apertura/Core Data/CDPersona.m" \
//      "$GEN/CDChatSession+CoreDataProperties.m" "$GEN/CDPersona+CoreDataProperties.m" \
//      -framework AperturaKit -framework Foundation -framework CoreData -o cdverify2
//
//  Run against a COPY of the live store (never the live store — migration writes):
//    cp ~/Library/Application\ Support/Apertura/Apertura.sqlite* /tmp/cdv2/
//    ./cdverify2 "$DD/Products/Debug/Apertura.app/Contents/Resources/Apertura.momd" \
//                /tmp/cdv2/Apertura.sqlite

#import <Foundation/Foundation.h>
#import <CoreData/CoreData.h>
#import <AperturaKit/AperturaKit.h>
#import "CDChatSession.h"
#import "CDPersona.h"

static int gFail = 0, gPass = 0;
static void check(BOOL ok, NSString * what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what.UTF8String);
    ok ? gPass++ : gFail++;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 3) { fprintf(stderr, "usage: cdverify2 <momd> <store-copy.sqlite>\n"); return 2; }
        NSURL * momd = [NSURL fileURLWithPath:@(argv[1])];
        NSURL * storeURL = [NSURL fileURLWithPath:@(argv[2])];

        NSManagedObjectModel * model = [[NSManagedObjectModel alloc] initWithContentsOfURL:momd];
        check(model != nil, @"v2 model loads");
        check(model.entitiesByName[@"CDPersona"] != nil, @"CDPersona entity present");

        // Open exactly the way APPersistence does: history tracking + lightweight migration.
        NSPersistentContainer * container =
            [[NSPersistentContainer alloc] initWithName:@"Apertura" managedObjectModel:model];
        NSPersistentStoreDescription * desc =
            [NSPersistentStoreDescription persistentStoreDescriptionWithURL:storeURL];
        [desc setOption:@YES forKey:NSPersistentHistoryTrackingKey];
        desc.shouldMigrateStoreAutomatically = YES;
        desc.shouldInferMappingModelAutomatically = YES;
        desc.shouldAddStoreAsynchronously = NO;
        container.persistentStoreDescriptions = @[ desc ];
        __block NSError * loadError = nil;
        [container loadPersistentStoresWithCompletionHandler:^(NSPersistentStoreDescription * d,
                                                               NSError * e) { loadError = e; }];
        check(loadError == nil, [NSString stringWithFormat:@"v1 store migrates (%@)",
                                 loadError.localizedDescription ?: @"lightweight"]);
        if (loadError) { printf("\n%d passed, %d failed\n", gPass, gFail); return 1; }
        NSManagedObjectContext * moc = container.viewContext;

        // ── migrated v1 rows ──
        NSArray<CDChatSession *> * old =
            [moc executeFetchRequest:CDChatSession.recentSessionsFetchRequest error:nil];
        check(old != nil, [NSString stringWithFormat:@"old sessions fetch (%lu rows)",
                           (unsigned long)old.count]);
        BOOL decodes = YES, counts = YES, defaults = YES;
        for (CDChatSession * s in old) {
            NSArray * m = s.messages;
            decodes = decodes && (m != nil);
            if (s.transcriptSchemaVersion == 1) counts = counts && ((int64_t)m.count == s.messageCount);
            defaults = defaults && fabs(s.temperature - 0.7) < 1e-9 && s.topK == 64
                                && fabs(s.topP - 0.95) < 1e-9 && s.seed == 0
                                && s.excludesReasoningFromContext && s.checkpointDate == nil;
        }
        check(decodes,  @"every migrated transcript decodes");
        check(counts,   @"messageCount matches decoded turns on v1 rows");
        check(defaults, @"v2 attribute defaults landed on migrated rows");

        // ── CDPersona version chain ──
        CDPersona * p = [CDPersona insertInContext:moc];
        p.name = @"cdverify2 persona";
        [p updateBody:@"A"];
        NSString * shaA = p.sha256;
        [p snapshotBeforeEditWithNote:@"to B" author:@"cdverify2"];
        [p updateBody:@"B"];
        [p snapshotBeforeEditWithNote:@"to C" author:@"cdverify2"];
        [p updateBody:@"C"];
        NSArray<CDPersona *> * chain = [p versionChain];
        check(chain.count == 3, @"version chain length 3 after two edits");
        check([chain[0].body isEqualToString:@"C"] && [chain[1].body isEqualToString:@"B"]
              && [chain[2].body isEqualToString:@"A"], @"chain newest-first: C,B,A");
        check([chain[2].sha256 isEqualToString:shaA], @"frozen row keeps the old hash");
        check(p.isCurrentVersion && !chain[1].isCurrentVersion, @"head detection");
        check([chain[1].notes containsString:@"cdverify2"], @"revision note recorded");
        check(p.identifier != nil && ![chain[1].identifier isEqual:p.identifier],
              @"history rows have their own identity; head keeps its UUID");

        // Heads-only fetch sees exactly one of the three rows.
        NSArray * heads = [moc executeFetchRequest:CDPersona.currentPersonasFetchRequest error:nil];
        NSUInteger mine = 0;
        for (CDPersona * h in heads) if ([h.name isEqualToString:@"cdverify2 persona"]) mine++;
        check(mine == 1, @"currentPersonasFetchRequest returns the head only");

        // ── session <-> persona, and v2 helpers ──
        CDChatSession * s = [CDChatSession insertInContext:moc];
        s.persona = p;
        check([p.sessions containsObject:s], @"sessions inverse");
        check([s.checkpointURL.lastPathComponent
                  isEqualToString:[s.identifier.UUIDString stringByAppendingString:@".safetensors"]]
              && [s.checkpointURL.URLByDeletingLastPathComponent.lastPathComponent
                  isEqualToString:@"Checkpoints"], @"checkpointURL shape");
        APGenerationOptions * opts = [s generationOptions];
        check(fabs(opts.temperature - 0.7) < 1e-6 && opts.topK == 64
              && fabs(opts.topP - 0.95) < 1e-6 && opts.seed == 0
              && opts.maximumResponseTokens == 0, @"generationOptions from row defaults");
        s.seed = 42; s.temperature = 1.0;
        opts = [s generationOptions];
        check(opts.seed == 42 && fabs(opts.temperature - 1.0) < 1e-6,
              @"generationOptions reflects row edits");

        check([moc save:nil], @"context saves");

        // ── cascade: deleting the head deletes its whole chain; the session survives ──
        NSUUID * headID = p.identifier;
        [moc deleteObject:p];
        check([moc save:nil], @"delete saves");
        check([CDPersona personaWithIdentifier:headID inContext:moc error:nil] == nil,
              @"head deleted");
        NSFetchRequest * all = [CDPersona fetchRequest];
        all.predicate = [NSPredicate predicateWithFormat:@"name == %@", @"cdverify2 persona"];
        check([moc executeFetchRequest:all error:nil].count == 0,
              @"history chain cascaded away");
        check(!s.isDeleted && s.persona == nil, @"session survives with persona nullified");
        [moc deleteObject:s];
        [moc save:nil];

        printf("\n%d passed, %d failed\n", gPass, gFail);
        return gFail ? 1 : 0;
    }
}
