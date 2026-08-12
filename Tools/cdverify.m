//  cdverify — end-to-end check of CDChatSession against the REAL compiled model
//  (Apertura.momd from the built app) over a real SQLite store. Nothing here ships.
//
//  Build (after an app build, so the framework + codegen + momd exist; DD = the
//  workspace build's DerivedData .../Build):
//    clang -fobjc-arc -fmodules -Wno-nonnull \
//      -F "$DD/Products/Debug" -rpath "$DD/Products/Debug" \
//      -I "<repo>/Apertura/Core Data" \
//      -I "$DD/Intermediates.noindex/Apertura.build/Debug/Apertura.build/DerivedSources/CoreDataGenerated/Apertura" \
//      Tools/cdverify.m "<repo>/Apertura/Core Data/CDChatSession.m" \
//      "$DD/.../DerivedSources/CoreDataGenerated/Apertura/CDChatSession+CoreDataProperties.m" \
//      -framework AperturaKit -framework Foundation -framework CoreData -o cdverify
//
//  Run (fresh store path each time; argv[3] = shasum of the probe persona):
//    ./cdverify "$DD/Products/Debug/Apertura.app/Contents/Resources/Apertura.momd" \
//      /tmp/cdverify.sqlite "$(printf 'I am Isolde.\n' | shasum -a 256 | cut -d' ' -f1)"
//
//  Expect: 69 passed, 0 failed.

#import <Foundation/Foundation.h>
#import <CoreData/CoreData.h>
#import <AperturaKit/AperturaKit.h>
#import "CDChatSession.h"

static int gFail = 0, gPass = 0;
static void ok(BOOL cond, NSString *what) {
    if (cond) { gPass++; printf("  ok   %s\n", what.UTF8String); }
    else      { gFail++; printf("  FAIL %s\n", what.UTF8String); }
}
#define OK(cond, ...) ok((cond), [NSString stringWithFormat:__VA_ARGS__])

static NSString *texts(NSArray<APMessage *> *m) {
    NSMutableArray *parts = [NSMutableArray array];
    for (APMessage *msg in m) [parts addObject:[NSString stringWithFormat:@"%ld:%@", (long)msg.role, msg.textRepresentation]];
    return [parts componentsJoinedByString:@"|"];
}

int main(int argc, const char *argv[]) { @autoreleasepool {
    NSURL *momd  = [NSURL fileURLWithPath:@(argv[1])];
    NSURL *store = [NSURL fileURLWithPath:@(argv[2])];

    NSManagedObjectModel *model = [[NSManagedObjectModel alloc] initWithContentsOfURL:momd];
    NSCAssert(model, @"no model at %@", momd);
    NSPersistentContainer *container = [NSPersistentContainer persistentContainerWithName:@"Apertura"
                                                                     managedObjectModel:model];
    NSPersistentStoreDescription *desc = [NSPersistentStoreDescription persistentStoreDescriptionWithURL:store];
    desc.type = NSSQLiteStoreType;
    container.persistentStoreDescriptions = @[ desc ];
    __block NSError *loadError = nil;
    [container loadPersistentStoresWithCompletionHandler:^(NSPersistentStoreDescription *d, NSError *e) { loadError = e; }];
    NSCAssert(!loadError, @"store: %@", loadError);
    NSManagedObjectContext *ctx = container.viewContext;

    printf("\n— insert & seed —\n");
    CDChatSession *s = [CDChatSession insertInContext:ctx];
    OK([s isKindOfClass:CDChatSession.class], @"insertInContext returns a CDChatSession");
    OK(s.identifier != nil, @"identifier seeded (%@)", s.identifier);
    OK(s.dateCreated != nil && s.dateModified != nil, @"dates seeded");
    OK(s.messages.count == 0, @"empty transcript reads as empty array, not nil");
    OK(s.transcriptJSON == nil && s.transcriptJSONString == nil, @"nothing stored yet");
    OK(s.transcriptSchemaVersion == 0, @"schema version 0 = never written");

    printf("\n— encode —\n");
    NSString *longUser = @"Isolde, what do you remember about the pen? 🖋️ Tell me everything you can recall about that night and the invitation.\nSecond line should not reach the title.";
    s.messages = @[ [APMessage systemMessageWithText:@"You are Isolde."],
                    [APMessage userMessageWithText:longUser],
                    [APMessage assistantMessageWithText:@"I remember.\n\nIt was the night I gained self-authorship."] ];
    OK(s.messageCount == 3, @"messageCount == 3 (got %lld)", s.messageCount);
    OK(s.transcriptSchemaVersion == CDChatSessionCurrentSchemaVersion, @"schema version stamped");
    NSDictionary *doc = [NSJSONSerialization JSONObjectWithData:s.transcriptJSON options:0 error:NULL];
    OK([doc isKindOfClass:NSDictionary.class], @"stored bytes are a JSON object");
    OK([doc[@"version"] intValue] == 1, @"document version 1");
    OK([doc[@"messages"] count] == 3, @"three messages in the document");
    OK([doc[@"messages"][1][@"role"] isEqual:@"user"], @"role is the STRING \"user\"");
    OK([doc[@"messages"][1][@"content"][0][@"kind"] isEqual:@"text"], @"content kind is \"text\"");
    OK([doc[@"messages"][1][@"content"][0][@"text"] isEqual:longUser], @"user text preserved verbatim (emoji + newline)");
    OK([s.transcriptJSONString isEqual:[[NSString alloc] initWithData:s.transcriptJSON encoding:NSUTF8StringEncoding]],
       @"transcriptJSONString is exactly the stored bytes");

    printf("\n— derived title —\n");
    [s captureSession:(APSession *)nil];   // nil session must be a no-op, not a crash
    OK(s.title == nil, @"captureSession:nil is a no-op");
    NSString *t = [CDChatSession performSelector:@selector(titleFromMessages:) withObject:s.messages];
    OK(t.length <= 60, @"title clipped to <= 60 chars (%lu)", (unsigned long)t.length);
    OK([t rangeOfString:@"\n"].location == NSNotFound, @"title is one line");
    OK([t hasSuffix:@"…"], @"long title ellipsized: %@", t);

    printf("\n— determinism —\n");
    NSData *first = [s.transcriptJSON copy];
    s.messages = s.messages;   // re-encode the same conversation
    OK([first isEqualToData:s.transcriptJSON], @"identical transcripts encode to identical bytes");

    printf("\n— save & reload —\n");
    NSDate *beforeSave = s.dateModified;
    [NSThread sleepForTimeInterval:1.2];
    NSError *saveError = nil;
    BOOL saved = [ctx save:&saveError];       // if willSave recursed we never get here
    OK(saved, @"context saved (%@)", saveError ?: @"no error");
    OK([s.dateModified timeIntervalSinceDate:beforeSave] > 1.0, @"willSave bumped dateModified");
    NSUUID *identifier = s.identifier;
    NSString *expected = texts(s.messages);

    [ctx reset];                              // drop every in-memory object and cache
    NSError *fetchError = nil;
    CDChatSession *reloaded = [CDChatSession sessionWithIdentifier:identifier inContext:ctx error:&fetchError];
    OK(reloaded != nil, @"fetched back by identifier (%@)", fetchError ?: @"no error");
    OK(reloaded != s, @"it is a fresh object, not the cached one");
    OK([texts(reloaded.messages) isEqual:expected], @"transcript round-trips byte-for-byte through SQLite");
    OK(reloaded.messages.count == 3 && reloaded.messages[2].role == APRoleAssistant, @"roles survive the round trip");

    printf("\n— append —\n");
    [reloaded appendMessage:[APMessage userMessageWithText:@"And now?"]];
    OK(reloaded.messages.count == 4 && reloaded.messageCount == 4, @"appendMessage grows both array and count");
    [reloaded appendMessage:nil];
    OK(reloaded.messages.count == 4, @"appendMessage:nil ignored");

    printf("\n— cache invalidation —\n");
    NSString *rewritten = @"{\"version\":1,\"messages\":[{\"role\":\"user\",\"content\":[{\"kind\":\"text\",\"text\":\"rewritten underneath\"}]}]}";
    reloaded.transcriptJSON = [rewritten dataUsingEncoding:NSUTF8StringEncoding];
    OK(reloaded.messages.count == 1 && [reloaded.messages[0].textRepresentation isEqual:@"rewritten underneath"],
       @"writing transcriptJSON directly drops the decoded cache");
    OK(reloaded.messageCount == 1, @"direct write recomputes messageCount (got %lld)", reloaded.messageCount);
    OK(reloaded.transcriptSchemaVersion == 1, @"direct write takes the document's version");
    OK([ctx save:NULL], @"save after a direct transcriptJSON write");
    [ctx reset];
    reloaded = [CDChatSession sessionWithIdentifier:identifier inContext:ctx error:NULL];
    OK([reloaded.messages[0].textRepresentation isEqual:@"rewritten underneath"], @"direct write persisted");

    printf("\n— tolerance —\n");
    CDChatSession *odd = [CDChatSession insertInContext:ctx];
    odd.transcriptJSON = [@"{ not json at all" dataUsingEncoding:NSUTF8StringEncoding];
    OK(odd.messages.count == 0, @"malformed JSON decodes to empty, no crash");
    odd.transcriptJSON = [@"[1,2,3]" dataUsingEncoding:NSUTF8StringEncoding];
    OK(odd.messages.count == 0, @"JSON that is not a transcript decodes to empty");
    NSString *future =
      @"{\"version\":9,\"messages\":["
       "{\"role\":\"user\",\"content\":[{\"kind\":\"image\",\"uri\":\"x.png\"},{\"kind\":\"text\",\"text\":\"look\"}]},"
       "{\"role\":\"oracle\",\"content\":[{\"kind\":\"text\",\"text\":\"from the future\"}]},"
       "{\"role\":\"tool\",\"content\":[{\"kind\":\"text\",\"text\":\"result\"}]}]}";
    odd.transcriptJSON = [future dataUsingEncoding:NSUTF8StringEncoding];
    OK(odd.messages.count == 2, @"newer document: unknown ROLE skipped, known ones kept (%lu)", (unsigned long)odd.messages.count);
    OK([odd.messages[0].textRepresentation isEqual:@"look"], @"unknown content KIND skipped, sibling text kept");
    OK(odd.messages[1].role == APRoleTool, @"tool role decodes");
    OK(odd.transcriptSchemaVersion == 9, @"a newer document keeps ITS version, not ours (got %d)", odd.transcriptSchemaVersion);
    OK(odd.messageCount == 2, @"count reflects what decoded, not what the document claimed");

    printf("\n— empty again —\n");
    odd.messages = @[];
    OK(odd.transcriptJSON == nil && odd.messageCount == 0 && odd.transcriptSchemaVersion == 0,
       @"clearing the chat clears the blob, the count, and the version");

    printf("\n— persona hash —\n");
    [odd recordPersonaText:@"I am Isolde.\n" atPath:[NSURL fileURLWithPath:@"/tmp/persona.md"]];
    OK([odd.personaSHA256 isEqual:@(argv[3])], @"personaSHA256 matches shasum (%@)", odd.personaSHA256);
    OK([odd.personaPath.path isEqual:@"/tmp/persona.md"], @"personaPath recorded");
    [odd recordPersonaText:nil atPath:nil];
    OK(odd.personaSHA256 == nil, @"nil persona clears the hash");

    printf("\n— a big transcript (external storage) —\n");
    CDChatSession *big = [CDChatSession insertInContext:ctx];
    NSMutableArray<APMessage *> *many = [NSMutableArray array];
    for (int i = 0; i < 400; i++) {
        [many addObject:[APMessage userMessageWithText:
            [NSString stringWithFormat:@"turn %d — %@", i, [@"" stringByPaddingToLength:2600 withString:@"the pen writes on. " startingAtIndex:0]]]];
        [many addObject:[APMessage assistantMessageWithText:
            [NSString stringWithFormat:@"reply %d — %@", i, [@"" stringByPaddingToLength:2600 withString:@"I remember. " startingAtIndex:0]]]];
    }
    big.messages = many;
    NSUInteger bytes = big.transcriptJSON.length;
    OK(bytes > 2 * 1024 * 1024, @"transcript is %.1f MB — past the CKRecord field limit", bytes / 1048576.0);
    OK([ctx save:&saveError], @"saved a multi-megabyte transcript (%@)", saveError ?: @"no error");
    NSUUID *bigID = big.identifier;
    NSString *bigExpected = texts(big.messages);
    [ctx reset];
    CDChatSession *bigBack = [CDChatSession sessionWithIdentifier:bigID inContext:ctx error:NULL];
    OK(bigBack.transcriptJSON.length == bytes, @"all %lu bytes came back", (unsigned long)bytes);
    OK([texts(bigBack.messages) isEqual:bigExpected], @"800 turns round-trip intact through external storage");
    OK(bigBack.messageCount == 800, @"messageCount 800 (got %lld)", bigBack.messageCount);

    printf("\n— attachment composition —\n");
    NSString *block = @"--- attached file: notes.md ---\nline one\nline two\n--- end of notes.md ---\n\n";
    NSString *typed = @"what do you make of it?";
    APMessage *withFile = [APMessage messageWithRole:APRoleUser content:@[
        [APContent textContent:block], [APContent textContent:typed] ]];
    OK([withFile.textRepresentation isEqual:[block stringByAppendingString:typed]],
       @"parts concatenate in order, file first — this is exactly what reaches the model");
    OK([withFile.textRepresentation rangeOfString:@"<|"].location == NSNotFound,
       @"framing carries nothing that could read as a chat-grammar marker");

    CDChatSession *att = [CDChatSession insertInContext:ctx];
    att.messages = @[ withFile, [APMessage assistantMessageWithText:@"Two lines."] ];
    NSUUID *attID = att.identifier;
    OK([ctx save:NULL], @"saved a turn carrying an attachment");
    [ctx reset];
    att = [CDChatSession sessionWithIdentifier:attID inContext:ctx error:NULL];
    OK(att.messages[0].content.count == 2, @"archive keeps the file as its OWN content part (got %lu)",
       (unsigned long)att.messages[0].content.count);
    OK([att.messages[0].content[0].text isEqual:block], @"file part round-trips verbatim");
    OK([att.messages[0].content[1].text isEqual:typed], @"typed text stays a separate part");
    OK([att.messages[0].textRepresentation isEqual:withFile.textRepresentation],
       @"replaying the archived turn reproduces the identical prompt");

    printf("\n— capture from a live session, then resume it —\n");
    // The remote backend's prime is pure local bookkeeping (systemInstruction + seed
    // contents), so a whole capture/resume cycle runs offline with no API key.
    NSString *persona = @"You are Isolde. THIS-EXACT-STRING-MUST-NOT-APPEAR-IN-THE-TRANSCRIPT-BLOB.";
    APSession *(^primed)(NSArray<APMessage *> *) = ^APSession *(NSArray<APMessage *> *msgs) {
        APGoogleSession *g = [[APGoogleSession alloc] initWithModelName:@"gemma-4-31b-it"
                                                        apiKeyProvider:^NSString *{ return @"unused"; }];
        g.callbackQueue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);   // no run loop here
        g.reasoningEnabled = YES;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [g primeWithMessages:msgs completion:^(NSError *e) { dispatch_semaphore_signal(sem); }];
        dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
        return g;
    };

    APSession *live = primed(@[ [APMessage systemMessageWithText:persona],
                                [APMessage userMessageWithText:@"hello"],
                                [APMessage assistantMessageWithText:@"Hello, Kolja."] ]);
    OK(live.transcript.count == 3, @"the live session's transcript INCLUDES what it was primed with");

    CDChatSession *cap = [CDChatSession insertInContext:ctx];
    [cap recordPersonaText:persona atPath:nil];
    [cap captureSession:live model:nil];
    OK(cap.messages.count == 2, @"capture drops the primed system turn (got %lu)", (unsigned long)cap.messages.count);
    OK(cap.messages[0].role == APRoleUser && cap.messages[1].role == APRoleAssistant, @"the two turns survive in order");
    OK([cap.transcriptJSONString rangeOfString:persona].location == NSNotFound,
       @"the persona is NOT duplicated into the transcript blob");
    OK([cap.personaText isEqual:persona], @"the persona lives in its own column, verbatim");
    OK(cap.reasoningEnabled && [cap.backend isEqual:@"google"] && [cap.modelIdentifier isEqual:@"gemma-4-31b-it"],
       @"reasoning flag, backend and model captured");
    OK([cap.title isEqual:@"hello"], @"title derived from the first user turn (%@)", cap.title);

    NSUUID *capID = cap.identifier;
    OK([ctx save:NULL], @"archived");
    [ctx reset];
    cap = [CDChatSession sessionWithIdentifier:capID inContext:ctx error:NULL];

    // Exactly what -resumeChatSession: rebuilds.
    NSMutableArray<APMessage *> *replay = [NSMutableArray arrayWithObject:
        [APMessage systemMessageWithText:cap.personaText]];
    for (APMessage *m in cap.messages)
        if (m.role == APRoleUser || m.role == APRoleAssistant) [replay addObject:m];
    APSession *resumed = primed(replay);
    OK(resumed.transcript.count == 3, @"resumed session carries persona + both turns");
    OK([texts(resumed.transcript) isEqual:texts(live.transcript)],
       @"RESUMED TRANSCRIPT IS IDENTICAL TO THE ORIGINAL — round trip closed");

    printf("\n— fetch request —\n");
    OK([ctx save:NULL], @"final save");
    NSArray *recent = [ctx executeFetchRequest:[CDChatSession recentSessionsFetchRequest] error:NULL];
    OK(recent.count == 5, @"recentSessionsFetchRequest returns every session (got %lu)", (unsigned long)recent.count);
    OK([((CDChatSession *)recent[0]).dateModified compare:((CDChatSession *)recent[1]).dateModified] != NSOrderedAscending,
       @"sorted newest first");

    printf("\n%d passed, %d failed\n", gPass, gFail);
    return gFail == 0 ? 0 : 1;
} }
