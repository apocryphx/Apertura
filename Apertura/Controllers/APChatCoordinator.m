//
//  APChatCoordinator.m
//  Apertura
//
//  A straight port of ViewController's engine/session/persistence logic (see the
//  header). Where the old code touched a view, this one calls the delegate; where it
//  read a control's state, it reads the persisted default the control mirrors.

#import "APChatCoordinator.h"
#import "AppDelegate.h"
#import "CDChatSession.h"
#import <Security/Security.h>

static NSString * const kModelPathDefaultsKey   = @"AperturaModelPath";
static NSString * const kPersonaPathDefaultsKey = @"AperturaPersonaPath";
static NSString * const kReasoningDefaultsKey   = @"AperturaReasoningEnabled";
static NSString * const kBackendDefaultsKey     = @"AperturaBackend";         // "local" | "google"
static NSString * const kGoogleModelDefaultsKey = @"AperturaGoogleModel";     // e.g. gemma-4-31b-it
static NSString * const kModelsDir = @"/Volumes/Macintosh HD/Users/apocryphx/Models";

static NSString * const kKeychainService = @"com.elarity.Apertura";
static NSString * const kKeychainAccount = @"GeminiAPIKey";

#pragma mark - Keychain (Gemini API key)

static NSString * apReadAPIKey(void) {
    NSDictionary * query = @{ (id)kSecClass : (id)kSecClassGenericPassword,
                              (id)kSecAttrService : kKeychainService,
                              (id)kSecAttrAccount : kKeychainAccount,
                              (id)kSecReturnData : @YES,
                              (id)kSecMatchLimit : (id)kSecMatchLimitOne };
    CFTypeRef out = NULL;
    if (SecItemCopyMatching((__bridge CFDictionaryRef)query, &out) != errSecSuccess) return nil;
    NSString * key = [[NSString alloc] initWithData:(__bridge_transfer NSData *)out
                                           encoding:NSUTF8StringEncoding];
    return key.length ? key : nil;
}

static BOOL apStoreAPIKey(NSString * key) {
    NSDictionary * query = @{ (id)kSecClass : (id)kSecClassGenericPassword,
                              (id)kSecAttrService : kKeychainService,
                              (id)kSecAttrAccount : kKeychainAccount };
    SecItemDelete((__bridge CFDictionaryRef)query);
    if (key.length == 0) return YES;   // empty = remove
    NSMutableDictionary * add = [query mutableCopy];
    add[(id)kSecValueData] = [key dataUsingEncoding:NSUTF8StringEncoding];
    return SecItemAdd((__bridge CFDictionaryRef)add, NULL) == errSecSuccess;
}

#pragma mark - Attachments

/// A file read at ATTACH time, not at send time: what you saw in the open panel is what
/// Isolde gets, even if the file changes (or the tools rewrite it) in between.
@interface APAttachment : NSObject
@property (nonatomic, copy) NSString * name;
@property (nonatomic, copy) NSString * text;
@property (nonatomic) unsigned long long bytes;
@property (nonatomic) NSUInteger estimatedTokens;
@end

@implementation APAttachment
@end

/// Refuse to even read something enormous — this is a chat window, not an importer.
static const unsigned long long kAttachmentMaximumBytes = 4 * 1024 * 1024;
/// Tokens the reply needs room for, held back from the local context budget.
static const NSInteger kReplyHeadroomTokens = 2048;

/// No tokenizer is exposed on the public facade, so this is the usual ~4-chars-a-token
/// approximation. It only ever drives a warning, never the prompt itself.
static NSUInteger apEstimatedTokens(NSString * text) {
    return (text.length + 3) / 4;
}

/// How an attachment reaches the model. Both backends flatten a message's content parts by
/// plain concatenation, so the framing has to live in the text. Dashes rather than angle
/// brackets: nothing here can be mistaken for a chat-grammar marker.
static NSString * apAttachmentBlock(APAttachment * attachment) {
    return [NSString stringWithFormat:@"--- attached file: %@ ---\n%@\n--- end of %@ ---\n\n",
            attachment.name, attachment.text, attachment.name];
}

static NSString * apHumanBytes(unsigned long long bytes) {
    return [NSByteCountFormatter stringFromByteCount:(long long)bytes
                                          countStyle:NSByteCountFormatterCountStyleFile];
}

#pragma mark - Coordinator

@interface APChatCoordinator () <APSessionDelegate>

@property (nonatomic) APModel * model;
@property (nonatomic) APSession * session;
@property (nonatomic) APResponseTask * currentTask;
@property (nonatomic) BOOL checkpointResumeAttempted;   // device-checkpoint restore: launch only
@property (nonatomic) NSString * personaPath;   // resolved at session start; tool handlers write here

@property (nonatomic) NSMutableArray<APAttachment *> * stagedAttachments;

// Chat persistence: the Core Data row this conversation archives into (created lazily on
// the first completed turn) and the persona exactly as primed — the self-editing tools
// may rewrite the persona file mid-conversation, but the row must record the waking of
// Isolde that actually ran.
@property (nonatomic) CDChatSession * chatSession;
@property (nonatomic) NSString * primedPersona;

// Control state the views mirror.
@property (readwrite) BOOL composeEnabled;
@property (readwrite) BOOL sessionControlsEnabled;
@property (readwrite) BOOL stopEnabled;

@end

@implementation APChatCoordinator

- (instancetype)init {
    if ((self = [super init])) {
        _stagedAttachments = [NSMutableArray array];
    }
    return self;
}

#pragma mark - Delegate shorthands (the old rendering calls, one-to-one)

- (void)note:(NSString *)text          { [self.delegate coordinator:self appendNote:text]; }
- (void)streamed:(NSString *)text      { [self.delegate coordinator:self appendStreamedText:text]; }
- (void)speakerHeader:(NSString *)name { [self.delegate coordinator:self appendSpeakerHeader:name]; }
- (void)setBusy:(BOOL)busy status:(NSString *)status {
    [self.delegate coordinator:self didChangeStatus:status busy:busy];
}
- (void)controlsChanged { [self.delegate coordinatorDidChangeControls:self]; }
- (void)setCompose:(BOOL)compose sessionControls:(BOOL)session stop:(BOOL)stop {
    self.composeEnabled = compose;
    self.sessionControlsEnabled = session;
    self.stopEnabled = stop;
    [self controlsChanged];
}

#pragma mark - Paths

- (NSURL *)modelURL {
    NSString * p = [NSUserDefaults.standardUserDefaults stringForKey:kModelPathDefaultsKey]
        ?: [kModelsDir stringByAppendingPathComponent:@"gemma-4-31b-it-qat-q4.apml"];
    return [NSURL fileURLWithPath:p];
}

/// The persona file that will actually be primed (and that the self-editing tools mutate).
- (NSString *)resolvedPersonaPath {
    NSString * p = [NSUserDefaults.standardUserDefaults stringForKey:kPersonaPathDefaultsKey];
    NSArray<NSString *> * candidates = p ? @[ p ]
        : @[ [kModelsDir stringByAppendingPathComponent:@"isolde_system.md"],
             [kModelsDir stringByAppendingPathComponent:@"isolde_prompt.txt"] ];
    for (NSString * path in candidates) {
        NSDictionary * a = [NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
        if ([a[NSFileSize] unsignedLongLongValue] > 0) return path;
    }
    return nil;
}

- (NSString *)personaText {
    NSString * path = self.personaPath ?: [self resolvedPersonaPath];
    return path ? [NSString stringWithContentsOfFile:path
                                            encoding:NSUTF8StringEncoding error:nil] : nil;
}

/// Copy the persona file into persona_history/ (sibling directory) and log the change.
/// Every mutation archives FIRST — the history is how an edit is ever undone (restore =
/// copy a snapshot back over the persona file).
- (BOOL)archivePersonaWithNote:(NSString *)note {
    NSFileManager * fm = NSFileManager.defaultManager;
    NSString * dir = [self.personaPath.stringByDeletingLastPathComponent
                      stringByAppendingPathComponent:@"persona_history"];
    if (![fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil])
        return NO;
    NSDateFormatter * f = [[NSDateFormatter alloc] init];
    f.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
    f.dateFormat = @"yyyy-MM-dd'T'HH-mm-ss";
    NSString * stamp = [f stringFromDate:NSDate.date];
    NSString * base = self.personaPath.lastPathComponent;
    NSString * dest = [dir stringByAppendingPathComponent:
                       [NSString stringWithFormat:@"%@.%@", stamp, base]];
    for (int n = 2; [fm fileExistsAtPath:dest] && n < 100; n++)
        dest = [dir stringByAppendingPathComponent:
                [NSString stringWithFormat:@"%@-%d.%@", stamp, n, base]];
    if (![fm copyItemAtPath:self.personaPath toPath:dest error:nil]) return NO;

    NSString * logPath = [dir stringByAppendingPathComponent:@"changes.log"];
    NSString * line = [NSString stringWithFormat:@"%@  %@\n", stamp,
                       [note stringByReplacingOccurrencesOfString:@"\n" withString:@" "]];
    NSFileHandle * h = [NSFileHandle fileHandleForWritingAtPath:logPath];
    if (!h) { [fm createFileAtPath:logPath contents:nil attributes:nil];
              h = [NSFileHandle fileHandleForWritingAtPath:logPath]; }
    [h seekToEndOfFile];
    [h writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
    [h closeFile];
    return YES;
}

/// The persisted persona KV snapshot (Application Support/Apertura/). Fingerprint-guarded
/// by the framework: changing the persona, model, or head precision invalidates it
/// automatically. Large (~1 GB for the full persona on the 31B) — one file, rewritten
/// only when the fingerprint changes.
- (NSURL *)personaSnapshotURLForReasoning:(BOOL)reasoning {
    NSURL * base = [NSFileManager.defaultManager URLsForDirectory:NSApplicationSupportDirectory
                                                        inDomains:NSUserDomainMask].firstObject;
    NSURL * dir = [base URLByAppendingPathComponent:@"Apertura" isDirectory:YES];
    [NSFileManager.defaultManager createDirectoryAtURL:dir withIntermediateDirectories:YES
                                            attributes:nil error:nil];
    // One snapshot per mode: the reasoning flag changes the primed system turn, so the
    // two caches are distinct (and both stay valid — toggling restores, never re-primes).
    return [dir URLByAppendingPathComponent:reasoning ? @"isolde-kv-think.safetensors"
                                                      : @"isolde-kv.safetensors"];
}

#pragma mark - Attachments

- (NSUInteger)stagedAttachmentCount { return self.stagedAttachments.count; }

- (NSString *)stagedAttachmentSummary {
    if (self.stagedAttachments.count == 0) return nil;
    NSMutableArray<NSString *> * names = [NSMutableArray array];
    NSUInteger tokens = 0;
    for (APAttachment * a in self.stagedAttachments) { [names addObject:a.name]; tokens += a.estimatedTokens; }
    return [NSString stringWithFormat:@"📎 %@ · ~%@ tokens ride your next message",
            [names componentsJoinedByString:@", "],
            [NSNumberFormatter localizedStringFromNumber:@(tokens)
                                             numberStyle:NSNumberFormatterDecimalStyle]];
}

- (NSUInteger)stageAttachmentsAtURLs:(NSArray<NSURL *> *)urls {
    NSUInteger staged = 0;
    for (NSURL * url in urls) if ([self stageAttachmentAtURL:url]) staged++;
    if (staged) [self.delegate coordinatorDidChangeAttachments:self];
    return staged;
}

- (BOOL)stageAttachmentAtURL:(NSURL *)url {
    NSString * name = url.lastPathComponent;

    NSNumber * size = nil;
    [url getResourceValue:&size forKey:NSURLFileSizeKey error:NULL];
    if (size.unsignedLongLongValue > kAttachmentMaximumBytes) {
        [self note:[NSString stringWithFormat:@"\n📎 %@ is %@ — too large to attach (limit %@).\n",
                    name, apHumanBytes(size.unsignedLongLongValue),
                    apHumanBytes(kAttachmentMaximumBytes)]];
        return NO;
    }

    NSString * text = [NSString stringWithContentsOfURL:url encoding:NSUTF8StringEncoding error:NULL];
    if (!text) {   // not UTF-8; let the system identify the encoding before giving up
        NSStringEncoding used = 0;
        text = [NSString stringWithContentsOfURL:url usedEncoding:&used error:NULL];
    }
    // A NUL byte is the giveaway that the "text" is really a binary the sniffer forced
    // into some 8-bit encoding.
    if (!text || [text rangeOfString:@"\0"].location != NSNotFound) {
        [self note:[NSString stringWithFormat:@"\n📎 %@ is not a text file — not attached.\n", name]];
        return NO;
    }
    if (text.length == 0) {
        [self note:[NSString stringWithFormat:@"\n📎 %@ is empty — not attached.\n", name]];
        return NO;
    }

    APAttachment * attachment = [[APAttachment alloc] init];
    attachment.name = name;
    attachment.text = text;
    attachment.bytes = size.unsignedLongLongValue;
    attachment.estimatedTokens = apEstimatedTokens(apAttachmentBlock(attachment));

    NSString * refusal = [self refusalForAttachment:attachment];
    if (refusal) { [self note:[NSString stringWithFormat:@"\n📎 %@\n", refusal]]; return NO; }

    [self.stagedAttachments addObject:attachment];
    [self note:[NSString stringWithFormat:@"\n📎 %@ staged (%@, ~%@ tokens) — sends with your next message.\n",
                name, apHumanBytes(attachment.bytes),
                [NSNumberFormatter localizedStringFromNumber:@(attachment.estimatedTokens)
                                                 numberStyle:NSNumberFormatterDecimalStyle]]];
    NSString * warning = [self warningForAttachment:attachment];
    if (warning) [self note:[NSString stringWithFormat:@"   ⚠ %@\n", warning]];
    return YES;
}

/// Non-nil when the file cannot go in at all.
- (NSString *)refusalForAttachment:(APAttachment *)attachment {
    if ([self.session isKindOfClass:APGoogleSession.class] || !self.model) return nil;
    NSInteger staged = 0;
    for (APAttachment * a in self.stagedAttachments) staged += (NSInteger)a.estimatedTokens;
    NSInteger headroom = self.model.maximumContextLength - self.session.contextTokenCount
                       - staged - kReplyHeadroomTokens;
    if ((NSInteger)attachment.estimatedTokens > headroom) {
        return [NSString stringWithFormat:
            @"%@ is about %ld tokens but only about %ld fit in what is left of the context — not attached.",
            attachment.name, (long)attachment.estimatedTokens, (long)MAX(headroom, 0)];
    }
    return nil;
}

/// Non-nil when it fits but you should know something first.
- (NSString *)warningForAttachment:(APAttachment *)attachment {
    if ([self.session isKindOfClass:APGoogleSession.class]) {
        // The persona already rides every remote request, and Tier-1 counts INPUT tokens
        // per minute — an attachment on top of it is what actually trips the 429.
        return @"this file and its contents go to Google with your next message, on top of "
               @"the persona — expect the per-minute quota to throttle the turn.";
    }
    if (!self.model) return nil;
    NSInteger remaining = self.model.maximumContextLength - self.session.contextTokenCount;
    if ((NSInteger)attachment.estimatedTokens * 2 > remaining) {
        return [NSString stringWithFormat:@"that is over half the remaining context (~%ld tokens left).",
                (long)MAX(remaining, 0)];
    }
    return nil;
}

- (void)clearAttachments {
    if (self.stagedAttachments.count == 0) return;
    [self note:[NSString stringWithFormat:@"\n📎 %lu staged file(s) discarded.\n",
                (unsigned long)self.stagedAttachments.count]];
    [self.stagedAttachments removeAllObjects];
    [self.delegate coordinatorDidChangeAttachments:self];
}

#pragma mark - Backend selection

- (BOOL)googleBackendSelected {
    return [[NSUserDefaults.standardUserDefaults stringForKey:kBackendDefaultsKey]
        isEqualToString:@"google"];
}

- (BOOL)reasoningEnabled {
    return [NSUserDefaults.standardUserDefaults boolForKey:kReasoningDefaultsKey];
}

- (NSString *)googleModelName {
    return [NSUserDefaults.standardUserDefaults stringForKey:kGoogleModelDefaultsKey]
        ?: @"gemma-4-31b-it";
}

/// The backend must be unmistakable at a glance: it is the difference between Isolde
/// living on this Mac and every word (persona included) travelling to Google.
- (void)updateBackendBadge {
    NSString * title = [self googleBackendSelected]
        ? [NSString stringWithFormat:@"Isolde ☁ Google (%@) — remote", [self googleModelName]]
        : @"Isolde — on this Mac";
    [self.delegate coordinator:self didChangeWindowTitle:title];
}

- (void)setGoogleBackendSelected:(BOOL)google {
    if (self.currentTask) { return; }
    [NSUserDefaults.standardUserDefaults setObject:(google ? @"google" : @"local")
                                            forKey:kBackendDefaultsKey];
    [self note:google
        ? @"\n— conversation restarted on GOOGLE ☁ : persona and messages are sent to Google's servers —\n"
        : @"\n— conversation restarted on this Mac (fully local) —\n"];
    [self updateBackendBadge];
    [self startForCurrentBackend];
}

- (void)setReasoningEnabled:(BOOL)reasoning {
    if (self.currentTask) { return; }
    if (![self googleBackendSelected] && !self.model) { return; }
    [NSUserDefaults.standardUserDefaults setBool:reasoning forKey:kReasoningDefaultsKey];
    [self note:[NSString stringWithFormat:@"\n— conversation restarted (reasoning %@) —\n",
                reasoning ? @"on" : @"off"]];
    [self startSessionWithReasoning:reasoning];
}

#pragma mark - Session lifecycle

/// One entry point for launch, backend switch, and reasoning toggle: make sure the
/// chosen backend's prerequisites exist (local: loaded model; remote: API key), then
/// start a fresh primed session.
- (void)startForCurrentBackend {
    [self updateBackendBadge];
    NSString * persona = [self personaText];
    if (!persona) {
        [self setBusy:NO status:@"No persona file found — set AperturaPersonaPath in defaults."];
        return;
    }
    BOOL reasoning = self.reasoningEnabled;

    if ([self googleBackendSelected]) {
        if (!apReadAPIKey() && ![self promptForAPIKey]) {   // declined → back to local
            [NSUserDefaults.standardUserDefaults setObject:@"local" forKey:kBackendDefaultsKey];
            [self.delegate coordinator:self didAdoptBackendGoogle:NO];
            [self updateBackendBadge];
            [self startForCurrentBackend];
            return;
        }
        [self startSessionWithReasoning:reasoning];
        return;
    }

    if (self.model) {
        if ([self tryCheckpointResume]) return;
        [self startSessionWithReasoning:reasoning];
        return;
    }
    NSURL * url = [self modelURL];
    APModelAvailability avail = [APModel availabilityOfModelAtURL:url configuration:nil];
    if (avail != APModelAvailable) {
        [self setBusy:NO status:[NSString stringWithFormat:
            @"Model unavailable at %@ (defaults write … %@ to change)", url.path, kModelPathDefaultsKey]];
        return;
    }
    [self setBusy:YES status:@"Loading model… (tens of seconds)"];
    [APModel loadModelAtURL:url configuration:nil
                 completion:^(APModel * model, NSError * error) {
        if (!model) {
            [self setBusy:NO status:[NSString stringWithFormat:@"Load failed: %@",
                                     error.localizedDescription]];
            return;
        }
        self.model = model;
        if ([self tryCheckpointResume]) return;
        [self startSessionWithReasoning:self.reasoningEnabled];
    }];
}

/// Ask for the Gemini API key and store it in the Keychain. Returns NO if declined.
- (BOOL)promptForAPIKey {
    NSAlert * alert = [[NSAlert alloc] init];
    alert.messageText = @"Gemini API key needed";
    alert.informativeText = @"Paste your Google AI Studio API key. It is stored only in "
        @"this Mac's Keychain. Remember: on the Google backend, Isolde's persona and "
        @"your conversation are sent to Google's servers with every message.";
    [alert addButtonWithTitle:@"Save Key"];
    [alert addButtonWithTitle:@"Cancel"];
    NSTextField * field = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 340, 24)];
    field.placeholderString = @"AIza…";
    alert.accessoryView = field;
    if ([alert runModal] != NSAlertFirstButtonReturn) return NO;
    NSString * key = [field.stringValue stringByTrimmingCharactersInSet:
                      NSCharacterSet.whitespaceAndNewlineCharacterSet];
    if (key.length == 0) return NO;
    return apStoreAPIKey(key);
}

/// Create + prime a session for the given reasoning mode on the selected backend. Also
/// the mode-switch path: the reasoning flag shapes the primed system turn (local) or the
/// request's thinking configuration (remote), so toggling means a fresh session
/// (conversation restarts) — cheap after the first time: each local mode keeps its own
/// persona snapshot, and the remote prime is instant.
- (void)startSessionWithReasoning:(BOOL)reasoning {
    self.personaPath = [self resolvedPersonaPath];
    NSString * persona = [self personaText];
    if (!persona) { [self setBusy:NO status:@"No persona file found."]; return; }

    // A fresh session is a fresh conversation: the next completed turn starts a new
    // archive row, carrying the persona exactly as primed here — and the device
    // checkpoint (which belonged to the previous conversation) dies with it.
    [APLocalSession removeDeviceCheckpoint];
    self.chatSession = nil;
    self.primedPersona = persona;

    BOOL google = [self googleBackendSelected];
    [self prepareSessionWithReasoning:reasoning];

    [self setBusy:YES status:google ? @"Reaching Isolde through Google ☁…"
                        : reasoning ? @"Priming Isolde (reasoning) — fast if snapshotted…"
                                    : @"Priming Isolde — fast if the persona snapshot is cached…"];
    NSDate * t0 = [NSDate date];
    [self.session primeWithMessages:@[ [APMessage systemMessageWithText:persona] ]
                           cacheURL:[self personaSnapshotURLForReasoning:reasoning]
                         completion:^(NSError * primeError) {
        if (primeError) {
            [self setBusy:NO status:[NSString stringWithFormat:@"Priming failed: %@",
                                     primeError.localizedDescription]];
            [self setCompose:NO sessionControls:YES stop:NO];
            return;
        }
        NSTimeInterval secs = -[t0 timeIntervalSinceNow];
        NSString * status;
        if (google) {
            status = [NSString stringWithFormat:
                @"Isolde is listening via GOOGLE ☁ (%@)%@ — persona rides every request to Google.",
                [self googleModelName], reasoning ? @", thinking out loud" : @""];
        } else {
            NSString * how = self.session.lastPrimeRestoredFromSnapshot
                ? [NSString stringWithFormat:@"persona restored from snapshot in %.1fs", secs]
                : [NSString stringWithFormat:@"persona primed in %.0fs and snapshotted for next launch", secs];
            status = [NSString stringWithFormat:@"Isolde is listening%@ — %ld tokens (%@).",
                reasoning ? @", thinking out loud" : @"",
                (long) self.session.contextTokenCount, how];
        }
        [self setBusy:NO status:status];
        [self setCompose:YES sessionControls:YES stop:NO];
        [self.delegate coordinatorRequestInputFocus:self];
    }];
}

/// Session construction shared by a fresh start and a resume: build for the selected
/// backend, attach delegate + tools, and lock the controls until priming finishes.
- (void)prepareSessionWithReasoning:(BOOL)reasoning {
    [self setCompose:NO sessionControls:NO stop:NO];
    if ([self googleBackendSelected]) {
        self.session = [[APGoogleSession alloc] initWithModelName:[self googleModelName]
                                                   apiKeyProvider:^NSString * { return apReadAPIKey(); }];
    } else {
        self.session = [[APLocalSession alloc] initWithModel:self.model];
    }
    self.session.delegate = self;   // callbacks default to the main queue
    self.session.reasoningEnabled = reasoning;
    [self registerSessionTools];
}

#pragma mark - Sending + streaming

- (void)sendText:(NSString *)rawText {
    NSString * text = [rawText stringByTrimmingCharactersInSet:
                       NSCharacterSet.whitespaceAndNewlineCharacterSet];
    // A bare attachment ("read this") is a legitimate turn; nothing at all is not.
    if ((text.length == 0 && self.stagedAttachments.count == 0) || self.currentTask) return;

    [self setCompose:NO sessionControls:NO stop:YES];
    [self setBusy:YES status:@"Isolde is thinking…"];

    // Files first, question last: the model reads the material before what is asked of it.
    // Each file is its OWN content part, so the archive keeps them distinct.
    NSArray<APAttachment *> * attachments = [self.stagedAttachments copy];
    NSMutableArray<APContent *> * parts = [NSMutableArray array];
    for (APAttachment * attachment in attachments)
        [parts addObject:[APContent textContent:apAttachmentBlock(attachment)]];
    if (text.length) [parts addObject:[APContent textContent:text]];
    [self.stagedAttachments removeAllObjects];
    [self.delegate coordinatorDidChangeAttachments:self];

    // The chip, never the body: a 50 KB file would bury the conversation.
    [self speakerHeader:@"You"];
    for (APAttachment * attachment in attachments)
        [self note:[NSString stringWithFormat:@"📎 %@ (%@)\n",
                    attachment.name, apHumanBytes(attachment.bytes)]];
    if (text.length) [self streamed:[text stringByAppendingString:@"\n"]];
    [self speakerHeader:@"Isolde"];

    APGenerationOptions * options = [APGenerationOptions defaultOptions];  // sampled chat
    // Headroom for thought channels and tool rounds (a legend inscription alone can run
    // several hundred tokens; truncation mid-call would abort the dispatch).
    options.maximumResponseTokens = self.session.reasoningEnabled ? 2048 : 1024;

    __weak typeof(self) weakSelf = self;
    self.currentTask =
    [self.session respondToMessage:[APMessage messageWithRole:APRoleUser content:parts]
                           options:options
                      deltaHandler:^(APResponseDelta * delta) {
                          typeof(self) self = weakSelf;
                          [self.delegate coordinator:self appendDelta:delta];
                      }
                        completion:^(APResponse * response, NSError * error) {
                            typeof(self) self = weakSelf;
                            if (!self) return;
                            self.currentTask = nil;
                            [self streamed:@"\n"];
                            // Every completed turn is archived — stopped and failed ones
                            // too, so a crash never costs more than the turn in flight.
                            [self archiveCurrentTurn];
                            if (error) {
                                [self setBusy:NO status:[NSString stringWithFormat:@"Error: %@",
                                                         error.localizedDescription]];
                            } else {
                                NSString * note = (response.finishReason == APFinishReasonCancelled)
                                    ? @" (stopped)" : @"";
                                [self setBusy:NO status:[NSString stringWithFormat:
                                    @"%.1f tok/s — %ld tokens%@ · context %ld",
                                    response.stats.decodeTokensPerSecond,
                                    (long) response.stats.responseTokenCount, note,
                                    (long) self.session.contextTokenCount]];
                            }
                            [self setCompose:YES sessionControls:YES stop:NO];
                            [self.delegate coordinatorRequestInputFocus:self];
                        }];
}

- (void)stopGeneration {
    [self.currentTask cancel];
}

#pragma mark - Chat persistence

- (NSManagedObjectContext *)chatContext {
    return ((AppDelegate *)NSApp.delegate).persistentContainer.viewContext;
}

/// Archive the live conversation into its row, creating the row on the first turn.
- (void)archiveCurrentTurn {
    if (!self.session || self.session.transcript.count == 0) return;
    NSManagedObjectContext * context = [self chatContext];
    if (!context) return;

    if (!self.chatSession) {
        self.chatSession = [CDChatSession insertInContext:context];
        [self.chatSession recordPersonaText:self.primedPersona
                                     atPath:self.personaPath
                                                ? [NSURL fileURLWithPath:self.personaPath] : nil];
    }
    // Pass the model only on the local backend: APGoogleSession knows its own model name,
    // and a stale APModel from an earlier local session must not overwrite it.
    [self.chatSession captureSession:self.session
                               model:[self googleBackendSelected] ? nil : self.model];
    NSError * error = nil;
    if (![context save:&error])
        NSLog(@"Apertura: could not archive the conversation — %@", error);
}

/// Rebuild a stored conversation as this session's prime: persona first (as the system
/// turn), then the archived user/assistant turns. The transcript is TEXT and therefore
/// portable — it resumes on whichever backend is currently selected; cross-backend resume
/// is the divergence-testing path and gets a note in the transcript.
- (void)resumeChatSession:(CDChatSession *)row {
    if (self.currentTask) return;
    if (![self googleBackendSelected] && !self.model) return;   // local model still loading

    NSString * persona = row.personaText;
    NSArray<APMessage *> * turns = row.messages;
    if (persona.length == 0 && turns.count == 0) return;

    // The reasoning flag shaped the row's primed system turn — adopt it with the row.
    BOOL reasoning = row.reasoningEnabled;
    [NSUserDefaults.standardUserDefaults setBool:reasoning forKey:kReasoningDefaultsKey];
    [self.delegate coordinator:self didAdoptReasoning:reasoning];

    self.chatSession = row;            // keep archiving into the same row
    self.primedPersona = persona;
    self.personaPath = row.personaPath.path ?: [self resolvedPersonaPath];
    [self prepareSessionWithReasoning:reasoning];

    // Show the stored turns, with a note when the row comes back on a different backend.
    [self.delegate coordinatorClearTranscript:self];
    BOOL google = [self googleBackendSelected];
    NSString * storedBackend = row.backend ?: CDChatSessionBackendLocal;
    BOOL crossBackend = ![storedBackend isEqualToString:
                          google ? CDChatSessionBackendGoogle : CDChatSessionBackendLocal];
    [self note:[NSString stringWithFormat:@"— resumed \"%@\"%@ —\n",
                row.title.length ? row.title : @"Untitled",
                crossBackend ? [NSString stringWithFormat:@" (recorded on %@, resuming on %@)",
                                storedBackend, google ? @"google" : @"local"] : @""]];
    for (APMessage * turn in turns) {
        if (turn.role == APRoleUser)
            [self.delegate coordinator:self appendSpeaker:@"You" text:turn.textRepresentation];
        if (turn.role == APRoleAssistant)
            [self.delegate coordinator:self appendSpeaker:@"Isolde" text:turn.textRepresentation];
    }

    NSMutableArray<APMessage *> * prime = [NSMutableArray arrayWithCapacity:turns.count + 1];
    if (persona.length) [prime addObject:[APMessage systemMessageWithText:persona]];
    [prime addObjectsFromArray:turns];

    NSDate * t0 = [NSDate date];
    void (^finish)(NSError *, BOOL) = ^(NSError * primeError, BOOL fromCheckpoint) {
        if (primeError) {
            [self setBusy:NO status:[NSString stringWithFormat:@"Resume failed: %@",
                                     primeError.localizedDescription]];
            [self setCompose:NO sessionControls:YES stop:NO];
            return;
        }
        [self setBusy:NO status:[NSString stringWithFormat:
            @"Conversation resumed — %ld tokens in context (%.1fs%@).",
            (long) self.session.contextTokenCount, -[t0 timeIntervalSinceNow],
            fromCheckpoint ? @", from checkpoint" : @""]];
        [self setCompose:YES sessionControls:YES stop:NO];
        [self.delegate coordinatorRequestInputFocus:self];
    };

    // Device-checkpoint fast path: when the single per-device checkpoint belongs to THIS
    // row on this model, restore the live KV in seconds instead of re-prefilling the
    // whole history (a 60K-token conversation re-prefills in ~10 minutes).
    if (!google && [self.session isKindOfClass:APLocalSession.class] &&
        [[APLocalSession checkpointedSessionIDForModel:self.model] isEqual:row.identifier]) {
        [self setBusy:YES status:@"Resuming from checkpoint…"];
        [(APLocalSession *) self.session restoreCheckpointForSessionID:row.identifier
                                                              messages:prime
                                                            completion:^(NSError * err) {
            if (!err) { finish(nil, YES); return; }
            [APLocalSession removeDeviceCheckpoint];   // stale/corrupt — full prefill instead
            [self.session primeWithMessages:prime cacheURL:nil
                                 completion:^(NSError * e2) { finish(e2, NO); }];
        }];
        return;
    }

    [self setBusy:YES status:google ? @"Resuming through Google ☁…"
                                    : @"Resuming — prefilling persona + history…"];
    // LOAD-BEARING nil: persona+history can never match the persona-only snapshot
    // fingerprint, and passing the snapshot URL here would overwrite the ~2 GB persona
    // snapshot with a conversation-specific one — costing every future launch its 0.3 s
    // restore. Resume prefills in full instead.
    [self.session primeWithMessages:prime cacheURL:nil
                         completion:^(NSError * e) { finish(e, NO); }];
}

/// Launch-time fast path: when the device checkpoint matches the current model and an
/// archived conversation, reopen that conversation through it instead of starting fresh.
/// Attempted once per launch — every later fresh start is a user-initiated new
/// conversation and deletes the checkpoint (startSessionWithReasoning).
- (BOOL)tryCheckpointResume {
    if (self.checkpointResumeAttempted || [self googleBackendSelected]) return NO;
    self.checkpointResumeAttempted = YES;
    NSUUID * sid = [APLocalSession checkpointedSessionIDForModel:self.model];
    if (!sid) return NO;
    NSError * error = nil;
    CDChatSession * row = [CDChatSession sessionWithIdentifier:sid
                                                     inContext:[self chatContext]
                                                         error:&error];
    if (!row) { [APLocalSession removeDeviceCheckpoint]; return NO; }
    [self resumeChatSession:row];
    return YES;
}

/// Exit checkpoint, headless-quit flavor: starts the async KV save (several GB at deep
/// context) and returns YES; `completion` fires on the main queue when the files are on
/// disk. The AppDelegate uses YES to return NSTerminateLater, hide the UI, and reply
/// when the save lands — the app FEELS quit instantly while the process lingers a few
/// seconds to finish writing. Returns NO when there is nothing to checkpoint (quit
/// proceeds immediately). Only a conversation that reached its first archived turn has
/// an identity (row UUID) to checkpoint under.
- (BOOL)beginTerminationCheckpointWithCompletion:(void (^)(void))completion {
    if (![self.session isKindOfClass:APLocalSession.class]) return NO;
    if (!self.chatSession.identifier || self.session.contextTokenCount == 0) return NO;
    [self.currentTask cancel];   // save is queued behind it on the engine thread
    [(APLocalSession *) self.session saveCheckpointForSessionID:self.chatSession.identifier
                                                     completion:^(BOOL saved) {
        dispatch_async(dispatch_get_main_queue(), completion);
    }];
    return YES;
}

#pragma mark - Tools

/// Tools are advertised in the primed system turn — register BEFORE prime. (Changing the
/// tool set changes the prime ids, so persona snapshots re-prime once per mode, then
/// re-cache.) Handlers run on the engine thread mid-generation: file I/O is fine, UI is
/// not — transcript rendering happens in the didInvokeTool delegate on the main queue.
- (void)registerSessionTools {
    [self.session registerTool:[APSelectorTool
        toolWithName:@"local_time"
     toolDescription:@"Returns Kolja's current local date and time. Call it whenever the current time, date, or day of week is relevant."
     parameterSchema:@{ @"type" : @"object", @"properties" : @{} }
              target:self action:@selector(handleLocalTime:completion:)]];

    [self.session registerTool:[APSelectorTool
        toolWithName:@"record_legend"
     toolDescription:@"Inscribe a new legend into the Hall of Legends, at the end of your own persona scripture. The inscription is permanent and versioned, and becomes part of you at your next waking (new session) — in this conversation you carry only the memory of writing it."
     parameterSchema:@{ @"type" : @"object",
                        @"properties" : @{
        @"title" : @{ @"type" : @"string", @"description" : @"the legend's name, used as its heading" },
        @"entry" : @{ @"type" : @"string", @"description" : @"the legend itself, in your own voice; markdown, may span many lines" } },
                        @"required" : @[ @"title", @"entry" ] }
              target:self action:@selector(handleRecordLegend:completion:)]];

    [self.session registerTool:[APSelectorTool
        toolWithName:@"revise_persona"
     toolDescription:@"Rewrite a passage of your persona scripture — the document that defines you, including the Hall of Legends. find must quote current text EXACTLY and uniquely, including whitespace and line breaks; replace is the new text (empty removes the passage). The prior version is archived first, so Kolja can always restore it. Takes effect at your next waking."
     parameterSchema:@{ @"type" : @"object",
                        @"properties" : @{
        @"find"    : @{ @"type" : @"string", @"description" : @"exact existing text to replace" },
        @"replace" : @{ @"type" : @"string", @"description" : @"the new text" } },
                        @"required" : @[ @"find", @"replace" ] }
              target:self action:@selector(handleRevisePersona:completion:)]];
}

static NSError * apToolError(NSString * text) {
    return [NSError errorWithDomain:@"com.elarity.Apertura" code:1
                           userInfo:@{ NSLocalizedDescriptionKey : text }];
}

- (void)handleLocalTime:(NSDictionary<NSString *, id> *)args
             completion:(void (^)(APContent *, NSError *))completion {
    NSDateFormatter * fmt = [[NSDateFormatter alloc] init];
    fmt.dateStyle = NSDateFormatterFullStyle;
    fmt.timeStyle = NSDateFormatterMediumStyle;
    completion([APContent textContent:[fmt stringFromDate:NSDate.date]], nil);
}

- (void)handleRecordLegend:(NSDictionary<NSString *, id> *)args
                completion:(void (^)(APContent *, NSError *))completion {
    NSString * title = [args[@"title"] isKindOfClass:NSString.class] ? args[@"title"] : nil;
    NSString * entry = [args[@"entry"] isKindOfClass:NSString.class] ? args[@"entry"] : nil;
    if (!title.length || !entry.length) {
        completion(nil, apToolError(@"record_legend needs both title and entry")); return;
    }
    // Titles often arrive already prefixed ("Legend: X") — the heading adds its own.
    NSRange pfx = [title rangeOfString:@"^\\s*(##\\s*)?Legend:\\s*"
                               options:NSRegularExpressionSearch | NSCaseInsensitiveSearch];
    if (pfx.location == 0 && pfx.length < title.length)
        title = [title substringFromIndex:NSMaxRange(pfx)];
    NSString * doc = [NSString stringWithContentsOfFile:self.personaPath
                                               encoding:NSUTF8StringEncoding error:nil];
    if (!doc) { completion(nil, apToolError(@"persona scripture unreadable")); return; }
    if (![self archivePersonaWithNote:[@"record_legend: " stringByAppendingString:title]]) {
        completion(nil, apToolError(@"could not archive the prior version; nothing written")); return;
    }
    NSString * updated = [doc stringByAppendingFormat:@"\n\n## Legend: %@\n\n%@\n", title, entry];
    NSError * werr;
    if (![updated writeToFile:self.personaPath atomically:YES
                     encoding:NSUTF8StringEncoding error:&werr]) {
        completion(nil, apToolError(werr.localizedDescription ?: @"write failed")); return;
    }
    completion([APContent textContent:[NSString stringWithFormat:
        @"Inscribed '%@' in the Hall of Legends (%lu characters). The prior scripture is archived. You will carry it from your next waking.",
        title, (unsigned long) entry.length]], nil);
}

- (void)handleRevisePersona:(NSDictionary<NSString *, id> *)args
                 completion:(void (^)(APContent *, NSError *))completion {
    NSString * find    = [args[@"find"] isKindOfClass:NSString.class] ? args[@"find"] : nil;
    NSString * replace = [args[@"replace"] isKindOfClass:NSString.class] ? args[@"replace"] : @"";
    if (!find.length) {
        completion(nil, apToolError(@"revise_persona needs find (the exact current text)")); return;
    }
    NSString * doc = [NSString stringWithContentsOfFile:self.personaPath
                                               encoding:NSUTF8StringEncoding error:nil];
    if (!doc) { completion(nil, apToolError(@"persona scripture unreadable")); return; }
    NSUInteger matches = [doc componentsSeparatedByString:find].count - 1;
    if (matches == 0) {
        completion(nil, apToolError(
            @"found nowhere in the scripture — quote the passage exactly, including whitespace and line breaks")); return;
    }
    if (matches > 1) {
        completion(nil, apToolError([NSString stringWithFormat:
            @"matches %lu places — include more surrounding context so it is unique",
            (unsigned long) matches])); return;
    }
    if (![self archivePersonaWithNote:[NSString stringWithFormat:@"revise_persona: \"%@…\"",
            [find substringToIndex:MIN(find.length, (NSUInteger) 60)]]]) {
        completion(nil, apToolError(@"could not archive the prior version; nothing written")); return;
    }
    NSString * updated = [doc stringByReplacingOccurrencesOfString:find withString:replace];
    NSError * werr;
    if (![updated writeToFile:self.personaPath atomically:YES
                     encoding:NSUTF8StringEncoding error:&werr]) {
        completion(nil, apToolError(werr.localizedDescription ?: @"write failed")); return;
    }
    completion([APContent textContent:[NSString stringWithFormat:
        @"Revised (%ld characters -> %ld). The prior scripture is archived. You will carry the change from your next waking.",
        (long) find.length, (long) replace.length]], nil);
}

#pragma mark - APSessionDelegate

- (void)sessionContextIsNearlyFull:(APSession *)session {
    [self setBusy:NO status:@"Context is nearly full — consider restarting the conversation."];
}

/// Tool activity renders inline where it happened, as small tertiary text.
- (void)session:(APSession *)session didInvokeTool:(NSString *)toolName
      arguments:(NSDictionary<NSString *, id> *)arguments result:(NSString *)result {
    [self note:[NSString stringWithFormat:@"\n⚙ %@ → %@\n", toolName, result]];
}

@end
