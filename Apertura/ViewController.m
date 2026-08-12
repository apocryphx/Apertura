//
//  ViewController.m
//  Apertura
//
//  Created by Kolja Wawrowsky on 6/16/26.
//
//  Isolde chat over AperturaKit: load the model at launch, prime the persona ONCE into a
//  persistent APSession (the engine's prefix cache makes every later turn cheap), then
//  stream replies token-by-token into the transcript.
//
//  Every completed turn is archived to Core Data (CDChatSession — one row per conversation,
//  created lazily on the first turn so launches never leave empties). Archiving is
//  capture-only: nothing is replayed into the engine at launch, deliberately — replaying
//  turns into prime would change the primed content and invalidate the ~2 GB persona KV
//  snapshot. "Resume…" re-primes a stored row (persona + turns) as a fresh session with
//  cacheURL:nil, on whichever backend is currently selected.
//
//  Paths are user-defaults-overridable for a research setup:
//    defaults write com.elarity.Apertura AperturaModelPath   /path/to/model.apml
//    defaults write com.elarity.Apertura AperturaPersonaPath /path/to/persona.md
//  The Reasoning checkbox persists as AperturaReasoningEnabled.

#import "ViewController.h"
#import "AppDelegate.h"
#import "CDChatSession.h"
#import <AperturaKit/AperturaKit.h>
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

@interface ViewController () <APSessionDelegate>

@property (nonatomic) APModel * model;
@property (nonatomic) APSession * session;
@property (nonatomic) APResponseTask * currentTask;
@property (nonatomic) BOOL startedLoading;

@property (nonatomic) NSTextView * transcriptView;
@property (nonatomic) NSTextField * inputField;
@property (nonatomic) NSButton * stopButton;
@property (nonatomic) NSTextField * statusLabel;
@property (nonatomic) NSProgressIndicator * spinner;
@property (nonatomic) NSButton * reasoningToggle;
@property (nonatomic) NSPopUpButton * backendPopup;
@property (nonatomic) NSButton * resumeButton;
@property (nonatomic) BOOL lastDeltaWasThought;
@property (nonatomic) NSString * personaPath;   // resolved at session start; tool handlers write here

// Chat persistence: the Core Data row this conversation archives into (created lazily on
// the first completed turn) and the persona exactly as primed — the self-editing tools
// may rewrite the persona file mid-conversation, but the row must record the waking of
// Isolde that actually ran.
@property (nonatomic) CDChatSession * chatSession;
@property (nonatomic) NSString * primedPersona;

@end

@implementation ViewController

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

#pragma mark - UI construction

- (void)viewDidLoad {
    [super viewDidLoad];

    NSScrollView * scroll = [NSTextView scrollableTextView];
    self.transcriptView = scroll.documentView;
    self.transcriptView.editable = NO;
    self.transcriptView.richText = YES;
    self.transcriptView.textContainerInset = NSMakeSize(12, 12);
    scroll.translatesAutoresizingMaskIntoConstraints = NO;

    self.inputField = [[NSTextField alloc] init];
    self.inputField.placeholderString = @"Say something to Isolde…";
    self.inputField.font = [NSFont systemFontOfSize:13];
    self.inputField.target = self;
    self.inputField.action = @selector(sendMessage:);
    self.inputField.enabled = NO;
    self.inputField.translatesAutoresizingMaskIntoConstraints = NO;

    self.stopButton = [NSButton buttonWithTitle:@"Stop" target:self action:@selector(stopGeneration:)];
    self.stopButton.enabled = NO;
    self.stopButton.translatesAutoresizingMaskIntoConstraints = NO;

    self.statusLabel = [NSTextField labelWithString:@""];
    self.statusLabel.font = [NSFont systemFontOfSize:11];
    self.statusLabel.textColor = NSColor.secondaryLabelColor;
    self.statusLabel.lineBreakMode = NSLineBreakByTruncatingTail;
    self.statusLabel.translatesAutoresizingMaskIntoConstraints = NO;

    self.reasoningToggle = [NSButton checkboxWithTitle:@"Reasoning"
                                                 target:self action:@selector(toggleReasoning:)];
    self.reasoningToggle.controlSize = NSControlSizeSmall;
    self.reasoningToggle.state = [NSUserDefaults.standardUserDefaults boolForKey:kReasoningDefaultsKey]
        ? NSControlStateValueOn : NSControlStateValueOff;
    self.reasoningToggle.enabled = NO;
    self.reasoningToggle.translatesAutoresizingMaskIntoConstraints = NO;

    self.backendPopup = [[NSPopUpButton alloc] init];
    self.backendPopup.controlSize = NSControlSizeSmall;
    self.backendPopup.font = [NSFont systemFontOfSize:11];
    [self.backendPopup addItemWithTitle:@"On this Mac"];
    [self.backendPopup addItemWithTitle:@"Google ☁ (remote)"];
    [self.backendPopup selectItemAtIndex:[self googleBackendSelected] ? 1 : 0];
    self.backendPopup.target = self;
    self.backendPopup.action = @selector(backendChanged:);
    self.backendPopup.enabled = NO;
    self.backendPopup.translatesAutoresizingMaskIntoConstraints = NO;

    self.resumeButton = [NSButton buttonWithTitle:@"Resume…"
                                            target:self action:@selector(showResumeMenu:)];
    self.resumeButton.controlSize = NSControlSizeSmall;
    self.resumeButton.font = [NSFont systemFontOfSize:11];
    self.resumeButton.enabled = NO;
    self.resumeButton.translatesAutoresizingMaskIntoConstraints = NO;

    self.spinner = [[NSProgressIndicator alloc] init];
    self.spinner.style = NSProgressIndicatorStyleSpinning;
    self.spinner.controlSize = NSControlSizeSmall;
    self.spinner.displayedWhenStopped = NO;
    self.spinner.translatesAutoresizingMaskIntoConstraints = NO;

    [self.view addSubview:scroll];
    [self.view addSubview:self.inputField];
    [self.view addSubview:self.stopButton];
    [self.view addSubview:self.statusLabel];
    [self.view addSubview:self.spinner];
    [self.view addSubview:self.reasoningToggle];
    [self.view addSubview:self.backendPopup];
    [self.view addSubview:self.resumeButton];

    [NSLayoutConstraint activateConstraints:@[
        [self.view.widthAnchor constraintGreaterThanOrEqualToConstant:560],
        [self.view.heightAnchor constraintGreaterThanOrEqualToConstant:480],

        [scroll.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [scroll.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [scroll.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],

        [self.inputField.topAnchor constraintEqualToAnchor:scroll.bottomAnchor constant:8],
        [self.inputField.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.stopButton.leadingAnchor constraintEqualToAnchor:self.inputField.trailingAnchor constant:8],
        [self.stopButton.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],
        [self.stopButton.centerYAnchor constraintEqualToAnchor:self.inputField.centerYAnchor],

        [self.spinner.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.spinner.centerYAnchor constraintEqualToAnchor:self.statusLabel.centerYAnchor],
        [self.statusLabel.leadingAnchor constraintEqualToAnchor:self.spinner.trailingAnchor constant:6],
        [self.statusLabel.trailingAnchor constraintLessThanOrEqualToAnchor:self.resumeButton.leadingAnchor constant:-8],
        [self.resumeButton.trailingAnchor constraintEqualToAnchor:self.backendPopup.leadingAnchor constant:-10],
        [self.resumeButton.centerYAnchor constraintEqualToAnchor:self.statusLabel.centerYAnchor],
        [self.backendPopup.trailingAnchor constraintEqualToAnchor:self.reasoningToggle.leadingAnchor constant:-10],
        [self.backendPopup.centerYAnchor constraintEqualToAnchor:self.statusLabel.centerYAnchor],
        [self.reasoningToggle.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],
        [self.reasoningToggle.centerYAnchor constraintEqualToAnchor:self.statusLabel.centerYAnchor],
        [self.statusLabel.topAnchor constraintEqualToAnchor:self.inputField.bottomAnchor constant:6],
        [self.statusLabel.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor constant:-8],
    ]];
}

- (void)viewDidAppear {
    [super viewDidAppear];
    [self updateBackendBadge];
    if (!self.startedLoading) {
        self.startedLoading = YES;
        [self startForCurrentBackend];
    }
}

#pragma mark - Backend selection

- (BOOL)googleBackendSelected {
    return [[NSUserDefaults.standardUserDefaults stringForKey:kBackendDefaultsKey]
        isEqualToString:@"google"];
}

- (NSString *)googleModelName {
    return [NSUserDefaults.standardUserDefaults stringForKey:kGoogleModelDefaultsKey]
        ?: @"gemma-4-31b-it";
}

/// The backend must be unmistakable at a glance: it is the difference between Isolde
/// living on this Mac and every word (persona included) travelling to Google.
- (void)updateBackendBadge {
    self.view.window.title = [self googleBackendSelected]
        ? [NSString stringWithFormat:@"Isolde ☁ Google (%@) — remote", [self googleModelName]]
        : @"Isolde — on this Mac";
}

- (void)backendChanged:(id)sender {
    if (self.currentTask) { return; }
    BOOL google = (self.backendPopup.indexOfSelectedItem == 1);
    [NSUserDefaults.standardUserDefaults setObject:(google ? @"google" : @"local")
                                            forKey:kBackendDefaultsKey];
    if (self.transcriptView.string.length > 0) {
        NSDictionary * attrs = @{
            NSFontAttributeName : [NSFont systemFontOfSize:11],
            NSForegroundColorAttributeName : NSColor.tertiaryLabelColor,
        };
        NSString * note = google
            ? @"\n— conversation restarted on GOOGLE ☁ : persona and messages are sent to Google's servers —\n"
            : @"\n— conversation restarted on this Mac (fully local) —\n";
        [self.transcriptView.textStorage appendAttributedString:
            [[NSAttributedString alloc] initWithString:note attributes:attrs]];
        [self scrollToEnd];
    }
    [self updateBackendBadge];
    [self startForCurrentBackend];
}

/// One entry point for launch, backend switch, and reasoning toggle: make sure the
/// chosen backend's prerequisites exist (local: loaded model; remote: API key), then
/// start a fresh primed session.
- (void)startForCurrentBackend {
    NSString * persona = [self personaText];
    if (!persona) {
        [self setBusy:NO status:@"No persona file found — set AperturaPersonaPath in defaults."];
        return;
    }
    BOOL reasoning = [NSUserDefaults.standardUserDefaults boolForKey:kReasoningDefaultsKey];

    if ([self googleBackendSelected]) {
        if (!apReadAPIKey() && ![self promptForAPIKey]) {   // declined → back to local
            [NSUserDefaults.standardUserDefaults setObject:@"local" forKey:kBackendDefaultsKey];
            [self.backendPopup selectItemAtIndex:0];
            [self updateBackendBadge];
            [self startForCurrentBackend];
            return;
        }
        [self startSessionWithReasoning:reasoning];
        return;
    }

    if (self.model) { [self startSessionWithReasoning:reasoning]; return; }
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
        [self startSessionWithReasoning:
            [NSUserDefaults.standardUserDefaults boolForKey:kReasoningDefaultsKey]];
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
    // archive row, carrying the persona exactly as primed here.
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
            self.reasoningToggle.enabled = YES;
            self.backendPopup.enabled = YES;
            self.resumeButton.enabled = YES;
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
        self.inputField.enabled = YES;
        self.reasoningToggle.enabled = YES;
        self.backendPopup.enabled = YES;
        self.resumeButton.enabled = YES;
        [self.view.window makeFirstResponder:self.inputField];
    }];
}

/// Session construction shared by a fresh start and a resume: build for the selected
/// backend, attach delegate + tools, and lock the controls until priming finishes.
- (void)prepareSessionWithReasoning:(BOOL)reasoning {
    self.inputField.enabled = NO;
    self.reasoningToggle.enabled = NO;
    self.backendPopup.enabled = NO;
    self.resumeButton.enabled = NO;
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

- (void)toggleReasoning:(id)sender {
    if (self.currentTask) { return; }
    if (![self googleBackendSelected] && !self.model) { return; }
    BOOL reasoning = (self.reasoningToggle.state == NSControlStateValueOn);
    [NSUserDefaults.standardUserDefaults setBool:reasoning forKey:kReasoningDefaultsKey];
    if (self.transcriptView.string.length > 0) {
        NSDictionary * attrs = @{
            NSFontAttributeName : [NSFont systemFontOfSize:11],
            NSForegroundColorAttributeName : NSColor.tertiaryLabelColor,
        };
        NSString * note = [NSString stringWithFormat:@"\n— conversation restarted (reasoning %@) —\n",
                           reasoning ? @"on" : @"off"];
        [self.transcriptView.textStorage appendAttributedString:
            [[NSAttributedString alloc] initWithString:note attributes:attrs]];
        [self scrollToEnd];
    }
    [self startSessionWithReasoning:reasoning];
}

#pragma mark - Sending + streaming

- (void)sendMessage:(id)sender {
    NSString * text = [self.inputField.stringValue stringByTrimmingCharactersInSet:
                       NSCharacterSet.whitespaceAndNewlineCharacterSet];
    if (text.length == 0 || self.currentTask) return;

    self.inputField.stringValue = @"";
    self.inputField.enabled = NO;
    self.stopButton.enabled = YES;
    self.reasoningToggle.enabled = NO;
    self.backendPopup.enabled = NO;
    self.resumeButton.enabled = NO;
    self.lastDeltaWasThought = NO;
    [self setBusy:YES status:@"Isolde is thinking…"];

    [self appendSpeaker:@"You" text:text];
    [self appendSpeakerHeader:@"Isolde"];

    APGenerationOptions * options = [APGenerationOptions defaultOptions];  // sampled chat
    // Headroom for thought channels and tool rounds (a legend inscription alone can run
    // several hundred tokens; truncation mid-call would abort the dispatch).
    options.maximumResponseTokens = self.session.reasoningEnabled ? 2048 : 1024;

    __weak typeof(self) weakSelf = self;
    self.currentTask =
    [self.session respondToMessage:[APMessage userMessageWithText:text]
                           options:options
                      deltaHandler:^(APResponseDelta * delta) {
                          [weakSelf appendDelta:delta];
                      }
                        completion:^(APResponse * response, NSError * error) {
                            typeof(self) self = weakSelf;
                            if (!self) return;
                            self.currentTask = nil;
                            self.stopButton.enabled = NO;
                            [self appendStreamedText:@"\n"];
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
                            self.inputField.enabled = YES;
                            self.reasoningToggle.enabled = YES;
                            self.backendPopup.enabled = YES;
                            self.resumeButton.enabled = YES;
                            [self.view.window makeFirstResponder:self.inputField];
                        }];
}

- (void)stopGeneration:(id)sender {
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

/// "Resume…" — recent conversations, newest first.
- (void)showResumeMenu:(NSButton *)sender {
    NSManagedObjectContext * context = [self chatContext];
    if (!context) return;
    NSFetchRequest<CDChatSession *> * request = [CDChatSession recentSessionsFetchRequest];
    request.fetchLimit = 20;
    NSError * error = nil;
    NSArray<CDChatSession *> * rows = [context executeFetchRequest:request error:&error];
    if (!rows) { NSLog(@"Apertura: resume fetch failed — %@", error); return; }

    NSMenu * menu = [[NSMenu alloc] init];
    if (rows.count == 0)
        [menu addItemWithTitle:@"No saved conversations" action:nil keyEquivalent:@""].enabled = NO;
    NSDateFormatter * f = [[NSDateFormatter alloc] init];
    f.dateStyle = NSDateFormatterShortStyle;
    f.timeStyle = NSDateFormatterShortStyle;
    for (CDChatSession * row in rows) {
        NSString * title = [NSString stringWithFormat:@"%@ — %@ (%@, %lld turns)",
                            row.title.length ? row.title : @"Untitled",
                            row.dateModified ? [f stringFromDate:row.dateModified] : @"–",
                            [row.backend isEqualToString:CDChatSessionBackendGoogle] ? @"Google ☁" : @"local",
                            row.messageCount];
        NSMenuItem * item = [menu addItemWithTitle:title
                                            action:@selector(resumeMenuItemChosen:)
                                     keyEquivalent:@""];
        item.target = self;
        item.representedObject = row;
    }
    [menu popUpMenuPositioningItem:nil atLocation:NSMakePoint(0, sender.bounds.size.height)
                            inView:sender];
}

- (void)resumeMenuItemChosen:(NSMenuItem *)item {
    CDChatSession * row = item.representedObject;
    if ([row isKindOfClass:CDChatSession.class]) [self resumeChatSession:row];
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
    self.reasoningToggle.state = reasoning ? NSControlStateValueOn : NSControlStateValueOff;

    self.chatSession = row;            // keep archiving into the same row
    self.primedPersona = persona;
    self.personaPath = row.personaPath.path ?: [self resolvedPersonaPath];
    [self prepareSessionWithReasoning:reasoning];

    // Show the stored turns, with a note when the row comes back on a different backend.
    [self.transcriptView.textStorage setAttributedString:[[NSAttributedString alloc] init]];
    BOOL google = [self googleBackendSelected];
    NSString * storedBackend = row.backend ?: CDChatSessionBackendLocal;
    BOOL crossBackend = ![storedBackend isEqualToString:
                          google ? CDChatSessionBackendGoogle : CDChatSessionBackendLocal];
    NSDictionary * noteAttrs = @{ NSFontAttributeName : [NSFont systemFontOfSize:11],
                                  NSForegroundColorAttributeName : NSColor.tertiaryLabelColor };
    NSString * note = [NSString stringWithFormat:@"— resumed \"%@\"%@ —\n",
                       row.title.length ? row.title : @"Untitled",
                       crossBackend ? [NSString stringWithFormat:@" (recorded on %@, resuming on %@)",
                                       storedBackend, google ? @"google" : @"local"] : @""];
    [self.transcriptView.textStorage appendAttributedString:
        [[NSAttributedString alloc] initWithString:note attributes:noteAttrs]];
    for (APMessage * turn in turns) {
        if (turn.role == APRoleUser)      [self appendSpeaker:@"You" text:turn.textRepresentation];
        if (turn.role == APRoleAssistant) [self appendSpeaker:@"Isolde" text:turn.textRepresentation];
    }

    NSMutableArray<APMessage *> * prime = [NSMutableArray arrayWithCapacity:turns.count + 1];
    if (persona.length) [prime addObject:[APMessage systemMessageWithText:persona]];
    [prime addObjectsFromArray:turns];

    [self setBusy:YES status:google ? @"Resuming through Google ☁…"
                                    : @"Resuming — prefilling persona + history…"];
    NSDate * t0 = [NSDate date];
    // LOAD-BEARING nil: persona+history can never match the persona-only snapshot
    // fingerprint, and passing the snapshot URL here would overwrite the ~2 GB persona
    // snapshot with a conversation-specific one — costing every future launch its 0.3 s
    // restore. Resume prefills in full instead.
    [self.session primeWithMessages:prime cacheURL:nil completion:^(NSError * primeError) {
        if (primeError) {
            [self setBusy:NO status:[NSString stringWithFormat:@"Resume failed: %@",
                                     primeError.localizedDescription]];
            self.reasoningToggle.enabled = YES;
            self.backendPopup.enabled = YES;
            self.resumeButton.enabled = YES;
            return;
        }
        [self setBusy:NO status:[NSString stringWithFormat:
            @"Conversation resumed — %ld tokens in context (%.1fs).",
            (long) self.session.contextTokenCount, -[t0 timeIntervalSinceNow]]];
        self.inputField.enabled = YES;
        self.reasoningToggle.enabled = YES;
        self.backendPopup.enabled = YES;
        self.resumeButton.enabled = YES;
        [self.view.window makeFirstResponder:self.inputField];
    }];
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
    self.statusLabel.stringValue = @"Context is nearly full — consider restarting the conversation.";
}

/// Tool activity renders inline where it happened, as small tertiary text.
- (void)session:(APSession *)session didInvokeTool:(NSString *)toolName
      arguments:(NSDictionary<NSString *, id> *)arguments result:(NSString *)result {
    NSString * line = [NSString stringWithFormat:@"\n⚙ %@ → %@\n", toolName, result];
    NSAttributedString * chunk = [[NSAttributedString alloc] initWithString:line attributes:@{
        NSFontAttributeName : [NSFont systemFontOfSize:11],
        NSForegroundColorAttributeName : NSColor.tertiaryLabelColor,
    }];
    [self.transcriptView.textStorage appendAttributedString:chunk];
    [self scrollToEnd];
}

#pragma mark - Transcript rendering

- (void)appendSpeaker:(NSString *)name text:(NSString *)text {
    [self appendSpeakerHeader:name];
    [self appendStreamedText:[text stringByAppendingString:@"\n"]];
}

- (void)appendSpeakerHeader:(NSString *)name {
    BOOL isUser = [name isEqualToString:@"You"];
    NSDictionary * attrs = @{
        NSFontAttributeName : [NSFont boldSystemFontOfSize:13],
        NSForegroundColorAttributeName : isUser ? NSColor.secondaryLabelColor
                                                : NSColor.controlAccentColor,
    };
    NSString * prefix = self.transcriptView.string.length ? @"\n" : @"";
    NSAttributedString * header = [[NSAttributedString alloc]
        initWithString:[NSString stringWithFormat:@"%@%@\n", prefix, name] attributes:attrs];
    [self.transcriptView.textStorage appendAttributedString:header];
    [self scrollToEnd];
}

- (void)appendStreamedText:(NSString *)text {
    if (text.length == 0) return;
    NSAttributedString * chunk = [[NSAttributedString alloc] initWithString:text attributes:@{
        NSFontAttributeName : [NSFont systemFontOfSize:13],
        NSForegroundColorAttributeName : NSColor.labelColor,
    }];
    [self.transcriptView.textStorage appendAttributedString:chunk];
    [self scrollToEnd];
}

/// Streamed delta rendering: reasoning-channel text appears as slanted secondary text
/// (thinking out loud); the answer follows in normal text after a separating line.
- (void)appendDelta:(APResponseDelta *)delta {
    if (delta.isThought) {
        self.lastDeltaWasThought = YES;
        NSAttributedString * chunk = [[NSAttributedString alloc] initWithString:delta.text attributes:@{
            NSFontAttributeName : [NSFont systemFontOfSize:12.5],
            NSObliquenessAttributeName : @0.18,
            NSForegroundColorAttributeName : NSColor.secondaryLabelColor,
        }];
        [self.transcriptView.textStorage appendAttributedString:chunk];
        [self scrollToEnd];
    } else {
        if (self.lastDeltaWasThought) {
            self.lastDeltaWasThought = NO;
            [self appendStreamedText:@"\n\n"];
        }
        [self appendStreamedText:delta.text];
    }
}

- (void)scrollToEnd {
    [self.transcriptView scrollRangeToVisible:NSMakeRange(self.transcriptView.string.length, 0)];
}

#pragma mark - Status helpers

- (void)setBusy:(BOOL)busy status:(NSString *)status {
    self.statusLabel.stringValue = status;
    if (busy) [self.spinner startAnimation:nil];
    else      [self.spinner stopAnimation:nil];
}

@end
