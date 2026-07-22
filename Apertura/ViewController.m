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
//  Paths are user-defaults-overridable for a research setup:
//    defaults write com.elarity.Apertura AperturaModelPath   /path/to/model.apml
//    defaults write com.elarity.Apertura AperturaPersonaPath /path/to/persona.md
//  The Reasoning checkbox persists as AperturaReasoningEnabled.

#import "ViewController.h"
#import <AperturaKit/AperturaKit.h>

static NSString * const kModelPathDefaultsKey   = @"AperturaModelPath";
static NSString * const kPersonaPathDefaultsKey = @"AperturaPersonaPath";
static NSString * const kReasoningDefaultsKey   = @"AperturaReasoningEnabled";
static NSString * const kModelsDir = @"/Volumes/Macintosh HD/Users/apocryphx/Models";

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
@property (nonatomic) BOOL lastDeltaWasThought;

@end

@implementation ViewController

#pragma mark - Paths

- (NSURL *)modelURL {
    NSString * p = [NSUserDefaults.standardUserDefaults stringForKey:kModelPathDefaultsKey]
        ?: [kModelsDir stringByAppendingPathComponent:@"gemma-4-31b-it-qat-q4.apml"];
    return [NSURL fileURLWithPath:p];
}

- (NSString *)personaText {
    NSString * p = [NSUserDefaults.standardUserDefaults stringForKey:kPersonaPathDefaultsKey];
    NSArray<NSString *> * candidates = p ? @[ p ]
        : @[ [kModelsDir stringByAppendingPathComponent:@"isolde_system.md"],
             [kModelsDir stringByAppendingPathComponent:@"isolde_prompt.txt"] ];
    for (NSString * path in candidates) {
        NSString * text = [NSString stringWithContentsOfFile:path
                                                    encoding:NSUTF8StringEncoding error:nil];
        if (text.length > 0) return text;
    }
    return nil;
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
        [self.statusLabel.trailingAnchor constraintLessThanOrEqualToAnchor:self.reasoningToggle.leadingAnchor constant:-8],
        [self.reasoningToggle.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],
        [self.reasoningToggle.centerYAnchor constraintEqualToAnchor:self.statusLabel.centerYAnchor],
        [self.statusLabel.topAnchor constraintEqualToAnchor:self.inputField.bottomAnchor constant:6],
        [self.statusLabel.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor constant:-8],
    ]];
}

- (void)viewDidAppear {
    [super viewDidAppear];
    self.view.window.title = @"Isolde";
    if (!self.startedLoading) {
        self.startedLoading = YES;
        [self loadAndPrime];
    }
}

#pragma mark - Launch: load model, prime persona

- (void)loadAndPrime {
    NSURL * url = [self modelURL];
    APModelAvailability avail = [APModel availabilityOfModelAtURL:url configuration:nil];
    if (avail != APModelAvailable) {
        [self setBusy:NO status:[NSString stringWithFormat:
            @"Model unavailable at %@ (defaults write … %@ to change)", url.path, kModelPathDefaultsKey]];
        return;
    }
    NSString * persona = [self personaText];
    if (!persona) {
        [self setBusy:NO status:@"No persona file found — set AperturaPersonaPath in defaults."];
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

/// Create + prime a session for the given reasoning mode. Also the mode-switch path:
/// the reasoning flag changes the primed system turn, so toggling means a fresh session
/// (conversation restarts) — cheap after the first time, since each mode keeps its own
/// persona snapshot.
- (void)startSessionWithReasoning:(BOOL)reasoning {
    NSString * persona = [self personaText];
    if (!persona) { [self setBusy:NO status:@"No persona file found."]; return; }

    self.inputField.enabled = NO;
    self.reasoningToggle.enabled = NO;
    self.session = [[APSession alloc] initWithModel:self.model];
    self.session.delegate = self;   // callbacks default to the main queue
    self.session.reasoningEnabled = reasoning;

    // Tools are advertised in the primed system turn — register BEFORE prime. Isolde has
    // no clock; give her one. (Adding a tool changes the prime ids, so existing persona
    // snapshots re-prime once per mode, then re-cache.)
    [self.session registerTool:[APSelectorTool
        toolWithName:@"local_time"
     toolDescription:@"Returns Kolja's current local date and time. Call it whenever the current time, date, or day of week is relevant."
     parameterSchema:@{ @"type" : @"object", @"properties" : @{} }
              target:self action:@selector(handleLocalTime:completion:)]];

    [self setBusy:YES status:reasoning ? @"Priming Isolde (reasoning) — fast if snapshotted…"
                                       : @"Priming Isolde — fast if the persona snapshot is cached…"];
    NSDate * t0 = [NSDate date];
    [self.session primeWithMessages:@[ [APMessage systemMessageWithText:persona] ]
                           cacheURL:[self personaSnapshotURLForReasoning:reasoning]
                         completion:^(NSError * primeError) {
        if (primeError) {
            [self setBusy:NO status:[NSString stringWithFormat:@"Priming failed: %@",
                                     primeError.localizedDescription]];
            self.reasoningToggle.enabled = YES;
            return;
        }
        NSTimeInterval secs = -[t0 timeIntervalSinceNow];
        NSString * how = self.session.lastPrimeRestoredFromSnapshot
            ? [NSString stringWithFormat:@"persona restored from snapshot in %.1fs", secs]
            : [NSString stringWithFormat:@"persona primed in %.0fs and snapshotted for next launch", secs];
        [self setBusy:NO status:[NSString stringWithFormat:
            @"Isolde is listening%@ — %ld tokens (%@).",
            reasoning ? @", thinking out loud" : @"",
            (long) self.session.contextTokenCount, how]];
        self.inputField.enabled = YES;
        self.reasoningToggle.enabled = YES;
        [self.view.window makeFirstResponder:self.inputField];
    }];
}

- (void)toggleReasoning:(id)sender {
    if (self.currentTask || !self.model) { return; }
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
    self.lastDeltaWasThought = NO;
    [self setBusy:YES status:@"Isolde is thinking…"];

    [self appendSpeaker:@"You" text:text];
    [self appendSpeakerHeader:@"Isolde"];

    APGenerationOptions * options = [APGenerationOptions defaultOptions];  // sampled chat
    options.maximumResponseTokens = self.session.reasoningEnabled ? 1536 : 512;

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
                            [self.view.window makeFirstResponder:self.inputField];
                        }];
}

- (void)stopGeneration:(id)sender {
    [self.currentTask cancel];
}

#pragma mark - Tools

- (void)handleLocalTime:(NSDictionary<NSString *, id> *)args
             completion:(void (^)(APContent *, NSError *))completion {
    NSDateFormatter * fmt = [[NSDateFormatter alloc] init];
    fmt.dateStyle = NSDateFormatterFullStyle;
    fmt.timeStyle = NSDateFormatterMediumStyle;
    completion([APContent textContent:[fmt stringFromDate:NSDate.date]], nil);
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
