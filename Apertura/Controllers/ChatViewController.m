//
//  ChatViewController.m
//  Apertura
//
//  View half of the old ViewController — layout, transcript attributed-text rendering,
//  and control forwarding are ported verbatim; the Resume… button is gone (the sidebar
//  lists conversations now).

#import "ChatViewController.h"

#pragma mark - Drop target

/// The window accepts dropped files; NSViewController's default view does not, so the root
/// view is this instead.
@interface APDropView : NSView
@property (nonatomic, copy) BOOL (^fileDropHandler)(NSArray<NSURL *> * urls);
@end

@implementation APDropView

- (instancetype)initWithFrame:(NSRect)frame {
    if ((self = [super initWithFrame:frame])) {
        [self registerForDraggedTypes:@[ NSPasteboardTypeFileURL ]];
    }
    return self;
}

- (NSArray<NSURL *> *)fileURLsFromDrag:(id<NSDraggingInfo>)sender {
    return [sender.draggingPasteboard readObjectsForClasses:@[ NSURL.class ]
                                                    options:@{ NSPasteboardURLReadingFileURLsOnlyKey : @YES }];
}

- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender {
    return self.fileDropHandler && [self fileURLsFromDrag:sender].count ? NSDragOperationCopy
                                                                        : NSDragOperationNone;
}

- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender {
    NSArray<NSURL *> * urls = [self fileURLsFromDrag:sender];
    return urls.count && self.fileDropHandler ? self.fileDropHandler(urls) : NO;
}

@end

#pragma mark - Chat view controller

@interface ChatViewController ()

@property (readwrite) APChatCoordinator * coordinator;

@property (nonatomic) NSTextView * transcriptView;
@property (nonatomic) NSTextField * inputField;
@property (nonatomic) NSButton * stopButton;
@property (nonatomic) NSTextField * statusLabel;
@property (nonatomic) NSProgressIndicator * spinner;
@property (nonatomic) NSButton * reasoningToggle;
@property (nonatomic) NSPopUpButton * backendPopup;
@property (nonatomic) BOOL lastDeltaWasThought;

@property (nonatomic) NSButton * attachButton;
@property (nonatomic) NSTextField * attachmentsLabel;
@property (nonatomic) NSButton * clearAttachmentsButton;
@property (nonatomic) NSLayoutConstraint * attachmentsBarHeight;
@property (nonatomic) NSLayoutConstraint * attachmentsBarTop;

@end

@implementation ChatViewController

- (instancetype)initWithCoordinator:(APChatCoordinator *)coordinator {
    if ((self = [super initWithNibName:nil bundle:nil])) {
        _coordinator = coordinator;
        coordinator.delegate = self;
    }
    return self;
}

#pragma mark - UI construction

/// A drop-aware root view; everything else is built in -viewDidLoad as usual.
- (void)loadView {
    APDropView * view = [[APDropView alloc] initWithFrame:NSMakeRect(0, 0, 760, 560)];
    __weak typeof(self) weakSelf = self;
    view.fileDropHandler = ^BOOL(NSArray<NSURL *> * urls) {
        return [weakSelf.coordinator stageAttachmentsAtURLs:urls] > 0;
    };
    self.view = view;
}

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
    self.reasoningToggle.state = self.coordinator.reasoningEnabled
        ? NSControlStateValueOn : NSControlStateValueOff;
    self.reasoningToggle.enabled = NO;
    self.reasoningToggle.translatesAutoresizingMaskIntoConstraints = NO;

    self.backendPopup = [[NSPopUpButton alloc] init];
    self.backendPopup.controlSize = NSControlSizeSmall;
    self.backendPopup.font = [NSFont systemFontOfSize:11];
    [self.backendPopup addItemWithTitle:@"On this Mac"];
    [self.backendPopup addItemWithTitle:@"Google ☁ (remote)"];
    [self.backendPopup selectItemAtIndex:self.coordinator.googleBackendSelected ? 1 : 0];
    self.backendPopup.target = self;
    self.backendPopup.action = @selector(backendChanged:);
    self.backendPopup.enabled = NO;
    self.backendPopup.translatesAutoresizingMaskIntoConstraints = NO;

    self.attachButton = [NSButton buttonWithTitle:@"Attach…"
                                            target:self action:@selector(attachFiles:)];
    self.attachButton.toolTip = @"Add text files to the conversation (or drop them on the window)";
    self.attachButton.enabled = NO;
    self.attachButton.translatesAutoresizingMaskIntoConstraints = NO;

    self.attachmentsLabel = [NSTextField labelWithString:@""];
    self.attachmentsLabel.font = [NSFont systemFontOfSize:11];
    self.attachmentsLabel.textColor = NSColor.secondaryLabelColor;
    self.attachmentsLabel.lineBreakMode = NSLineBreakByTruncatingMiddle;
    self.attachmentsLabel.translatesAutoresizingMaskIntoConstraints = NO;

    self.clearAttachmentsButton = [NSButton buttonWithTitle:@"Clear"
                                                     target:self action:@selector(clearAttachments:)];
    self.clearAttachmentsButton.controlSize = NSControlSizeSmall;
    self.clearAttachmentsButton.font = [NSFont systemFontOfSize:11];
    self.clearAttachmentsButton.translatesAutoresizingMaskIntoConstraints = NO;

    self.spinner = [[NSProgressIndicator alloc] init];
    self.spinner.style = NSProgressIndicatorStyleSpinning;
    self.spinner.controlSize = NSControlSizeSmall;
    self.spinner.displayedWhenStopped = NO;
    self.spinner.translatesAutoresizingMaskIntoConstraints = NO;

    [self.view addSubview:scroll];
    [self.view addSubview:self.attachmentsLabel];
    [self.view addSubview:self.clearAttachmentsButton];
    [self.view addSubview:self.inputField];
    [self.view addSubview:self.attachButton];
    [self.view addSubview:self.stopButton];
    [self.view addSubview:self.statusLabel];
    [self.view addSubview:self.spinner];
    [self.view addSubview:self.reasoningToggle];
    [self.view addSubview:self.backendPopup];

    // The attachments bar collapses to nothing when nothing is staged, so the window looks
    // exactly as it did before anyone attached a file.
    self.attachmentsBarTop = [self.attachmentsLabel.topAnchor constraintEqualToAnchor:scroll.bottomAnchor
                                                                             constant:6];
    self.attachmentsBarHeight = [self.attachmentsLabel.heightAnchor constraintEqualToConstant:0];

    [NSLayoutConstraint activateConstraints:@[
        [self.view.widthAnchor constraintGreaterThanOrEqualToConstant:560],
        [self.view.heightAnchor constraintGreaterThanOrEqualToConstant:480],

        [scroll.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [scroll.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [scroll.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],

        self.attachmentsBarTop,
        self.attachmentsBarHeight,
        [self.attachmentsLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.attachmentsLabel.trailingAnchor constraintLessThanOrEqualToAnchor:self.clearAttachmentsButton.leadingAnchor
                                                                       constant:-8],
        [self.clearAttachmentsButton.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],
        [self.clearAttachmentsButton.centerYAnchor constraintEqualToAnchor:self.attachmentsLabel.centerYAnchor],

        [self.inputField.topAnchor constraintEqualToAnchor:self.attachmentsLabel.bottomAnchor constant:8],
        [self.inputField.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.attachButton.leadingAnchor constraintEqualToAnchor:self.inputField.trailingAnchor constant:8],
        [self.attachButton.centerYAnchor constraintEqualToAnchor:self.inputField.centerYAnchor],
        [self.stopButton.leadingAnchor constraintEqualToAnchor:self.attachButton.trailingAnchor constant:8],
        [self.stopButton.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],
        [self.stopButton.centerYAnchor constraintEqualToAnchor:self.inputField.centerYAnchor],

        [self.spinner.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.spinner.centerYAnchor constraintEqualToAnchor:self.statusLabel.centerYAnchor],
        [self.statusLabel.leadingAnchor constraintEqualToAnchor:self.spinner.trailingAnchor constant:6],
        [self.statusLabel.trailingAnchor constraintLessThanOrEqualToAnchor:self.backendPopup.leadingAnchor constant:-8],
        [self.backendPopup.trailingAnchor constraintEqualToAnchor:self.reasoningToggle.leadingAnchor constant:-10],
        [self.backendPopup.centerYAnchor constraintEqualToAnchor:self.statusLabel.centerYAnchor],
        [self.reasoningToggle.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],
        [self.reasoningToggle.centerYAnchor constraintEqualToAnchor:self.statusLabel.centerYAnchor],
        [self.statusLabel.topAnchor constraintEqualToAnchor:self.inputField.bottomAnchor constant:6],
        [self.statusLabel.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor constant:-8],
    ]];

    [self coordinatorDidChangeAttachments:self.coordinator];
}

#pragma mark - Actions (forwarded)

- (void)sendMessage:(id)sender {
    NSString * text = self.inputField.stringValue;
    self.inputField.stringValue = @"";
    self.lastDeltaWasThought = NO;
    [self.coordinator sendText:text];
}

- (void)stopGeneration:(id)sender { [self.coordinator stopGeneration]; }

- (void)toggleReasoning:(id)sender {
    [self.coordinator setReasoningEnabled:(self.reasoningToggle.state == NSControlStateValueOn)];
}

- (void)backendChanged:(id)sender {
    [self.coordinator setGoogleBackendSelected:(self.backendPopup.indexOfSelectedItem == 1)];
}

- (void)attachFiles:(id)sender {
    NSOpenPanel * panel = [NSOpenPanel openPanel];
    panel.allowsMultipleSelection = YES;
    panel.canChooseDirectories = NO;
    panel.message = @"Add text files to the conversation. They ride your next message.";
    panel.prompt = @"Attach";
    // No content-type filter: markdown, source, JSON, CSV and plain text are all wanted,
    // and decoding is the honest gate — anything that is not text fails to read.
    if ([panel runModal] != NSModalResponseOK) return;
    [self.coordinator stageAttachmentsAtURLs:panel.URLs];
}

- (void)clearAttachments:(id)sender { [self.coordinator clearAttachments]; }

#pragma mark - APChatCoordinatorDelegate

- (void)coordinatorClearTranscript:(APChatCoordinator *)coordinator {
    [self.transcriptView.textStorage setAttributedString:[[NSAttributedString alloc] init]];
}

- (void)coordinator:(APChatCoordinator *)coordinator didChangeStatus:(NSString *)status busy:(BOOL)busy {
    self.statusLabel.stringValue = status;
    if (busy) [self.spinner startAnimation:nil];
    else      [self.spinner stopAnimation:nil];
}

- (void)coordinatorDidChangeControls:(APChatCoordinator *)coordinator {
    self.inputField.enabled = coordinator.composeEnabled;
    self.attachButton.enabled = coordinator.composeEnabled;
    self.reasoningToggle.enabled = coordinator.sessionControlsEnabled;
    self.backendPopup.enabled = coordinator.sessionControlsEnabled;
    self.stopButton.enabled = coordinator.stopEnabled;
}

- (void)coordinatorDidChangeAttachments:(APChatCoordinator *)coordinator {
    NSString * summary = coordinator.stagedAttachmentSummary;
    BOOL any = summary != nil;
    self.attachmentsLabel.stringValue = summary ?: @"";
    self.attachmentsLabel.hidden = !any;
    self.clearAttachmentsButton.hidden = !any;
    self.attachmentsBarHeight.constant = any ? 16 : 0;
    self.attachmentsBarTop.constant = any ? 6 : 0;
    self.attachButton.title = any ? [NSString stringWithFormat:@"Attach… (%lu)",
                                     (unsigned long)coordinator.stagedAttachmentCount]
                                  : @"Attach…";
}

- (void)coordinator:(APChatCoordinator *)coordinator didChangeWindowTitle:(NSString *)title {
    self.view.window.title = title;
}

- (void)coordinatorRequestInputFocus:(APChatCoordinator *)coordinator {
    [self.view.window makeFirstResponder:self.inputField];
}

- (void)coordinator:(APChatCoordinator *)coordinator didAdoptReasoning:(BOOL)reasoning {
    self.reasoningToggle.state = reasoning ? NSControlStateValueOn : NSControlStateValueOff;
}

- (void)coordinator:(APChatCoordinator *)coordinator didAdoptBackendGoogle:(BOOL)google {
    [self.backendPopup selectItemAtIndex:google ? 1 : 0];
}

#pragma mark - Transcript rendering (ported verbatim)

- (void)coordinator:(APChatCoordinator *)coordinator appendSpeaker:(NSString *)name text:(NSString *)text {
    [self coordinator:coordinator appendSpeakerHeader:name];
    [self coordinator:coordinator appendStreamedText:[text stringByAppendingString:@"\n"]];
}

- (void)coordinator:(APChatCoordinator *)coordinator appendSpeakerHeader:(NSString *)name {
    if ([name isEqualToString:@"Isolde"]) self.lastDeltaWasThought = NO;
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

- (void)coordinator:(APChatCoordinator *)coordinator appendStreamedText:(NSString *)text {
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
- (void)coordinator:(APChatCoordinator *)coordinator appendDelta:(APResponseDelta *)delta {
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
            [self coordinator:coordinator appendStreamedText:@"\n\n"];
        }
        [self coordinator:coordinator appendStreamedText:delta.text];
    }
}

/// Small tertiary text for machinery: attachment chips, restart notes, tool activity.
- (void)coordinator:(APChatCoordinator *)coordinator appendNote:(NSString *)text {
    NSAttributedString * chunk = [[NSAttributedString alloc] initWithString:text attributes:@{
        NSFontAttributeName : [NSFont systemFontOfSize:11],
        NSForegroundColorAttributeName : NSColor.tertiaryLabelColor,
    }];
    [self.transcriptView.textStorage appendAttributedString:chunk];
    [self scrollToEnd];
}

- (void)scrollToEnd {
    [self.transcriptView scrollRangeToVisible:NSMakeRange(self.transcriptView.string.length, 0)];
}

@end
