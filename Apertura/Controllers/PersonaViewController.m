//
//  PersonaViewController.m
//  Apertura
//

#import "PersonaViewController.h"
#import "APPersistence.h"

@interface PersonaViewController ()

@property (nonatomic, nullable) CDPersona * persona;      // the HEAD row being edited
@property (nonatomic, nullable) CDPersona * shownVersion; // head, or a history row (read-only)

@property (nonatomic) NSTextField * nameField;
@property (nonatomic) NSTextView * bodyView;
@property (nonatomic) NSTextField * versionLabel;
@property (nonatomic) NSSegmentedControl * versionStepper;
@property (nonatomic) NSButton * saveButton;
@property (nonatomic) NSButton * importButton;
@property (nonatomic) NSTextField * hintLabel;

@end

@implementation PersonaViewController

- (void)loadView {
    self.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 760, 560)];
}

- (void)viewDidLoad {
    [super viewDidLoad];

    self.nameField = [[NSTextField alloc] init];
    self.nameField.placeholderString = @"Persona name";
    self.nameField.font = [NSFont boldSystemFontOfSize:15];
    self.nameField.translatesAutoresizingMaskIntoConstraints = NO;

    NSScrollView * scroll = [NSTextView scrollableTextView];
    self.bodyView = scroll.documentView;
    self.bodyView.richText = NO;
    self.bodyView.font = [NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular];
    self.bodyView.textContainerInset = NSMakeSize(12, 12);
    self.bodyView.allowsUndo = YES;
    scroll.translatesAutoresizingMaskIntoConstraints = NO;

    // ‹ › through the version chain; index 0 = the editable head.
    self.versionStepper = [NSSegmentedControl segmentedControlWithLabels:@[ @"‹", @"›" ]
                                                            trackingMode:NSSegmentSwitchTrackingMomentary
                                                                  target:self
                                                                  action:@selector(stepVersion:)];
    self.versionStepper.translatesAutoresizingMaskIntoConstraints = NO;

    self.versionLabel = [NSTextField labelWithString:@""];
    self.versionLabel.font = [NSFont systemFontOfSize:11];
    self.versionLabel.textColor = NSColor.secondaryLabelColor;
    self.versionLabel.translatesAutoresizingMaskIntoConstraints = NO;

    self.saveButton = [NSButton buttonWithTitle:@"Save" target:self action:@selector(save:)];
    self.saveButton.keyEquivalent = @"s";
    self.saveButton.keyEquivalentModifierMask = NSEventModifierFlagCommand;
    self.saveButton.translatesAutoresizingMaskIntoConstraints = NO;

    self.importButton = [NSButton buttonWithTitle:@"Import from Files…"
                                            target:self action:@selector(importFiles:)];
    self.importButton.toolTip = @"Append the selected files to the body, joined with the "
                                @"\\n\\n---\\n\\n section separator (selection order)";
    self.importButton.translatesAutoresizingMaskIntoConstraints = NO;

    self.hintLabel = [NSTextField labelWithString:
        @"Saving archives the previous version. Body edits re-prime on the next conversation."];
    self.hintLabel.font = [NSFont systemFontOfSize:11];
    self.hintLabel.textColor = NSColor.tertiaryLabelColor;
    self.hintLabel.lineBreakMode = NSLineBreakByTruncatingTail;
    self.hintLabel.translatesAutoresizingMaskIntoConstraints = NO;

    [self.view addSubview:self.nameField];
    [self.view addSubview:scroll];
    [self.view addSubview:self.versionStepper];
    [self.view addSubview:self.versionLabel];
    [self.view addSubview:self.saveButton];
    [self.view addSubview:self.importButton];
    [self.view addSubview:self.hintLabel];

    [NSLayoutConstraint activateConstraints:@[
        [self.nameField.topAnchor constraintEqualToAnchor:self.view.topAnchor constant:12],
        [self.nameField.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.nameField.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],

        [scroll.topAnchor constraintEqualToAnchor:self.nameField.bottomAnchor constant:8],
        [scroll.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [scroll.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],

        [self.versionStepper.topAnchor constraintEqualToAnchor:scroll.bottomAnchor constant:8],
        [self.versionStepper.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.versionLabel.centerYAnchor constraintEqualToAnchor:self.versionStepper.centerYAnchor],
        [self.versionLabel.leadingAnchor constraintEqualToAnchor:self.versionStepper.trailingAnchor constant:8],
        [self.saveButton.centerYAnchor constraintEqualToAnchor:self.versionStepper.centerYAnchor],
        [self.saveButton.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],
        [self.importButton.centerYAnchor constraintEqualToAnchor:self.versionStepper.centerYAnchor],
        [self.importButton.trailingAnchor constraintEqualToAnchor:self.saveButton.leadingAnchor constant:-8],

        [self.hintLabel.topAnchor constraintEqualToAnchor:self.versionStepper.bottomAnchor constant:6],
        [self.hintLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.hintLabel.trailingAnchor constraintLessThanOrEqualToAnchor:self.view.trailingAnchor constant:-12],
        [self.hintLabel.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor constant:-8],
    ]];

    [self showPersona:nil];
}

#pragma mark - Content

- (void)showPersona:(CDPersona *)persona {
    self.persona = persona;
    [self showVersion:persona];
}

- (void)showVersion:(CDPersona *)version {
    self.shownVersion = version;
    BOOL isHead = (version != nil && version == self.persona);
    self.nameField.stringValue = version.name ?: @"";
    self.bodyView.string = version.body ?: @"";
    self.nameField.editable = isHead;
    self.bodyView.editable = isHead;
    self.saveButton.enabled = isHead;
    self.importButton.enabled = isHead;

    if (!version) {
        self.versionLabel.stringValue = @"";
        self.versionStepper.enabled = NO;
        return;
    }
    NSArray<CDPersona *> * chain = [self.persona versionChain];
    NSUInteger index = [chain indexOfObject:version];
    NSDateFormatter * f = [[NSDateFormatter alloc] init];
    f.dateStyle = NSDateFormatterShortStyle;
    f.timeStyle = NSDateFormatterShortStyle;
    self.versionLabel.stringValue = index == 0
        ? [NSString stringWithFormat:@"current · %lu older version%@", (unsigned long)chain.count - 1,
           chain.count == 2 ? @"" : @"s"]
        : [NSString stringWithFormat:@"version %lu of %lu · %@%@ — read-only",
           (unsigned long)(chain.count - index), (unsigned long)chain.count,
           version.dateModified ? [f stringFromDate:version.dateModified] : @"",
           version.notes.length ? [@" · " stringByAppendingString:version.notes] : @""];
    self.versionStepper.enabled = YES;
    [self.versionStepper setEnabled:(index + 1 < chain.count) forSegment:0];   // ‹ older
    [self.versionStepper setEnabled:(index > 0) forSegment:1];                 // › newer
}

- (void)stepVersion:(NSSegmentedControl *)sender {
    NSArray<CDPersona *> * chain = [self.persona versionChain];
    NSInteger index = (NSInteger)[chain indexOfObject:self.shownVersion];
    index += (sender.selectedSegment == 0) ? 1 : -1;
    if (index < 0 || index >= (NSInteger)chain.count) return;
    [self showVersion:chain[(NSUInteger)index]];
}

#pragma mark - Editing

- (void)save:(id)sender {
    CDPersona * persona = self.persona;
    if (!persona || self.shownVersion != persona) return;
    NSString * newName = self.nameField.stringValue;
    NSString * newBody = self.bodyView.string;
    BOOL changed = ![newBody isEqualToString:(persona.body ?: @"")]
                || ![newName isEqualToString:(persona.name ?: @"")];
    if (!changed) return;
    [persona snapshotBeforeEditWithNote:@"edited in the persona pane" author:@"user"];
    persona.name = newName;
    [persona updateBody:[newBody copy]];
    NSError * error = nil;
    if (![persona.managedObjectContext save:&error])
        NSLog(@"Apertura: persona save failed — %@", error);
    [self showVersion:persona];
}

- (void)importFiles:(id)sender {
    CDPersona * persona = self.persona;
    if (!persona) return;
    NSOpenPanel * panel = [NSOpenPanel openPanel];
    panel.allowsMultipleSelection = YES;
    panel.canChooseDirectories = NO;
    panel.message = @"Selected files are appended to the body in selection order, "
                    @"joined by the section separator.";
    panel.prompt = @"Import";
    if ([panel runModal] != NSModalResponseOK || panel.URLs.count == 0) return;

    NSMutableArray<NSString *> * sections = [NSMutableArray array];
    NSString * existing = self.bodyView.string;
    if (existing.length) [sections addObject:existing];
    for (NSURL * url in panel.URLs) {
        NSString * text = [NSString stringWithContentsOfURL:url
                                                   encoding:NSUTF8StringEncoding error:nil];
        if (text.length) [sections addObject:text];
    }
    self.bodyView.string = [sections componentsJoinedByString:CDPersonaSectionSeparator];
}

@end
