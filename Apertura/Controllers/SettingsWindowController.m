//
//  SettingsWindowController.m
//  Apertura
//

#import "SettingsWindowController.h"
#import "APModelRegistry.h"

NSString * const AperturaGenTemperatureKey    = @"AperturaGenTemperature";
NSString * const AperturaGenTopKKey           = @"AperturaGenTopK";
NSString * const AperturaGenTopPKey           = @"AperturaGenTopP";
NSString * const AperturaGenSeedKey           = @"AperturaGenSeed";
NSString * const AperturaGenMaxTokensKey      = @"AperturaGenMaxTokens";
NSString * const AperturaExcludesReasoningKey = @"AperturaExcludesReasoning";

static NSString * apBytesString(unsigned long long bytes) {
    return [NSByteCountFormatter stringFromByteCount:(long long)bytes
                                          countStyle:NSByteCountFormatterCountStyleFile];
}

#pragma mark - Models tab

@interface ModelsSettingsViewController : NSViewController
    <NSTableViewDataSource, NSTableViewDelegate>
@property (nonatomic) NSTableView * table;
@property (nonatomic) NSArray<APInstalledModel *> * models;
@property (nonatomic) NSTextField * infoLabel;
@property (nonatomic) NSPopUpButton * headBitsPopup;
@property (nonatomic) NSPopUpButton * cacheModePopup;
@property (nonatomic) NSTextField * chunkField;
@property (nonatomic) NSTextField * maxContextField;
@property (nonatomic) NSButton * useButton;
@end

@implementation ModelsSettingsViewController

- (void)loadView { self.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 560, 380)]; }

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"Models";

    NSScrollView * scroll = [[NSScrollView alloc] init];
    scroll.hasVerticalScroller = YES;
    scroll.borderType = NSBezelBorder;
    scroll.translatesAutoresizingMaskIntoConstraints = NO;
    self.table = [[NSTableView alloc] init];
    self.table.headerView = nil;
    NSTableColumn * col = [[NSTableColumn alloc] initWithIdentifier:@"m"];
    [self.table addTableColumn:col];
    self.table.dataSource = self;
    self.table.delegate = self;
    scroll.documentView = self.table;

    NSButton * importButton = [NSButton buttonWithTitle:@"Import…"
                                                 target:self action:@selector(importModel:)];
    importButton.translatesAutoresizingMaskIntoConstraints = NO;
    NSButton * removeButton = [NSButton buttonWithTitle:@"Remove"
                                                 target:self action:@selector(removeModel:)];
    removeButton.translatesAutoresizingMaskIntoConstraints = NO;
    self.useButton = [NSButton buttonWithTitle:@"Use This Model"
                                        target:self action:@selector(useModel:)];
    self.useButton.translatesAutoresizingMaskIntoConstraints = NO;

    self.infoLabel = [NSTextField labelWithString:@""];
    self.infoLabel.font = [NSFont systemFontOfSize:11];
    self.infoLabel.textColor = NSColor.secondaryLabelColor;
    self.infoLabel.lineBreakMode = NSLineBreakByTruncatingMiddle;
    self.infoLabel.translatesAutoresizingMaskIntoConstraints = NO;

    // Load-time configuration form. "Applies at next model load."
    self.headBitsPopup = [[NSPopUpButton alloc] init];
    [self.headBitsPopup addItemsWithTitles:@[ @"8-bit head (shipped)", @"4-bit head (requantize)" ]];
    self.headBitsPopup.translatesAutoresizingMaskIntoConstraints = NO;
    self.headBitsPopup.target = self; self.headBitsPopup.action = @selector(configChanged:);

    self.cacheModePopup = [[NSPopUpButton alloc] init];
    [self.cacheModePopup addItemsWithTitles:@[ @"Standard KV cache",
                                               @"Raw (half depth bytes)",
                                               @"Raw Q8 (quarter bytes)" ]];
    self.cacheModePopup.translatesAutoresizingMaskIntoConstraints = NO;
    self.cacheModePopup.target = self; self.cacheModePopup.action = @selector(configChanged:);

    self.chunkField = [[NSTextField alloc] init];
    self.chunkField.placeholderString = @"512";
    self.chunkField.translatesAutoresizingMaskIntoConstraints = NO;
    self.chunkField.target = self; self.chunkField.action = @selector(configChanged:);

    self.maxContextField = [[NSTextField alloc] init];
    self.maxContextField.placeholderString = @"0 = model max";
    self.maxContextField.translatesAutoresizingMaskIntoConstraints = NO;
    self.maxContextField.target = self; self.maxContextField.action = @selector(configChanged:);

    NSTextField * chunkLabel = [NSTextField labelWithString:@"Prefill chunk"];
    chunkLabel.translatesAutoresizingMaskIntoConstraints = NO;
    NSTextField * ctxLabel = [NSTextField labelWithString:@"Max context"];
    ctxLabel.translatesAutoresizingMaskIntoConstraints = NO;
    NSTextField * applyLabel = [NSTextField labelWithString:@"Configuration applies at the next model load."];
    applyLabel.font = [NSFont systemFontOfSize:11];
    applyLabel.textColor = NSColor.tertiaryLabelColor;
    applyLabel.translatesAutoresizingMaskIntoConstraints = NO;

    for (NSView * v in @[ scroll, importButton, removeButton, self.useButton, self.infoLabel,
                          self.headBitsPopup, self.cacheModePopup, self.chunkField,
                          self.maxContextField, chunkLabel, ctxLabel, applyLabel ])
        [self.view addSubview:v];

    [NSLayoutConstraint activateConstraints:@[
        [scroll.topAnchor constraintEqualToAnchor:self.view.topAnchor constant:12],
        [scroll.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [scroll.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],
        [scroll.heightAnchor constraintEqualToConstant:140],

        [importButton.topAnchor constraintEqualToAnchor:scroll.bottomAnchor constant:8],
        [importButton.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [removeButton.centerYAnchor constraintEqualToAnchor:importButton.centerYAnchor],
        [removeButton.leadingAnchor constraintEqualToAnchor:importButton.trailingAnchor constant:8],
        [self.useButton.centerYAnchor constraintEqualToAnchor:importButton.centerYAnchor],
        [self.useButton.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],

        [self.infoLabel.topAnchor constraintEqualToAnchor:importButton.bottomAnchor constant:10],
        [self.infoLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.infoLabel.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-12],

        [self.headBitsPopup.topAnchor constraintEqualToAnchor:self.infoLabel.bottomAnchor constant:10],
        [self.headBitsPopup.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.headBitsPopup.widthAnchor constraintEqualToConstant:220],
        [self.cacheModePopup.centerYAnchor constraintEqualToAnchor:self.headBitsPopup.centerYAnchor],
        [self.cacheModePopup.leadingAnchor constraintEqualToAnchor:self.headBitsPopup.trailingAnchor constant:8],
        [self.cacheModePopup.widthAnchor constraintEqualToConstant:220],

        [chunkLabel.topAnchor constraintEqualToAnchor:self.headBitsPopup.bottomAnchor constant:10],
        [chunkLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [self.chunkField.centerYAnchor constraintEqualToAnchor:chunkLabel.centerYAnchor],
        [self.chunkField.leadingAnchor constraintEqualToAnchor:chunkLabel.trailingAnchor constant:8],
        [self.chunkField.widthAnchor constraintEqualToConstant:80],
        [ctxLabel.centerYAnchor constraintEqualToAnchor:chunkLabel.centerYAnchor],
        [ctxLabel.leadingAnchor constraintEqualToAnchor:self.chunkField.trailingAnchor constant:16],
        [self.maxContextField.centerYAnchor constraintEqualToAnchor:chunkLabel.centerYAnchor],
        [self.maxContextField.leadingAnchor constraintEqualToAnchor:ctxLabel.trailingAnchor constant:8],
        [self.maxContextField.widthAnchor constraintEqualToConstant:100],

        [applyLabel.topAnchor constraintEqualToAnchor:chunkLabel.bottomAnchor constant:10],
        [applyLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12],
        [applyLabel.bottomAnchor constraintLessThanOrEqualToAnchor:self.view.bottomAnchor constant:-12],
    ]];

    [self reload];
}

- (void)reload {
    self.models = [APModelRegistry installedModels];
    [self.table reloadData];
    [self selectionChanged];
}

- (nullable APInstalledModel *)selectedModel {
    NSInteger row = self.table.selectedRow;
    return (row >= 0 && row < (NSInteger)self.models.count) ? self.models[(NSUInteger)row] : nil;
}

- (void)selectionChanged {
    APInstalledModel * m = [self selectedModel];
    NSString * selectedName = [APModelRegistry selectedModelName];
    if (!m) {
        NSURL * legacy = [APModelRegistry resolvedModelURL];
        self.infoLabel.stringValue = self.models.count
            ? @"Select a model to configure it."
            : (legacy ? [NSString stringWithFormat:@"No registered models — using legacy path %@", legacy.path]
                      : @"No models registered. Import a .apml bundle.");
        self.useButton.enabled = NO;
        return;
    }
    self.useButton.enabled = ![m.name isEqualToString:selectedName] || self.models.count > 1;
    self.infoLabel.stringValue = [NSString stringWithFormat:@"%@ · %@%@%@",
        m.name, apBytesString(m.sizeBytes),
        m.isSymlink ? @" · linked in place" : @" · copied",
        [m.name isEqualToString:selectedName] ? @" · ACTIVE" : @""];
    APModelConfiguration * config = [APModelRegistry configurationForModelNamed:m.name];
    [self.headBitsPopup selectItemAtIndex:(config.headBits == 4) ? 1 : 0];
    [self.cacheModePopup selectItemAtIndex:MIN(MAX(config.globalKVCacheMode, 0), 2)];
    self.chunkField.integerValue = config.prefillChunkLength;
    self.maxContextField.integerValue = config.maximumContextLength;
}

- (void)configChanged:(id)sender {
    APInstalledModel * m = [self selectedModel];
    if (!m) return;
    APModelConfiguration * config = [[APModelConfiguration alloc] init];
    config.headBits = (self.headBitsPopup.indexOfSelectedItem == 1) ? 4 : 8;
    config.globalKVCacheMode = self.cacheModePopup.indexOfSelectedItem;
    config.prefillChunkLength = self.chunkField.integerValue;
    config.maximumContextLength = self.maxContextField.integerValue;
    [APModelRegistry setConfiguration:config forModelNamed:m.name];
}

- (void)importModel:(id)sender {
    NSOpenPanel * panel = [NSOpenPanel openPanel];
    panel.canChooseDirectories = YES;    // .apml bundles and HF snapshot folders
    panel.allowsMultipleSelection = NO;
    panel.message = @"Choose a .apml model bundle.";
    panel.prompt = @"Register";
    NSButton * copyBox = [NSButton checkboxWithTitle:@"Copy into Library (large!)" target:nil action:nil];
    copyBox.state = NSControlStateValueOff;   // default: link in place, no multi-GB copy
    panel.accessoryView = copyBox;
    panel.accessoryViewDisclosed = YES;
    if ([panel runModal] != NSModalResponseOK || !panel.URL) return;
    BOOL copy = (copyBox.state == NSControlStateValueOn);
    [APModelRegistry importModelAtURL:panel.URL copy:copy
                           completion:^(APInstalledModel * m, NSError * error) {
        if (error) { [[NSAlert alertWithError:error] runModal]; return; }
        if (![APModelRegistry selectedModelName]) [APModelRegistry setSelectedModelName:m.name];
        [self reload];
    }];
}

- (void)removeModel:(id)sender {
    APInstalledModel * m = [self selectedModel];
    if (!m) return;
    BOOL deleteFiles = NO;
    if (!m.isSymlink) {
        NSAlert * alert = [[NSAlert alloc] init];
        alert.messageText = [NSString stringWithFormat:@"Remove %@?", m.name];
        alert.informativeText = @"This model was copied into the Library. Removing it deletes "
                                @"those files.";
        [alert addButtonWithTitle:@"Delete Files"];
        [alert addButtonWithTitle:@"Cancel"];
        if ([alert runModal] != NSAlertFirstButtonReturn) return;
        deleteFiles = YES;
    }
    [APModelRegistry removeModelNamed:m.name deleteFiles:deleteFiles];
    [self reload];
}

- (void)useModel:(id)sender {
    APInstalledModel * m = [self selectedModel];
    if (!m) return;
    [APModelRegistry setSelectedModelName:m.name];
    [self selectionChanged];
}

#pragma mark table

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView { return (NSInteger)self.models.count; }

- (NSView *)tableView:(NSTableView *)tableView viewForTableColumn:(NSTableColumn *)column row:(NSInteger)row {
    NSTableCellView * cell = [tableView makeViewWithIdentifier:@"cell" owner:self];
    if (!cell) {
        cell = [[NSTableCellView alloc] init];
        cell.identifier = @"cell";
        NSTextField * label = [NSTextField labelWithString:@""];
        label.translatesAutoresizingMaskIntoConstraints = NO;
        [cell addSubview:label];
        cell.textField = label;
        [NSLayoutConstraint activateConstraints:@[
            [label.leadingAnchor constraintEqualToAnchor:cell.leadingAnchor constant:4],
            [label.trailingAnchor constraintEqualToAnchor:cell.trailingAnchor constant:-4],
            [label.centerYAnchor constraintEqualToAnchor:cell.centerYAnchor],
        ]];
    }
    APInstalledModel * m = self.models[(NSUInteger)row];
    BOOL active = [m.name isEqualToString:[APModelRegistry selectedModelName]];
    cell.textField.stringValue = [NSString stringWithFormat:@"%@%@ — %@",
        active ? @"● " : @"", m.name, apBytesString(m.sizeBytes)];
    return cell;
}

- (void)tableViewSelectionDidChange:(NSNotification *)notification { [self selectionChanged]; }

@end

#pragma mark - Generation tab

@interface GenerationSettingsViewController : NSViewController
@property (nonatomic) NSTextField * tempField, * topKField, * topPField, * seedField, * maxField;
@property (nonatomic) NSButton * reasoningBox, * excludeBox;
@end

@implementation GenerationSettingsViewController

- (void)loadView { self.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 560, 380)]; }

- (NSTextField *)fieldWithLabel:(NSString *)label below:(NSView *)anchor {
    NSTextField * l = [NSTextField labelWithString:label];
    l.translatesAutoresizingMaskIntoConstraints = NO;
    NSTextField * f = [[NSTextField alloc] init];
    f.translatesAutoresizingMaskIntoConstraints = NO;
    f.target = self; f.action = @selector(save:);
    [self.view addSubview:l];
    [self.view addSubview:f];
    [NSLayoutConstraint activateConstraints:@[
        [l.topAnchor constraintEqualToAnchor:anchor ? anchor.bottomAnchor : self.view.topAnchor
                                    constant:anchor ? 12 : 16],
        [l.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16],
        [l.widthAnchor constraintEqualToConstant:170],
        [f.centerYAnchor constraintEqualToAnchor:l.centerYAnchor],
        [f.leadingAnchor constraintEqualToAnchor:l.trailingAnchor constant:8],
        [f.widthAnchor constraintEqualToConstant:120],
    ]];
    return f;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"Generation";

    self.tempField = [self fieldWithLabel:@"Temperature (0 = greedy)" below:nil];
    self.topKField = [self fieldWithLabel:@"Top-K (0 = off)" below:self.tempField];
    self.topPField = [self fieldWithLabel:@"Top-P (1 = off)" below:self.topKField];
    self.seedField = [self fieldWithLabel:@"Seed (0 = default)" below:self.topPField];
    self.maxField  = [self fieldWithLabel:@"Max tokens (0 = auto)" below:self.seedField];

    self.reasoningBox = [NSButton checkboxWithTitle:@"Reasoning on for new conversations"
                                              target:self action:@selector(save:)];
    self.reasoningBox.translatesAutoresizingMaskIntoConstraints = NO;
    self.excludeBox = [NSButton checkboxWithTitle:@"Exclude reasoning traces from context (recommended)"
                                            target:self action:@selector(save:)];
    self.excludeBox.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.reasoningBox];
    [self.view addSubview:self.excludeBox];
    [NSLayoutConstraint activateConstraints:@[
        [self.reasoningBox.topAnchor constraintEqualToAnchor:self.maxField.bottomAnchor constant:16],
        [self.reasoningBox.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16],
        [self.excludeBox.topAnchor constraintEqualToAnchor:self.reasoningBox.bottomAnchor constant:8],
        [self.excludeBox.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16],
    ]];

    [self load];
}

- (void)load {
    NSUserDefaults * d = NSUserDefaults.standardUserDefaults;
    self.tempField.doubleValue = [d doubleForKey:AperturaGenTemperatureKey];
    self.topKField.integerValue = [d integerForKey:AperturaGenTopKKey];
    self.topPField.doubleValue = [d doubleForKey:AperturaGenTopPKey];
    self.seedField.integerValue = [d integerForKey:AperturaGenSeedKey];
    self.maxField.integerValue = [d integerForKey:AperturaGenMaxTokensKey];
    self.reasoningBox.state = [d boolForKey:@"AperturaReasoningEnabled"]
        ? NSControlStateValueOn : NSControlStateValueOff;
    self.excludeBox.state = [d boolForKey:AperturaExcludesReasoningKey]
        ? NSControlStateValueOn : NSControlStateValueOff;
}

- (void)save:(id)sender {
    NSUserDefaults * d = NSUserDefaults.standardUserDefaults;
    [d setDouble:self.tempField.doubleValue forKey:AperturaGenTemperatureKey];
    [d setInteger:self.topKField.integerValue forKey:AperturaGenTopKKey];
    [d setDouble:self.topPField.doubleValue forKey:AperturaGenTopPKey];
    [d setInteger:self.seedField.integerValue forKey:AperturaGenSeedKey];
    [d setInteger:self.maxField.integerValue forKey:AperturaGenMaxTokensKey];
    [d setBool:(self.reasoningBox.state == NSControlStateValueOn) forKey:@"AperturaReasoningEnabled"];
    [d setBool:(self.excludeBox.state == NSControlStateValueOn) forKey:AperturaExcludesReasoningKey];
}

@end

#pragma mark - Window controller

@implementation SettingsWindowController

+ (void)initialize {
    if (self != SettingsWindowController.class) return;
    [NSUserDefaults.standardUserDefaults registerDefaults:@{
        AperturaGenTemperatureKey    : @0.7,
        AperturaGenTopKKey           : @64,
        AperturaGenTopPKey           : @0.95,
        AperturaGenSeedKey           : @0,
        AperturaGenMaxTokensKey      : @0,
        AperturaExcludesReasoningKey : @YES,
    }];
}

+ (instancetype)sharedController {
    static SettingsWindowController * shared;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSTabViewController * tabs = [[NSTabViewController alloc] init];
        tabs.tabStyle = NSTabViewControllerTabStyleToolbar;
        [tabs addChildViewController:[[ModelsSettingsViewController alloc] init]];
        [tabs addChildViewController:[[GenerationSettingsViewController alloc] init]];
        NSWindow * window = [NSWindow windowWithContentViewController:tabs];
        window.title = @"Settings";
        [window setFrameAutosaveName:@"AperturaSettings"];
        shared = [[SettingsWindowController alloc] initWithWindow:window];
    });
    return shared;
}

- (void)showSettings {
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

@end
