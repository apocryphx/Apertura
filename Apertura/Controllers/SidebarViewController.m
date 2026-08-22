//
//  SidebarViewController.m
//  Apertura
//

#import "SidebarViewController.h"
#import "APPersistence.h"
#import "CDChatSession.h"
#import "CDPersona.h"

// Outline items: two static group markers; children are the managed objects themselves.
static NSString * const kGroupGPTs  = @"CUSTOM GPTS";
static NSString * const kGroupChats = @"CHATS";

@interface SidebarViewController () <NSOutlineViewDataSource, NSOutlineViewDelegate>

@property (nonatomic) NSOutlineView * outline;
@property (nonatomic) NSArray<CDPersona *> * personas;
@property (nonatomic) NSArray<CDChatSession *> * chats;

@end

@implementation SidebarViewController

- (void)loadView {
    self.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 240, 560)];
}

- (void)viewDidLoad {
    [super viewDidLoad];

    NSScrollView * scroll = [[NSScrollView alloc] init];
    scroll.hasVerticalScroller = YES;
    scroll.drawsBackground = NO;
    scroll.translatesAutoresizingMaskIntoConstraints = NO;

    self.outline = [[NSOutlineView alloc] init];
    self.outline.headerView = nil;
    self.outline.style = NSTableViewStyleSourceList;
    self.outline.floatsGroupRows = NO;
    self.outline.rowSizeStyle = NSTableViewRowSizeStyleDefault;
    NSTableColumn * column = [[NSTableColumn alloc] initWithIdentifier:@"main"];
    column.resizingMask = NSTableColumnAutoresizingMask;
    [self.outline addTableColumn:column];
    self.outline.outlineTableColumn = column;
    self.outline.dataSource = self;
    self.outline.delegate = self;
    self.outline.target = self;
    self.outline.action = @selector(rowClicked:);
    scroll.documentView = self.outline;

    NSButton * newChat = [NSButton buttonWithTitle:@"New Chat"
                                            target:self action:@selector(newChat:)];
    newChat.bezelStyle = NSBezelStyleRounded;
    newChat.translatesAutoresizingMaskIntoConstraints = NO;

    NSButton * newPersona = [NSButton buttonWithTitle:@"New GPT"
                                               target:self action:@selector(newPersona:)];
    newPersona.bezelStyle = NSBezelStyleRounded;
    newPersona.translatesAutoresizingMaskIntoConstraints = NO;

    [self.view addSubview:scroll];
    [self.view addSubview:newChat];
    [self.view addSubview:newPersona];

    [NSLayoutConstraint activateConstraints:@[
        [self.view.widthAnchor constraintGreaterThanOrEqualToConstant:200],
        [scroll.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [scroll.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [scroll.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [newChat.topAnchor constraintEqualToAnchor:scroll.bottomAnchor constant:8],
        [newChat.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:10],
        [newPersona.centerYAnchor constraintEqualToAnchor:newChat.centerYAnchor],
        [newPersona.leadingAnchor constraintEqualToAnchor:newChat.trailingAnchor constant:6],
        [newChat.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor constant:-10],
    ]];

    // Refresh on our own saves, on other processes' writes, and on activation (the
    // remote-change notification is file-coordination based and can lag).
    NSManagedObjectContext * moc = APPersistence.sharedContainer.viewContext;
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(dataChanged:)
                                               name:NSManagedObjectContextObjectsDidChangeNotification
                                             object:moc];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(dataChanged:)
                                               name:NSPersistentStoreRemoteChangeNotification
                                             object:APPersistence.sharedContainer.persistentStoreCoordinator];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(dataChanged:)
                                               name:NSApplicationDidBecomeActiveNotification
                                             object:nil];
    [self reloadData];
}

- (void)dealloc { [NSNotificationCenter.defaultCenter removeObserver:self]; }

- (void)dataChanged:(NSNotification *)note {
    dispatch_async(dispatch_get_main_queue(), ^{ [self reloadData]; });
}

- (void)reloadData {
    NSManagedObjectContext * moc = APPersistence.sharedContainer.viewContext;
    self.personas = [moc executeFetchRequest:CDPersona.currentPersonasFetchRequest error:nil] ?: @[];
    self.chats = [moc executeFetchRequest:CDChatSession.recentSessionsFetchRequest error:nil] ?: @[];
    [self.outline reloadData];
    [self.outline expandItem:nil expandChildren:YES];
}

#pragma mark - Actions

- (void)newChat:(id)sender { [self.delegate sidebarDidRequestNewChat:self]; }

- (void)newPersona:(id)sender {
    NSManagedObjectContext * moc = APPersistence.sharedContainer.viewContext;
    CDPersona * persona = [CDPersona insertInContext:moc];
    persona.name = @"New Persona";
    [persona updateBody:@""];
    [moc save:nil];
    [self reloadData];
    [self.delegate sidebar:self didSelectPersona:persona];
}

- (void)rowClicked:(id)sender {
    id item = [self.outline itemAtRow:self.outline.clickedRow];
    if ([item isKindOfClass:CDChatSession.class])
        [self.delegate sidebar:self didSelectChatSession:item];
    else if ([item isKindOfClass:CDPersona.class])
        [self.delegate sidebar:self didSelectPersona:item];
}

#pragma mark - Outline data source

- (NSInteger)outlineView:(NSOutlineView *)outlineView numberOfChildrenOfItem:(id)item {
    if (item == nil) return 2;
    if (item == kGroupGPTs)  return (NSInteger)self.personas.count;
    if (item == kGroupChats) return (NSInteger)self.chats.count;
    return 0;
}

- (id)outlineView:(NSOutlineView *)outlineView child:(NSInteger)index ofItem:(id)item {
    if (item == nil) return index == 0 ? kGroupGPTs : kGroupChats;
    if (item == kGroupGPTs)  return self.personas[(NSUInteger)index];
    return self.chats[(NSUInteger)index];
}

- (BOOL)outlineView:(NSOutlineView *)outlineView isItemExpandable:(id)item {
    return item == kGroupGPTs || item == kGroupChats;
}

- (BOOL)outlineView:(NSOutlineView *)outlineView isGroupItem:(id)item {
    return item == kGroupGPTs || item == kGroupChats;
}

- (BOOL)outlineView:(NSOutlineView *)outlineView shouldSelectItem:(id)item {
    return !(item == kGroupGPTs || item == kGroupChats);
}

#pragma mark - Outline delegate

- (NSView *)outlineView:(NSOutlineView *)outlineView viewForTableColumn:(NSTableColumn *)column
                   item:(id)item {
    NSTableCellView * cell = [outlineView makeViewWithIdentifier:@"cell" owner:self];
    if (!cell) {
        cell = [[NSTableCellView alloc] init];
        cell.identifier = @"cell";
        NSTextField * label = [NSTextField labelWithString:@""];
        label.lineBreakMode = NSLineBreakByTruncatingTail;
        label.translatesAutoresizingMaskIntoConstraints = NO;
        [cell addSubview:label];
        cell.textField = label;
        [NSLayoutConstraint activateConstraints:@[
            [label.leadingAnchor constraintEqualToAnchor:cell.leadingAnchor constant:2],
            [label.trailingAnchor constraintEqualToAnchor:cell.trailingAnchor constant:-2],
            [label.centerYAnchor constraintEqualToAnchor:cell.centerYAnchor],
        ]];
    }

    if (item == kGroupGPTs || item == kGroupChats) {
        cell.textField.stringValue = item;
        cell.textField.font = [NSFont systemFontOfSize:11 weight:NSFontWeightSemibold];
        cell.textField.textColor = NSColor.secondaryLabelColor;
    } else if ([item isKindOfClass:CDPersona.class]) {
        CDPersona * persona = item;
        cell.textField.stringValue = persona.name.length ? persona.name : @"Untitled";
        cell.textField.font = [NSFont systemFontOfSize:13];
        cell.textField.textColor = NSColor.labelColor;
    } else {
        CDChatSession * row = item;
        NSDateFormatter * f = [[NSDateFormatter alloc] init];
        f.dateStyle = NSDateFormatterShortStyle;
        f.timeStyle = NSDateFormatterShortStyle;
        NSString * badge = [row.backend isEqualToString:CDChatSessionBackendGoogle] ? @" ☁" : @"";
        NSString * checkpoint = row.checkpointDate ? @" ●" : @"";
        cell.textField.stringValue = [NSString stringWithFormat:@"%@%@%@\n%@ · %lld turns",
            row.title.length ? row.title : @"Untitled", badge, checkpoint,
            row.dateModified ? [f stringFromDate:row.dateModified] : @"–", row.messageCount];
        cell.textField.font = [NSFont systemFontOfSize:12];
        cell.textField.textColor = NSColor.labelColor;
        cell.textField.maximumNumberOfLines = 2;
    }
    return cell;
}

- (CGFloat)outlineView:(NSOutlineView *)outlineView heightOfRowByItem:(id)item {
    return [item isKindOfClass:CDChatSession.class] ? 36 : 22;
}

@end
