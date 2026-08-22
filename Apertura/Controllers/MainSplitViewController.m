//
//  MainSplitViewController.m
//  Apertura
//

#import "MainSplitViewController.h"
#import "APChatCoordinator.h"
#import "AppDelegate.h"
#import "ChatViewController.h"
#import "PersonaViewController.h"
#import "SidebarViewController.h"
#import "CDChatSession.h"
#import "CDPersona.h"

@interface MainSplitViewController () <SidebarViewControllerDelegate>

@property (nonatomic) APChatCoordinator * coordinator;
@property (nonatomic) SidebarViewController * sidebar;
@property (nonatomic) ChatViewController * chat;
@property (nonatomic) PersonaViewController * personaEditor;
@property (nonatomic) NSViewController * contentContainer;   // hosts chat OR persona editor
@property (nonatomic) BOOL startedLoading;

@end

@implementation MainSplitViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    self.coordinator = [[APChatCoordinator alloc] init];
    ((AppDelegate *) NSApp.delegate).chatCoordinator = self.coordinator;

    self.sidebar = [[SidebarViewController alloc] init];
    self.sidebar.delegate = self;
    self.chat = [[ChatViewController alloc] initWithCoordinator:self.coordinator];
    self.personaEditor = [[PersonaViewController alloc] init];

    self.contentContainer = [[NSViewController alloc] init];
    self.contentContainer.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 760, 560)];
    [self showContent:self.chat];

    NSSplitViewItem * side = [NSSplitViewItem sidebarWithViewController:self.sidebar];
    side.minimumThickness = 200;
    side.canCollapse = YES;
    NSSplitViewItem * content = [NSSplitViewItem splitViewItemWithViewController:self.contentContainer];
    [self addSplitViewItem:side];
    [self addSplitViewItem:content];
}

- (void)viewDidAppear {
    [super viewDidAppear];
    if (!self.startedLoading) {
        self.startedLoading = YES;
        [self.coordinator startForCurrentBackend];
    }
}

/// Swap the content pane's child (chat <-> persona editor).
- (void)showContent:(NSViewController *)child {
    NSViewController * current = self.contentContainer.childViewControllers.firstObject;
    if (current == child) return;
    [current.view removeFromSuperview];
    [current removeFromParentViewController];
    [self.contentContainer addChildViewController:child];
    child.view.frame = self.contentContainer.view.bounds;
    child.view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [self.contentContainer.view addSubview:child.view];
}

#pragma mark - SidebarViewControllerDelegate

- (void)sidebar:(SidebarViewController *)sidebar didSelectChatSession:(CDChatSession *)row {
    [self showContent:self.chat];
    [self.coordinator resumeChatSession:row];
}

- (void)sidebar:(SidebarViewController *)sidebar didSelectPersona:(CDPersona *)persona {
    [self showContent:self.personaEditor];
    [self.personaEditor showPersona:persona];
}

- (void)sidebarDidRequestNewChat:(SidebarViewController *)sidebar {
    [self showContent:self.chat];
    [self.coordinator startForCurrentBackend];
}

@end
