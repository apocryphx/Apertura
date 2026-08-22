//
//  RootContainerViewController.m
//  Apertura
//

#import "RootContainerViewController.h"
#import "MainSplitViewController.h"

@implementation RootContainerViewController

- (void)loadView {
    self.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 1000, 620)];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    MainSplitViewController * split = [[MainSplitViewController alloc] init];
    [self addChildViewController:split];
    split.view.frame = self.view.bounds;
    split.view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [self.view addSubview:split.view];
}

@end
