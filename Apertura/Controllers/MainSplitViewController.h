//
//  MainSplitViewController.h
//  Apertura
//
//  The one window: sidebar (Custom GPTs + Chats) beside a content pane that swaps
//  between the chat and the persona editor. Owns the APChatCoordinator and wires the
//  AppDelegate's termination-checkpoint hook to it. Installed by
//  RootContainerViewController (the storyboard's scene root).

#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

@interface MainSplitViewController : NSSplitViewController
@end

NS_ASSUME_NONNULL_END
