//
//  SidebarViewController.h
//  Apertura
//
//  The source list: two groups — CUSTOM GPTS (CDPersona heads) and CHATS (every saved
//  conversation, newest first, uncapped; the old 20-row Resume popup is retired).
//  Selection drives the content pane through the delegate; data refreshes on Core Data
//  changes (this process and remote ones) and on app activation.

#import <Cocoa/Cocoa.h>

@class CDChatSession, CDPersona, SidebarViewController;

NS_ASSUME_NONNULL_BEGIN

@protocol SidebarViewControllerDelegate <NSObject>
- (void)sidebar:(SidebarViewController *)sidebar didSelectChatSession:(CDChatSession *)row;
- (void)sidebar:(SidebarViewController *)sidebar didSelectPersona:(CDPersona *)persona;
- (void)sidebarDidRequestNewChat:(SidebarViewController *)sidebar;
@end

@interface SidebarViewController : NSViewController

@property (weak, nullable) id<SidebarViewControllerDelegate> delegate;

/// Re-fetch and reload (also runs automatically on data changes).
- (void)reloadData;

@end

NS_ASSUME_NONNULL_END
