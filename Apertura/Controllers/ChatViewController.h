//
//  ChatViewController.h
//  Apertura
//
//  The chat pane: transcript rendering, input, attachments bar, and the control row —
//  the view half of the old ViewController. All engine/session work lives in
//  APChatCoordinator; this class renders what the coordinator says and forwards what
//  the user does.

#import <Cocoa/Cocoa.h>
#import "APChatCoordinator.h"

NS_ASSUME_NONNULL_BEGIN

@interface ChatViewController : NSViewController <APChatCoordinatorDelegate>

- (instancetype)initWithCoordinator:(APChatCoordinator *)coordinator;

@property (readonly) APChatCoordinator * coordinator;

@end

NS_ASSUME_NONNULL_END
