//
//  ViewController.h
//  Apertura
//
//  Created by Kolja Wawrowsky on 6/16/26.
//

#import <Cocoa/Cocoa.h>

@interface ViewController : NSViewController

/// Headless-quit checkpoint: starts the async KV-cache save for the live conversation
/// and returns YES (completion fires on the main queue when the write lands), or
/// returns NO when there is nothing to save. Called by the AppDelegate from
/// applicationShouldTerminate: — YES means "reply NSTerminateLater, hide the UI, and
/// call replyToApplicationShouldTerminate: from the completion".
- (BOOL)beginTerminationCheckpointWithCompletion:(void (^)(void))completion;

@end

