//
//  AppDelegate.h
//  Apertura
//
//  Created by Kolja Wawrowsky on 6/16/26.
//

#import <Cocoa/Cocoa.h>
#import <CoreData/CoreData.h>

@class APChatCoordinator;

@interface AppDelegate : NSObject <NSApplicationDelegate>

@property (readonly, strong) NSPersistentContainer *persistentContainer;

/// The chat coordinator, for the headless-quit checkpoint flow (set by
/// MainSplitViewController in -viewDidLoad).
@property (weak, nullable) APChatCoordinator *chatCoordinator;

@end

