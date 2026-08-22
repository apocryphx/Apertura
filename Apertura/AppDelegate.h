//
//  AppDelegate.h
//  Apertura
//
//  Created by Kolja Wawrowsky on 6/16/26.
//

#import <Cocoa/Cocoa.h>
#import <CoreData/CoreData.h>

@class ViewController;

@interface AppDelegate : NSObject <NSApplicationDelegate>

@property (readonly, strong) NSPersistentContainer *persistentContainer;

/// The chat view controller, for the headless-quit checkpoint flow (set in -viewDidLoad).
@property (weak, nullable) ViewController *mainViewController;

@end

