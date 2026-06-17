//
//  AppDelegate.h
//  Apertura
//
//  Created by Kolja Wawrowsky on 6/16/26.
//

#import <Cocoa/Cocoa.h>
#import <CoreData/CoreData.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>

@property (readonly, strong) NSPersistentCloudKitContainer *persistentContainer;


@end

