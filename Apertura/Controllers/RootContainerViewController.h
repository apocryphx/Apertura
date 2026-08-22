//
//  RootContainerViewController.h
//  Apertura
//
//  The storyboard's scene root (replacing the old ViewController there): a plain
//  NSViewController that embeds MainSplitViewController. Exists because the storyboard's
//  scene is a generic <viewController> archetype — an NSSplitViewController cannot be
//  swapped in by customClass alone.

#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

@interface RootContainerViewController : NSViewController
@end

NS_ASSUME_NONNULL_END
