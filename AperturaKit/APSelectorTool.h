//  APSelectorTool — wrap a target/selector as a tool: the Objective-C-native adapter.
//  Register a method, get a tool; dispatch goes through NSInvocation at call time, so
//  the target can be any object (including one loaded at runtime).
//
//  The action must have the shape:
//    - (void)handleArgs:(NSDictionary<NSString *, id> *)arguments
//            completion:(void (^)(APContent *_Nullable result, NSError *_Nullable error))completion;
#import <Foundation/Foundation.h>
#import <AperturaKit/APTool.h>

NS_ASSUME_NONNULL_BEGIN

@interface APSelectorTool : NSObject <APTool>

+ (instancetype)toolWithName:(NSString *)name
             toolDescription:(NSString *)description
             parameterSchema:(NSDictionary<NSString *, id> *)schema
                      target:(id)target
                      action:(SEL)action;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
