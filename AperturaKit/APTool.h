//  APTool — the tool protocol. Object-based (not function-based) so tools can carry
//  state and lifecycle — the shape a future memory tool needs.
//
//  Registered tools are advertised in the primed system turn and dispatched live when
//  the model emits a call (see APSession registerTool: for the flow; --tools-verify
//  gates it against the reference grammar).
#import <Foundation/Foundation.h>
#import <AperturaKit/APContent.h>

NS_ASSUME_NONNULL_BEGIN

@class APSession;

@protocol APTool <NSObject>

@property (readonly) NSString *name;                              // e.g. "search_notes"
@property (readonly) NSString *toolDescription;                   // shown to the model
/// JSON-Schema fragment describing the arguments object.
@property (readonly) NSDictionary<NSString *, id> *parameterSchema;

/// Asynchronous by design — tools do IO. The result becomes an APRoleTool message.
- (void)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments
                 completion:(void (^)(APContent *_Nullable result,
                                      NSError *_Nullable error))completion;

@optional
/// Lifecycle for stateful tools (memory stores, connections).
- (void)willAttachToSession:(APSession *)session;
- (void)didDetachFromSession:(APSession *)session;

@end

NS_ASSUME_NONNULL_END
