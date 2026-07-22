#import "APSelectorTool.h"

typedef void (^APToolCompletion)(APContent * _Nullable, NSError * _Nullable);

@implementation APSelectorTool {
    NSString * _name;
    NSString * _toolDescription;
    NSDictionary<NSString *, id> * _parameterSchema;
    id _target;    // strong: the tool keeps its handler alive for the session's lifetime
    SEL _action;
}

@synthesize name = _name;
@synthesize toolDescription = _toolDescription;
@synthesize parameterSchema = _parameterSchema;

+ (instancetype)toolWithName:(NSString *)name
             toolDescription:(NSString *)description
             parameterSchema:(NSDictionary<NSString *, id> *)schema
                      target:(id)target
                      action:(SEL)action {
    APSelectorTool * tool = [[self alloc] initPrivate];
    tool->_name = [name copy];
    tool->_toolDescription = [description copy];
    tool->_parameterSchema = [schema copy];
    tool->_target = target;
    tool->_action = action;
    return tool;
}

- (instancetype)initPrivate { return [super init]; }

- (void)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments
                 completion:(APToolCompletion)completion {
    NSMethodSignature * sig = [_target methodSignatureForSelector:_action];
    if (!sig || sig.numberOfArguments != 4) {   // self, _cmd, args, completion
        completion(nil, [NSError errorWithDomain:@"com.apertura.AperturaKit" code:6
                                        userInfo:@{ NSLocalizedDescriptionKey :
            [NSString stringWithFormat:@"tool '%@': action must take (arguments, completion)", _name] }]);
        return;
    }
    NSInvocation * inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = _target;
    inv.selector = _action;
    NSDictionary * args = arguments;
    APToolCompletion done = [completion copy];
    [inv setArgument:&args atIndex:2];
    [inv setArgument:&done atIndex:3];
    [inv retainArguments];
    [inv invoke];
}

@end
