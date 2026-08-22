//
//  MCPToolRegistry.h
//  apertura-mcp
//
//  The tool table: name + description + JSON-Schema + handler block. MCPServer's
//  tools/list and tools/call both read it; MCPEngineHost fills it.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Returns the tool's text result, or nil with *error set (reported as isError content,
/// never as a JSON-RPC error). Handlers run on the stdio thread and may block.
typedef NSString *_Nullable (^MCPToolHandler)(NSDictionary * arguments,
                                              NSError *_Nullable *_Nullable error);

@interface MCPTool : NSObject
@property (nonatomic, copy) NSString * name;
@property (nonatomic, copy) NSString * toolDescription;
@property (nonatomic, copy) NSDictionary * inputSchema;
@property (nonatomic, copy) MCPToolHandler handler;
@end

@interface MCPToolRegistry : NSObject

- (void)registerToolNamed:(NSString *)name
              description:(NSString *)description
                   schema:(NSDictionary *)schema
                  handler:(MCPToolHandler)handler;

- (NSArray<NSDictionary *> *)toolListPayload;         // for tools/list
- (nullable MCPTool *)toolNamed:(NSString *)name;

@end

NS_ASSUME_NONNULL_END
