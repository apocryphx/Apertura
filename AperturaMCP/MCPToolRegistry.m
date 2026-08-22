//
//  MCPToolRegistry.m
//  apertura-mcp
//

#import "MCPToolRegistry.h"

@implementation MCPTool
@end

@implementation MCPToolRegistry {
    NSMutableArray<MCPTool *> * _tools;
}

- (instancetype)init {
    if ((self = [super init])) _tools = [NSMutableArray array];
    return self;
}

- (void)registerToolNamed:(NSString *)name
              description:(NSString *)description
                   schema:(NSDictionary *)schema
                  handler:(MCPToolHandler)handler {
    MCPTool * tool = [[MCPTool alloc] init];
    tool.name = name;
    tool.toolDescription = description;
    tool.inputSchema = schema;
    tool.handler = handler;
    [_tools addObject:tool];
}

- (NSArray<NSDictionary *> *)toolListPayload {
    NSMutableArray * list = [NSMutableArray arrayWithCapacity:_tools.count];
    for (MCPTool * tool in _tools)
        [list addObject:@{ @"name" : tool.name,
                           @"description" : tool.toolDescription,
                           @"inputSchema" : tool.inputSchema }];
    return list;
}

- (MCPTool *)toolNamed:(NSString *)name {
    for (MCPTool * tool in _tools)
        if ([tool.name isEqualToString:name]) return tool;
    return nil;
}

@end
