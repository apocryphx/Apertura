//
//  MCPServer.h
//  apertura-mcp
//
//  The protocol layer: JSON-RPC 2.0 dispatch for the MCP subset a stdio tool server
//  needs — initialize, tools/list, tools/call, ping. Notifications are consumed
//  silently; unknown methods get -32601. Transport (framing, IO) lives in main.m.

#import <Foundation/Foundation.h>
#import "MCPToolRegistry.h"

NS_ASSUME_NONNULL_BEGIN

@interface MCPServer : NSObject

- (instancetype)initWithRegistry:(MCPToolRegistry *)registry;

/// Handle one parsed JSON-RPC message. Returns the response object to serialize, or
/// nil when no response is due (notifications).
- (nullable NSDictionary *)responseForMessage:(NSDictionary *)message;

@end

NS_ASSUME_NONNULL_END
