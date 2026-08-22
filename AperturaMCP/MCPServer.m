//
//  MCPServer.m
//  apertura-mcp
//

#import "MCPServer.h"

static NSString * const kProtocolVersion = @"2025-06-18";
static NSString * const kServerVersion   = @"0.1.0";

@implementation MCPServer {
    MCPToolRegistry * _registry;
}

- (instancetype)initWithRegistry:(MCPToolRegistry *)registry {
    if ((self = [super init])) _registry = registry;
    return self;
}

static NSDictionary * apResult(id msgID, NSDictionary * result) {
    return @{ @"jsonrpc" : @"2.0", @"id" : msgID, @"result" : result };
}

static NSDictionary * apError(id msgID, NSInteger code, NSString * message) {
    return @{ @"jsonrpc" : @"2.0", @"id" : msgID ?: NSNull.null,
              @"error" : @{ @"code" : @(code), @"message" : message } };
}

- (NSDictionary *)responseForMessage:(NSDictionary *)message {
    NSString * method = [message[@"method"] isKindOfClass:NSString.class] ? message[@"method"] : nil;
    id msgID = message[@"id"];
    BOOL isNotification = (msgID == nil || msgID == NSNull.null);

    if (!method) {
        return isNotification ? nil : apError(msgID, -32600, @"not a request");
    }
    if ([method hasPrefix:@"notifications/"]) return nil;   // consumed silently

    if ([method isEqualToString:@"initialize"]) {
        // Echo a protocol version we know; ours otherwise.
        NSString * requested = message[@"params"][@"protocolVersion"];
        NSString * version = [requested isKindOfClass:NSString.class] ? requested : kProtocolVersion;
        return apResult(msgID, @{
            @"protocolVersion" : version,
            @"capabilities" : @{ @"tools" : @{} },
            @"serverInfo" : @{ @"name" : @"apertura-mcp", @"version" : kServerVersion },
        });
    }
    if ([method isEqualToString:@"ping"]) {
        return apResult(msgID, @{});
    }
    if ([method isEqualToString:@"tools/list"]) {
        return apResult(msgID, @{ @"tools" : [_registry toolListPayload] });
    }
    if ([method isEqualToString:@"tools/call"]) {
        NSDictionary * params = [message[@"params"] isKindOfClass:NSDictionary.class]
            ? message[@"params"] : @{};
        NSString * name = [params[@"name"] isKindOfClass:NSString.class] ? params[@"name"] : nil;
        MCPTool * tool = name ? [_registry toolNamed:name] : nil;
        if (!tool) return apError(msgID, -32602, [NSString stringWithFormat:
                                                  @"unknown tool: %@", name ?: @"(none)"]);
        NSDictionary * arguments = [params[@"arguments"] isKindOfClass:NSDictionary.class]
            ? params[@"arguments"] : @{};
        NSError * error = nil;
        NSString * text = tool.handler(arguments, &error);
        // Tool failures are isError CONTENT — the protocol call itself succeeded.
        BOOL failed = (text == nil);
        return apResult(msgID, @{
            @"content" : @[ @{ @"type" : @"text",
                               @"text" : failed ? (error.localizedDescription ?: @"tool failed")
                                                : text } ],
            @"isError" : @(failed),
        });
    }
    return isNotification ? nil : apError(msgID, -32601, [NSString stringWithFormat:
                                                          @"method not found: %@", method]);
}

@end
