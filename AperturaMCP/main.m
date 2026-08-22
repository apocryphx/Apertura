//
//  main.m
//  apertura-mcp — Apertura as an MCP stdio tool server.
//
//  Transport: newline-delimited JSON-RPC 2.0 on stdin/stdout. stdout carries ONLY
//  protocol frames — every log goes to stderr (stdout purity is protocol-critical;
//  a stray printf breaks the client's parser). One synchronous loop: read a line,
//  dispatch, write a line, flush. Long tool calls (model load, generation) block the
//  loop by design — single-client stdio. EOF on stdin is a clean shutdown.
//
//  Register with Claude Code:  claude mcp add apertura -- <path>/apertura-mcp

#import <Foundation/Foundation.h>
#import "MCPEngineHost.h"
#import "MCPServer.h"

static void apWriteFrame(NSDictionary * frame) {
    NSData * data = [NSJSONSerialization dataWithJSONObject:frame options:0 error:nil];
    if (!data) return;
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        MCPToolRegistry * registry = [[MCPToolRegistry alloc] init];
        MCPEngineHost * host = [[MCPEngineHost alloc] init];
        [host registerToolsInto:registry];
        MCPServer * server = [[MCPServer alloc] initWithRegistry:registry];
        fprintf(stderr, "[apertura-mcp] ready (stdio)\n");

        NSMutableData * buffer = [NSMutableData data];
        char chunk[65536];
        ssize_t got;
        // read(2), not fread: fread loops until it fills the whole buffer or hits EOF,
        // which stalls forever on an interactive pipe that stays open between requests.
        while ((got = read(STDIN_FILENO, chunk, sizeof chunk)) > 0) {
            [buffer appendBytes:chunk length:got];
            // Split complete lines off the front; keep the remainder buffered.
            for (;;) {
                NSRange newline = [buffer rangeOfData:[NSData dataWithBytes:"\n" length:1]
                                              options:0
                                                range:NSMakeRange(0, buffer.length)];
                if (newline.location == NSNotFound) break;
                NSData * line = [buffer subdataWithRange:NSMakeRange(0, newline.location)];
                [buffer replaceBytesInRange:NSMakeRange(0, newline.location + 1)
                                  withBytes:NULL length:0];
                if (line.length == 0) continue;

                NSError * parseError = nil;
                id message = [NSJSONSerialization JSONObjectWithData:line options:0
                                                               error:&parseError];
                if (![message isKindOfClass:NSDictionary.class]) {
                    apWriteFrame(@{ @"jsonrpc" : @"2.0", @"id" : NSNull.null,
                                    @"error" : @{ @"code" : @(-32700),
                                                  @"message" : @"parse error" } });
                    continue;
                }
                @autoreleasepool {
                    NSDictionary * response = [server responseForMessage:message];
                    if (response) apWriteFrame(response);
                }
            }
        }
        fprintf(stderr, "[apertura-mcp] stdin closed — shutting down\n");
        [host shutdown];
    }
    return 0;
}
