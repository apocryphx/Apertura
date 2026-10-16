//
//  ESModelConfigTests.mm
//  aptransformerTests
//
//  ESModelConfig reads per-checkpoint facts from the files beside config.json. Thinking-off
//  prompt shape: the chat template is the oracle (12B/26B/31B pre-close an empty thought
//  channel, E2B/E4B do not); without a template the family decides (PLE => no pre-close).
//

#import <XCTest/XCTest.h>
#include "ESModelConfig.h"
#include <fstream>

@interface ESModelConfigTests : XCTestCase
@end

@implementation ESModelConfigTests {
    NSString * _dir;
}

- (void)setUp {
    _dir = [NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"ESModelConfigTests-%@", NSUUID.UUID.UUIDString]];
    [NSFileManager.defaultManager createDirectoryAtPath:_dir withIntermediateDirectories:YES
                                             attributes:nil error:nil];
}

- (void)tearDown {
    [NSFileManager.defaultManager removeItemAtPath:_dir error:nil];
}

- (void)writeConfigWithPLE:(BOOL)ple {
    NSString * json = [NSString stringWithFormat:
        @"{\"text_config\":{\"hidden_size\":64,\"num_hidden_layers\":2,\"layer_types\":"
        @"[\"sliding_attention\",\"full_attention\"],\"hidden_size_per_layer_input\":%d}}", ple ? 256 : 0];
    [json writeToFile:[_dir stringByAppendingPathComponent:@"config.json"] atomically:YES
             encoding:NSUTF8StringEncoding error:nil];
}

- (void)writeTemplate:(NSString *)jinja {
    [jinja writeToFile:[_dir stringByAppendingPathComponent:@"chat_template.jinja"] atomically:YES
              encoding:NSUTF8StringEncoding error:nil];
}

- (es::ESModelConfig)load {
    return es::ESModelConfig::fromConfigJSON(
        std::string([_dir stringByAppendingPathComponent:@"config.json"].UTF8String));
}

- (void)testTemplateWithPreclosePrecloses {
    [self writeConfigWithPLE:YES];   // the template wins over the family
    [self writeTemplate:@"{%- if not enable_thinking -%}\n{{- '<|channel>thought\\n<channel|>' -}}\n{%- endif -%}"];
    XCTAssertTrue([self load].preclosesEmptyThought);
}

- (void)testTemplateWithoutPrecloseDoesNot {
    [self writeConfigWithPLE:NO];
    [self writeTemplate:@"{{- '<|turn>model\\n' -}}"];
    XCTAssertFalse([self load].preclosesEmptyThought);
}

- (void)testNoTemplateFallsBackToFamily {
    [self writeConfigWithPLE:YES];
    XCTAssertFalse([self load].preclosesEmptyThought);   // E2B/E4B: no pre-close
    [self writeConfigWithPLE:NO];
    XCTAssertTrue([self load].preclosesEmptyThought);    // 12B/26B/31B: pre-close
}

@end
