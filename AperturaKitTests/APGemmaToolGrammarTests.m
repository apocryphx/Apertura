//  APGemmaToolGrammarTests — the compact tool grammar codec, model-free.
//  Expected strings are the reference template's rendering (the CLI --tools file
//  format); --tools-verify gates the same codec end to end against a live model.
#import <XCTest/XCTest.h>
#import <AperturaKit/AperturaKit.h>
#import "APGemmaToolGrammar.h"

@interface APGrammarStubTool : NSObject <APTool>
@property (copy) NSString * name;
@property (copy) NSString * toolDescription;
@property (copy) NSDictionary<NSString *, id> * parameterSchema;
@end
@implementation APGrammarStubTool
- (void)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments
                 completion:(void (^)(APContent *, NSError *))completion {
    completion([APContent textContent:@""], nil);
}
@end

@interface APGemmaToolGrammarTests : XCTestCase
@end

@implementation APGemmaToolGrammarTests

- (void)testSanitizeBreaksMarkerText {
    XCTAssertEqualObjects([APGemmaToolGrammar sanitize:@"a<|b|>c"], @"a< |b| >c");
    XCTAssertEqualObjects([APGemmaToolGrammar sanitize:@"plain"], @"plain");
    XCTAssertEqualObjects([APGemmaToolGrammar sanitize:@"<|\"|>"], @"< |\"| >");
}

- (void)testCompactValueScalars {
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:@"hi"], @"<|\"|>hi<|\"|>");
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:@"x<|y"], @"<|\"|>x< |y<|\"|>");
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:@YES], @"true");
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:@NO], @"false");
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:@42], @"42");
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:@1.5], @"1.5");
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:nil], @"null");
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:NSNull.null], @"null");
}

- (void)testCompactValueContainersSortKeys {
    id v = @{ @"zeta" : @[ @1, @"two", @{ @"b" : @NO, @"a" : @[] } ], @"alpha" : @"A" };
    XCTAssertEqualObjects([APGemmaToolGrammar compactValue:v],
        @"{alpha:<|\"|>A<|\"|>,zeta:[1,<|\"|>two<|\"|>,{a:[],b:false}]}");
}

- (void)testUppercaseSchemaTypesOnlyTouchesTypeValues {
    id schema = @{ @"type" : @"object",
                   @"properties" : @{ @"q" : @{ @"type" : @"string", @"description" : @"lower stays" },
                                      @"n" : @{ @"type" : @[ @"integer", @"null" ], @"enum" : @[ @"a" ] } },
                   @"required" : @[ @"q" ] };
    id up = [APGemmaToolGrammar uppercaseSchemaTypes:schema];
    XCTAssertEqualObjects(up[@"type"], @"OBJECT");
    XCTAssertEqualObjects(up[@"properties"][@"q"][@"type"], @"STRING");
    XCTAssertEqualObjects(up[@"properties"][@"q"][@"description"], @"lower stays");
    XCTAssertEqualObjects(up[@"properties"][@"n"][@"type"], (@[ @"INTEGER", @"NULL" ]));
    XCTAssertEqualObjects(up[@"properties"][@"n"][@"enum"], @[ @"a" ]);
    XCTAssertEqualObjects(up[@"required"], @[ @"q" ]);
    XCTAssertEqualObjects(schema[@"type"], @"object");   // input untouched
}

- (void)testDeclarationForTool {
    APGrammarStubTool * t = [[APGrammarStubTool alloc] init];
    t.name = @"lookup_codeword";
    t.toolDescription = @"Returns the codeword.";
    t.parameterSchema = @{ @"type" : @"object",
                           @"properties" : @{ @"key" : @{ @"type" : @"string" } } };
    XCTAssertEqualObjects([APGemmaToolGrammar declarationForTool:t],
        @"declaration:lookup_codeword{description:<|\"|>Returns the codeword.<|\"|>,"
         "parameters:{properties:{key:{type:<|\"|>STRING<|\"|>}},type:<|\"|>OBJECT<|\"|>}}");
    t.parameterSchema = nil;   // nil schema renders as an empty object
    XCTAssertEqualObjects([APGemmaToolGrammar declarationForTool:t],
        @"declaration:lookup_codeword{description:<|\"|>Returns the codeword.<|\"|>,parameters:{}}");
}

- (void)testJSONEscape {
    XCTAssertEqualObjects([APGemmaToolGrammar JSONEscape:@"a\"b\\c\nd\re\tf\x01g"],
                          @"a\\\"b\\\\c\\nd\\re\\tf\\u0001g");
    XCTAssertEqualObjects([APGemmaToolGrammar JSONEscape:@"ümläut 🜂"], @"ümläut 🜂");
}

- (void)testParseArgumentsBareKeysAndQuotedSpans {
    NSDictionary * d = [APGemmaToolGrammar parseArguments:
        @"query:<|\"|>line one\nline \"two\" {braces}<|\"|>,limit:3,flag:true"];
    XCTAssertEqualObjects(d[@"query"], @"line one\nline \"two\" {braces}");
    XCTAssertEqualObjects(d[@"limit"], @3);
    XCTAssertEqualObjects(d[@"flag"], @YES);
}

- (void)testParseArgumentsAcceptsBracedBodyAndNesting {
    NSDictionary * d = [APGemmaToolGrammar parseArguments:
        @"{ outer:{inner:<|\"|>v<|\"|>, list:[1,2]} }"];
    XCTAssertEqualObjects(d[@"outer"][@"inner"], @"v");
    XCTAssertEqualObjects(d[@"outer"][@"list"], (@[ @1, @2 ]));
}

- (void)testParseArgumentsEmptyAndFallback {
    XCTAssertEqualObjects([APGemmaToolGrammar parseArguments:@""], @{});
    XCTAssertEqualObjects([APGemmaToolGrammar parseArguments:nil], @{});
    NSDictionary * bad = [APGemmaToolGrammar parseArguments:@"not : : json ,,"];
    XCTAssertEqualObjects(bad[@"_raw"], @"not : : json ,,");
}

- (void)testParseArgumentsRoundTripsCompactValue {
    NSDictionary * args = @{ @"text" : @"with <|marker|> and\nnewline", @"n" : @7, @"ok" : @NO };
    NSString * body = [APGemmaToolGrammar compactValue:args];
    NSDictionary * back = [APGemmaToolGrammar parseArguments:body];
    XCTAssertEqualObjects(back[@"text"], @"with < |marker| > and\nnewline");   // sanitized on the way in
    XCTAssertEqualObjects(back[@"n"], @7);
    XCTAssertEqualObjects(back[@"ok"], @NO);
}

@end
