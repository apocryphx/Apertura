//  APCodeHighlighterTests — the grammars and the readable JSON layout (ported from the
//  prototype's PCParsingTests).
#import "APChatModelTestSupport.h"

@interface APCodeHighlighterTests : XCTestCase
@end

@implementation APCodeHighlighterTests

- (void)testGrammars {
    NSArray * swift = APTokens([APCodeHighlighter highlightCode:@"let s = \"a // b\" // note\n@MainActor struct V {}" language:@"swift"]);
    XCTAssertEqualObjects(swift, (@[ APToken(@"let", APCodeTokenKindKeyword), APToken(@"\"a // b\"", APCodeTokenKindString),
                                     APToken(@"// note", APCodeTokenKindComment), APToken(@"@MainActor", APCodeTokenKindAttribute),
                                     APToken(@"struct", APCodeTokenKindKeyword), APToken(@"V", APCodeTokenKindType) ]));

    NSArray * objc = APTokens([APCodeHighlighter highlightCode:@"#import <Foundation/Foundation.h>\nNSLog(@\"n=%d\", 42);" language:@"objc"]);
    XCTAssertEqualObjects(objc, (@[ APToken(@"#import <Foundation/Foundation.h>", APCodeTokenKindAttribute), APToken(@"NSLog", APCodeTokenKindType),
                                    APToken(@"@\"n=%d\"", APCodeTokenKindString), APToken(@"42", APCodeTokenKindNumber) ]));

    NSArray * python = APTokens([APCodeHighlighter highlightCode:@"@dataclass\ndef f(x): return r'\\d' # c" language:@"py"]);
    XCTAssertEqualObjects(python, (@[ APToken(@"@dataclass", APCodeTokenKindAttribute), APToken(@"def", APCodeTokenKindKeyword),
                                      APToken(@"return", APCodeTokenKindKeyword), APToken(@"r'\\d'", APCodeTokenKindString),
                                      APToken(@"# c", APCodeTokenKindComment) ]));

    NSArray * json = APTokens([APCodeHighlighter highlightCode:@"{\"a\": -1.5, \"b\": \"s\", \"c\": null}" language:@"JSON"]);
    XCTAssertEqualObjects(json, (@[ APToken(@"\"a\"", APCodeTokenKindAttribute), APToken(@"-1.5", APCodeTokenKindNumber),
                                    APToken(@"\"b\"", APCodeTokenKindAttribute), APToken(@"\"s\"", APCodeTokenKindString),
                                    APToken(@"\"c\"", APCodeTokenKindAttribute), APToken(@"null", APCodeTokenKindKeyword) ]));

    NSArray * shell = APTokens([APCodeHighlighter highlightCode:@"export APP=\"$HOME\" # x\nxcodebuild --quiet ${DIR}" language:@"bash"]);
    XCTAssertEqualObjects(shell, (@[ APToken(@"export", APCodeTokenKindKeyword), APToken(@"\"$HOME\"", APCodeTokenKindString),
                                     APToken(@"# x", APCodeTokenKindComment), APToken(@"--quiet", APCodeTokenKindType),
                                     APToken(@"${DIR}", APCodeTokenKindAttribute) ]));

    XCTAssertEqual(APTokens([APCodeHighlighter highlightCode:@"let x = 1" language:@"cobol"]).count, 0u);
    XCTAssertEqual(APTokens([APCodeHighlighter highlightCode:@"let x = 1" language:nil]).count, 0u);
}

- (void)testMoreGrammars {
    NSArray * ts = APTokens([APCodeHighlighter highlightCode:@"const s = `a // b ${x}`; // n\ninterface P { id: number }" language:@"ts"]);
    XCTAssertEqualObjects(ts, (@[ APToken(@"const", APCodeTokenKindKeyword), APToken(@"`a // b ${x}`", APCodeTokenKindString),
                                  APToken(@"// n", APCodeTokenKindComment), APToken(@"interface", APCodeTokenKindKeyword),
                                  APToken(@"P", APCodeTokenKindType), APToken(@"number", APCodeTokenKindType) ]));

    NSArray * rust = APTokens([APCodeHighlighter highlightCode:@"#[derive(Debug)]\nfn f<'a>(s: &'a str) -> u32 { println!(\"{}\", 'x') }" language:@"rs"]);
    XCTAssertEqualObjects(rust, (@[ APToken(@"#[derive(Debug)]", APCodeTokenKindAttribute), APToken(@"fn", APCodeTokenKindKeyword),
                                    APToken(@"'a", APCodeTokenKindAttribute), APToken(@"'a", APCodeTokenKindAttribute),
                                    APToken(@"str", APCodeTokenKindType), APToken(@"u32", APCodeTokenKindType),
                                    APToken(@"println!", APCodeTokenKindAttribute), APToken(@"\"{}\"", APCodeTokenKindString),
                                    APToken(@"'x'", APCodeTokenKindString) ]));

    NSArray * go = APTokens([APCodeHighlighter highlightCode:@"x := `a // b` // n\nfunc main() { var n int = 42 }" language:@"go"]);
    XCTAssertEqualObjects(go, (@[ APToken(@"`a // b`", APCodeTokenKindString), APToken(@"// n", APCodeTokenKindComment),
                                  APToken(@"func", APCodeTokenKindKeyword), APToken(@"var", APCodeTokenKindKeyword),
                                  APToken(@"int", APCodeTokenKindType), APToken(@"42", APCodeTokenKindNumber) ]));

    NSArray * html = APTokens([APCodeHighlighter highlightCode:@"<!-- c -->\n<a href=\"x\" class='y'>don't</a>" language:@"html"]);
    XCTAssertEqualObjects(html, (@[ APToken(@"<!-- c -->", APCodeTokenKindComment), APToken(@"a", APCodeTokenKindKeyword),
                                    APToken(@"href", APCodeTokenKindAttribute), APToken(@"\"x\"", APCodeTokenKindString),
                                    APToken(@"class", APCodeTokenKindAttribute), APToken(@"'y'", APCodeTokenKindString),
                                    APToken(@"a", APCodeTokenKindKeyword) ]));

    NSArray * css = APTokens([APCodeHighlighter highlightCode:@"@media print { .a { color: #fff; margin: -2px; background: url(http://x/a.png) !important; } }" language:@"css"]);
    XCTAssertEqualObjects(css, (@[ APToken(@"@media", APCodeTokenKindKeyword), APToken(@".a", APCodeTokenKindType),
                                   APToken(@"color", APCodeTokenKindAttribute), APToken(@"#fff", APCodeTokenKindNumber),
                                   APToken(@"margin", APCodeTokenKindAttribute), APToken(@"-2px", APCodeTokenKindNumber),
                                   APToken(@"background", APCodeTokenKindAttribute), APToken(@"url(http://x/a.png)", APCodeTokenKindString),
                                   APToken(@"!important", APCodeTokenKindKeyword) ]));

    NSArray * yaml = APTokens([APCodeHighlighter highlightCode:@"name: \"a: b\" # c\nport: 80\nok: true" language:@"yml"]);
    XCTAssertEqualObjects(yaml, (@[ APToken(@"name", APCodeTokenKindAttribute), APToken(@"\"a: b\"", APCodeTokenKindString),
                                    APToken(@"# c", APCodeTokenKindComment), APToken(@"port", APCodeTokenKindAttribute),
                                    APToken(@"80", APCodeTokenKindNumber), APToken(@"ok", APCodeTokenKindAttribute),
                                    APToken(@"true", APCodeTokenKindKeyword) ]));

    NSArray * sql = APTokens([APCodeHighlighter highlightCode:@"-- c\nSELECT id, 'it''s' FROM t WHERE n >= 10;" language:@"sql"]);
    XCTAssertEqualObjects(sql, (@[ APToken(@"-- c", APCodeTokenKindComment), APToken(@"SELECT", APCodeTokenKindKeyword),
                                   APToken(@"'it''s'", APCodeTokenKindString), APToken(@"FROM", APCodeTokenKindKeyword),
                                   APToken(@"WHERE", APCodeTokenKindKeyword), APToken(@"10", APCodeTokenKindNumber) ]));

    NSArray * java = APTokens([APCodeHighlighter highlightCode:@"@Override public long f() { return 10L; }" language:@"java"]);
    XCTAssertEqualObjects(java, (@[ APToken(@"@Override", APCodeTokenKindAttribute), APToken(@"public", APCodeTokenKindKeyword),
                                    APToken(@"long", APCodeTokenKindType), APToken(@"return", APCodeTokenKindKeyword),
                                    APToken(@"10L", APCodeTokenKindNumber) ]));

    NSArray * kotlin = APTokens([APCodeHighlighter highlightCode:@"fun f(): String = \"\"\"a // b\"\"\" // n" language:@"kt"]);
    XCTAssertEqualObjects(kotlin, (@[ APToken(@"fun", APCodeTokenKindKeyword), APToken(@"String", APCodeTokenKindType),
                                      APToken(@"\"\"\"a // b\"\"\"", APCodeTokenKindString), APToken(@"// n", APCodeTokenKindComment) ]));
}

- (void)testSnippet {
    APCodeSnippet * snippet = [APCodeSnippet snippetWithLanguage:@"swift" code:@"let x = 1"];
    XCTAssertEqualObjects(snippet.language, @"swift");
    XCTAssertEqualObjects(snippet.code, @"let x = 1");
    XCTAssertEqualObjects(snippet.highlighted.string, @"let x = 1");
    XCTAssertEqualObjects(APTokens(snippet.highlighted), (@[ APToken(@"let", APCodeTokenKindKeyword), APToken(@"1", APCodeTokenKindNumber) ]));
}

- (void)testReadableJSON {
    NSAttributedString * readable = [APReadableJSON formatText:@"{\"n\": 1, \"man\": \"a\\nb\", \"ok\": true}"];
    XCTAssertEqualObjects(readable.string, @"{\n  \"man\":\n    a\n    b\n  \"n\": 1,\n  \"ok\": true\n}");
    XCTAssertTrue([APTokens(readable) containsObject:APToken(@"true", APCodeTokenKindKeyword)]);
    NSRange textBlock = [readable.string rangeOfString:@"\n    a\n    b"];
    NSRange untagged;
    XCTAssertNil([readable attribute:APCodeTokenAttributeName atIndex:textBlock.location
                longestEffectiveRange:&untagged inRange:textBlock]);
    XCTAssertTrue(NSEqualRanges(untagged, textBlock), @"Text blocks stay untagged");

    XCTAssertNil([APReadableJSON formatText:@"{\"a\": \"single line\"}"]);
    XCTAssertNil([APReadableJSON formatText:@"not json"]);

    NSAttributedString * value = [APReadableJSON formatValue:@{ @"b" : @[ @1, @2.5, @NO ], @"a" : [NSNull null], @"e" : @{} }];
    XCTAssertEqualObjects(value.string, @"{\n  \"a\": null,\n  \"b\": [\n    1,\n    2.5,\n    false\n  ],\n  \"e\": {}\n}");
    XCTAssertEqualObjects([APReadableJSON formatValue:@{}].string, @"{}");
}

@end
