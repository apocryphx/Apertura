//  APMarkdownInternal — PROJECT header (never public): the parser entry point and the
//  document initializer the streaming parser (APMarkdownStream) builds snapshots with.
#pragma once
#import "APMarkdownDocument.h"

NS_ASSUME_NONNULL_BEGIN

@interface APMarkdownDocument (Internal)
/// The blocks of `source`, parsed as a complete document.
+ (NSArray<APMarkdownBlock *> *)blocksByParsing:(NSString *)source;
/// A document whose blocks were produced elsewhere (incrementally); parses nothing.
- (instancetype)initWithSource:(NSString *)source blocks:(NSArray<APMarkdownBlock *> *)blocks;
@end

NS_ASSUME_NONNULL_END
