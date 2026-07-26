//  APGoogleSession — the remote backend: Gemma on Google's Gemini API.
//
//  Same conversation contract as every APSession backend, spoken over streaming REST
//  (streamGenerateContent, Server-Sent Events):
//    prime            → the messages become the request's systemInstruction (+ seed
//                       turns); instant, resent with every request. cacheURL is IGNORED
//                       (no local KV; Google's implicit prefix caching plays that role).
//    reasoningEnabled → thinkingConfig thinkingLevel "high" + includeThoughts; the
//                       API's thought-flagged parts stream as isThought deltas (the
//                       server returns thought SUMMARIES — terser than local reasoning).
//    tools            → native function calling: registered APTool schemas are sent as
//                       function declarations; functionCall parts dispatch through the
//                       SAME delegate veto + invoke path as the local backend, and the
//                       functionResponse continues the turn — all within one respond.
//
//  PRIVACY: every request ships the full persona and conversation to Google's servers.
//  Give the user a loud, visible indication when this backend is active.
#import <AperturaKit/APSession.h>

NS_ASSUME_NONNULL_BEGIN

@interface APGoogleSession : APSession

/// `modelName` is a Gemini-API model id, e.g. @"gemma-4-31b-it" or @"gemma-4-26b-a4b-it".
/// `apiKeyProvider` is called once per HTTP request, off the main thread — back it with
/// the Keychain, never a hard-coded literal. Returning nil fails the turn with
/// APErrorMissingAPIKey.
- (instancetype)initWithModelName:(NSString *)modelName
                   apiKeyProvider:(NSString *_Nullable (^)(void))apiKeyProvider;

@property (readonly, copy) NSString *modelName;

@end

NS_ASSUME_NONNULL_END
