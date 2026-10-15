#import "APError.h"

NSErrorDomain const APErrorDomain = @"com.apertura.AperturaKit";
NSErrorUserInfoKey const APErrorToolNameKey = @"APErrorToolName";

NSError * APMakeError(APErrorCode code, NSString * message) {
    return [NSError errorWithDomain:APErrorDomain code:code
                           userInfo:@{ NSLocalizedDescriptionKey : message }];
}
