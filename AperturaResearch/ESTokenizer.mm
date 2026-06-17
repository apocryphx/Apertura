#include "ESTokenizer.h"

#import <Foundation/Foundation.h>
#import "OCTTokenizer.h"

#include <stdexcept>

namespace es {

ESTokenizer::ESTokenizer(const std::string & tokenizerJsonPath) : tok_(nullptr) {
    @autoreleasepool {
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:tokenizerJsonPath.c_str()]];
        NSError * err = nil;
        OCTTokenizer * t = [OCTTokenizer tokenizerWithJSONFileURL:url error:&err];
        if (!t) {
            std::string msg = err ? err.localizedDescription.UTF8String : "unknown error";
            throw std::runtime_error("ESTokenizer: failed to load " + tokenizerJsonPath + ": " + msg);
        }
        tok_ = (void *) CFBridgingRetain(t);
    }
}

ESTokenizer::~ESTokenizer() {
    if (tok_) CFBridgingRelease(tok_);
}

std::vector<int> ESTokenizer::encode(const std::string & text, bool addSpecialTokens) const {
    @autoreleasepool {
        OCTTokenizer * t = (__bridge OCTTokenizer *) tok_;
        NSError * err = nil;
        NSArray<NSNumber *> * ids = [t encode:[NSString stringWithUTF8String:text.c_str()]
                             addSpecialTokens:addSpecialTokens
                                        error:&err];
        if (!ids) throw std::runtime_error("ESTokenizer: encode failed");
        std::vector<int> out;
        out.reserve(ids.count);
        for (NSNumber * n in ids) out.push_back((int) n.integerValue);
        return out;
    }
}

std::string ESTokenizer::decode(const std::vector<int> & ids, bool skipSpecialTokens) const {
    @autoreleasepool {
        OCTTokenizer * t = (__bridge OCTTokenizer *) tok_;
        NSMutableArray<NSNumber *> * arr = [NSMutableArray arrayWithCapacity:ids.size()];
        for (int id : ids) [arr addObject:@(id)];
        NSError * err = nil;
        NSString * s = [t decode:arr skipSpecialTokens:skipSpecialTokens error:&err];
        if (!s) throw std::runtime_error("ESTokenizer: decode failed");
        return std::string(s.UTF8String);
    }
}

}  // namespace es
