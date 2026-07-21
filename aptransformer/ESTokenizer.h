#pragma once
//  ESTokenizer — thin ObjC++ wrapper over the battle-tested ObjCTokenizer (OCTTokenizer).
//  Loads a HF tokenizer.json; exposes encode/decode over std::vector<int>.
//
//  Lives at the driver level for now (the aptransformer framework stays pure MLX compute);
//  it can be promoted into the framework once ObjCTokenizer is linked as a formal dependency.
#include <string>
#include <vector>

namespace es {

class ESTokenizer {
public:
    // tokenizerJsonPath: a HF tokenizer.json (e.g. from the model snapshot).
    explicit ESTokenizer(const std::string & tokenizerJsonPath);
    ~ESTokenizer();

    std::vector<int> encode(const std::string & text, bool addSpecialTokens = false) const;
    std::string      decode(const std::vector<int> & ids, bool skipSpecialTokens = true) const;

private:
    void * tok_;  // OCTTokenizer* (opaque to avoid leaking ObjC into C++ consumers)
};

}  // namespace es
