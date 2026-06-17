#pragma once
//  ESConformance — load PyTorch fixtures (safetensors) and compare Apertura outputs.
//
//  Fixtures are bf16/int32 (the canonical compute dtype). Comparisons upcast both sides to
//  f32 and gate on p99 of relative + absolute deviation (robust to single FP-sensitive tails);
//  the true completion gate is argmax / token-id match, checked separately.
#include "mlx/mlx.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace es {
namespace mx = mlx::core;

struct ESDevStats { float max, median, p99, mean; };

class ESConformance {
public:
    explicit ESConformance(const std::string & fixturesSafetensorsPath);

    bool has(const std::string & name) const { return fixtures_.count(name) > 0; }
    const mx::array & get(const std::string & name) const;

    // int32 fixture -> vector<int> (flattened).
    std::vector<int> ints(const std::string & name) const;

    static ESDevStats stats(const mx::array & a);  // a is f32

    // Compare got vs fixture[refName]; prints a line; returns pass.
    bool compare(const std::string & label,
                 const mx::array &   got,
                 const std::string & refName,
                 float relP99Max = 5e-2f,
                 float absP99Max = 5e-2f) const;

private:
    std::unordered_map<std::string, mx::array> fixtures_;
};

}  // namespace es
