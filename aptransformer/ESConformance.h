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

    // ULP-relative comparison — the scale-free form of the above.
    //
    // Absolute tolerances are meaningless without knowing the magnitude: 1 bf16 ULP at 280 is 2.0,
    // at 0.5 it is 0.004. Gating on |delta| therefore fires on last-bit differences in large
    // activations while hiding real faults in small ones. This measures deviation in units of
    // ulp(ref) = 2^(floor(log2|ref|) - 7) and gates on the DISTRIBUTION rather than the max: a
    // lone element at 3 ULP is noise, 20% of elements past 4 ULP is a broken kernel.
    //
    // Intended for teacher-forced (per-op) comparisons, where each buffer is computed from the
    // reference's own inputs so accumulation is excluded and any real deviation is the op's.
    struct UlpStats {
        double pctExact;    // bit-identical
        double pctWithin1;  // <= 1 ULP
        double pctWithin2;  // <= 2 ULP
        double pctBeyond4;  // >  4 ULP
        double maxUlp;      // worst element, ignoring near-zero refs
        size_t n;
    };
    UlpStats ulpStats(const mx::array & got, const std::string & refName) const;

    bool compareUlp(const std::string & label,
                    const mx::array &   got,
                    const std::string & refName,
                    double minWithin1Pct = 75.0,
                    double maxBeyond4Pct = 10.0) const;

private:
    std::unordered_map<std::string, mx::array> fixtures_;
};

}  // namespace es
