#include "ESConformance.h"

#include <cstdio>
#include <stdexcept>

namespace es {

ESConformance::ESConformance(const std::string & path) {
    auto loaded = mx::load_safetensors(path);  // {map, metadata}
    fixtures_ = std::move(loaded.first);
    if (fixtures_.empty()) {
        throw std::runtime_error("ESConformance: no tensors in " + path);
    }
}

const mx::array & ESConformance::get(const std::string & name) const {
    auto it = fixtures_.find(name);
    if (it == fixtures_.end()) {
        throw std::runtime_error("ESConformance: no fixture named '" + name + "'");
    }
    return it->second;
}

std::vector<int> ESConformance::ints(const std::string & name) const {
    mx::array a = mx::astype(get(name), mx::int32);
    mx::eval(a);
    const int * p = a.data<int>();
    return std::vector<int>(p, p + a.size());
}

ESDevStats ESConformance::stats(const mx::array & a) {
    mx::array flat   = mx::reshape(a, {(int) a.size()});
    mx::array sorted = mx::sort(flat, 0);
    mx::array meanv  = mx::mean(a);
    mx::eval(sorted, meanv);
    const float * p = sorted.data<float>();
    int n = (int) sorted.size();
    return {p[n - 1], p[n / 2], p[(int) (0.99 * (n - 1))], meanv.item<float>()};
}

bool ESConformance::compare(const std::string & label,
                            const mx::array &   got,
                            const std::string & refName,
                            float relP99Max,
                            float absP99Max) const {
    mx::array ref = mx::astype(get(refName), mx::float32);
    mx::array g   = mx::astype(got, mx::float32);

    if (g.shape() != ref.shape()) {
        // Allow flattened-equivalent shapes (same element count) for convenience.
        if (g.size() == ref.size()) {
            g   = mx::reshape(g, {(int) g.size()});
            ref = mx::reshape(ref, {(int) ref.size()});
        } else {
            std::printf("[%-26s] SHAPE MISMATCH  got.size=%zu ref.size=%zu (ref '%s')\n",
                        label.c_str(), (size_t) g.size(), (size_t) ref.size(), refName.c_str());
            return false;
        }
    }

    mx::array d   = mx::abs(mx::subtract(g, ref));
    mx::array rel = mx::divide(d, mx::add(mx::abs(ref), mx::array(1e-6f)));
    ESDevStats a = stats(d);
    ESDevStats r = stats(rel);
    bool pass = (r.p99 <= relP99Max) && (a.p99 <= absP99Max);
    std::printf("[%-26s] %s  abs(max=%.2e med=%.2e p99=%.2e)  rel(max=%.2e med=%.2e p99=%.2e)\n",
                label.c_str(), pass ? "PASS" : "FAIL",
                a.max, a.median, a.p99, r.max, r.median, r.p99);
    return pass;
}

ESConformance::UlpStats ESConformance::ulpStats(const mx::array & got,
                                                const std::string & refName) const {
    mx::array ref = mx::astype(get(refName), mx::float32);
    mx::array g   = mx::astype(got, mx::float32);
    if (g.shape() != ref.shape() && g.size() == ref.size()) {
        g   = mx::reshape(g, {(int) g.size()});
        ref = mx::reshape(ref, {(int) ref.size()});
    }
    if (g.shape() != ref.shape()) return {0, 0, 0, 100, 0, 0};

    mx::array d = mx::abs(mx::subtract(g, ref));
    mx::array a = mx::abs(ref);

    // ulp(x) for bf16 (7 explicit mantissa bits) = 2^(floor(log2|x|) - 7).
    // Elements whose reference is ~0 have no meaningful ULP: |delta|/ulp explodes and swamps the
    // statistics. They are excluded from the ratio and counted only via pctExact.
    const float kTiny = 1e-30f;
    mx::array valid = mx::greater(a, mx::array(kTiny));
    mx::array safeA = mx::maximum(a, mx::array(kTiny));
    mx::array ulp   = mx::power(mx::array(2.0f), mx::subtract(mx::floor(mx::log2(safeA)), mx::array(7.0f)));
    mx::array ud    = mx::divide(d, ulp);

    mx::array nValid = mx::sum(mx::astype(valid, mx::float32));
    auto pctOf = [&](const mx::array & mask) {
        mx::array c = mx::sum(mx::astype(mx::logical_and(mask, valid), mx::float32));
        mx::eval(c, nValid);
        double den = (double) nValid.item<float>();
        return den > 0 ? 100.0 * (double) c.item<float>() / den : 0.0;
    };

    UlpStats s{};
    s.n          = (size_t) got.size();
    s.pctExact   = pctOf(mx::equal(d, mx::array(0.0f)));
    s.pctWithin1 = pctOf(mx::less_equal(ud, mx::array(1.0f)));
    s.pctWithin2 = pctOf(mx::less_equal(ud, mx::array(2.0f)));
    s.pctBeyond4 = pctOf(mx::greater(ud, mx::array(4.0f)));
    mx::array mx_ulp = mx::max(mx::where(valid, ud, mx::array(0.0f)));
    mx::eval(mx_ulp);
    s.maxUlp = (double) mx_ulp.item<float>();
    return s;
}

bool ESConformance::compareUlp(const std::string & label,
                               const mx::array &   got,
                               const std::string & refName,
                               double minWithin1Pct,
                               double maxBeyond4Pct) const {
    UlpStats s = ulpStats(got, refName);
    bool pass = (s.pctWithin1 >= minWithin1Pct) && (s.pctBeyond4 <= maxBeyond4Pct);
    std::printf("[%-26s] %s  exact %5.1f%%  <=1ulp %5.1f%%  <=2ulp %5.1f%%  >4ulp %5.2f%%  maxUlp %7.1f\n",
                label.c_str(), pass ? "PASS" : "FAIL",
                s.pctExact, s.pctWithin1, s.pctWithin2, s.pctBeyond4, s.maxUlp);
    return pass;
}

}  // namespace es
