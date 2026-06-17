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

}  // namespace es
