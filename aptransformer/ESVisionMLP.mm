#include "ESVisionMLP.h"
#include "ESWeightLoader.h"
#include "ESOps.h"

namespace es {

ESVisionMLP::ESVisionMLP(const std::string & prefix, const ESWeightLoader & w)
    : gate_(esMakeClippableLinear(w, prefix + ".gate_proj")),
      up_(esMakeClippableLinear(w, prefix + ".up_proj")),
      down_(esMakeClippableLinear(w, prefix + ".down_proj")) {}

mx::array ESVisionMLP::forward(const mx::array & x) const {
    return down_.forward(mx::multiply(geluTanh(gate_.forward(x)), up_.forward(x)));
}

}  // namespace es
