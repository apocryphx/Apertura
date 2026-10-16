#include "ESVisionEncoderLayer.h"
#include "ESWeightLoader.h"

namespace es {

static std::string layerPrefix(int i) { return "vision_tower.encoder.layers." + std::to_string(i); }

ESVisionEncoderLayer::ESVisionEncoderLayer(const ESVisionConfig & cfg, int layerIdx, const ESWeightLoader & w)
    : inputLN_(w.get(layerPrefix(layerIdx) + ".input_layernorm.weight"), cfg.rmsNormEps),
      postAttnLN_(w.get(layerPrefix(layerIdx) + ".post_attention_layernorm.weight"), cfg.rmsNormEps),
      preFFLN_(w.get(layerPrefix(layerIdx) + ".pre_feedforward_layernorm.weight"), cfg.rmsNormEps),
      postFFLN_(w.get(layerPrefix(layerIdx) + ".post_feedforward_layernorm.weight"), cfg.rmsNormEps),
      attn_(cfg, layerPrefix(layerIdx) + ".self_attn", w),
      mlp_(layerPrefix(layerIdx) + ".mlp", w) {}

mx::array ESVisionEncoderLayer::forward(const mx::array & x, const mx::array & cos, const mx::array & sin) const {
    mx::array h = postAttnLN_.forward(attn_.forward(inputLN_.forward(x), cos, sin));
    mx::array x1 = mx::add(x, h);
    h = postFFLN_.forward(mlp_.forward(preFFLN_.forward(x1)));
    return mx::add(x1, h);
}

}  // namespace es
