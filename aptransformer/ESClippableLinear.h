#pragma once
//  ESClippableLinear — Gemma4ClippableLinear: the QAT-clipped projection of the vision tower.
//
//    y = clamp( clamp(x, input_min, input_max) @ W^T , output_min, output_max )
//
//  The four bounds are bf16 scalar buffers stored with the checkpoint (`<name>.input_min`, ...);
//  a checkpoint without them (use_clipped_linears false) degenerates to a plain ESLinear. The
//  weight itself lives under `<name>.linear.weight`. The clamps are not an optimization — they
//  reproduce the activation range the model was trained to see, and the bounds are finite.
#include "mlx/mlx.h"
#include "ESLinear.h"
#include <optional>

namespace es {
namespace mx = mlx::core;

class ESClippableLinear {
public:
    // Unclipped.
    explicit ESClippableLinear(ESLinear linear);
    // Clipped: bounds are 0-d arrays in the compute dtype.
    ESClippableLinear(ESLinear linear, mx::array inputMin, mx::array inputMax,
                      mx::array outputMin, mx::array outputMax);

    mx::array forward(const mx::array & x) const;
    bool clipped() const { return clipped_; }

private:
    ESLinear  linear_;
    bool      clipped_;
    mx::array inMin_, inMax_, outMin_, outMax_;
};

}  // namespace es
