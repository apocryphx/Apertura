#import "MLXTest.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

// Inner C++ function — C++ exceptions flow cleanly here
static NSString* _mlxCheck() {
    mx::array a = mx::array({1.0f, 2.0f, 3.0f}, {3}, mx::float32);
    mx::array b = mx::array({4.0f, 5.0f, 6.0f}, {3}, mx::float32);
    mx::array c = mx::add(a, b);
    mx::eval(c);

    const float* data = c.data<float>();
    const float expected[] = {5.0f, 7.0f, 9.0f};
    for (int i = 0; i < 3; i++) {
        if (data[i] != expected[i]) {
            return [NSString stringWithFormat:
                @"MLX result mismatch at [%d]: got %.1f, expected %.1f",
                i, data[i], expected[i]];
        }
    }
    return nil;
}

/// Returns nil on success, or an error string on failure.
NSString* MLXSanityCheck(void) {
    try {
        return _mlxCheck();
    } catch (const std::exception& e) {
        return [NSString stringWithFormat:@"MLX error: %s", e.what()];
    }
}
