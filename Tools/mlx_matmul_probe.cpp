// One bf16 matmul, computed on MLX's CPU backend and MLX's Metal backend.
// Isolates per-op backend variance with no model and no accumulation chain,
// so it can be compared directly against PyTorch's CPU-vs-MPS spread.
#include "mlx/mlx.h"
#include <cstdio>
#include <unordered_map>

namespace mx = mlx::core;

int main() {
    const int M = 7, K = 5376, N = 5376;   // seq x hidden x hidden, Apertura's shape

    mx::random::seed(0);
    mx::array a = mx::astype(mx::random::normal({M, K}), mx::bfloat16);
    mx::array b = mx::astype(mx::random::normal({K, N}), mx::bfloat16);
    mx::eval(a, b);

    // Same inputs, same dtype, only the device differs.
    mx::array on_cpu = mx::matmul(a, b, mx::Device::cpu);
    mx::array on_gpu = mx::matmul(a, b, mx::Device::gpu);
    mx::eval(on_cpu, on_gpu);

    // fp64 reference is unavailable in MLX; fp32 on CPU is the closest thing and
    // is enough to say which backend sits nearer the higher-precision answer.
    mx::array a32 = mx::astype(a, mx::float32), b32 = mx::astype(b, mx::float32);
    mx::array ref32 = mx::matmul(a32, b32, mx::Device::cpu);
    mx::eval(ref32);

    std::unordered_map<std::string, mx::array> out;
    out.insert({"mlx_cpu",  mx::astype(on_cpu, mx::float32)});
    out.insert({"mlx_gpu",  mx::astype(on_gpu, mx::float32)});
    out.insert({"mlx_ref32", ref32});
    out.insert({"a", mx::astype(a, mx::float32)});
    out.insert({"b", mx::astype(b, mx::float32)});
    for (auto& kv : out) mx::eval(kv.second);
    mx::save_safetensors("mlx_matmul_probe.safetensors", out);

    mx::array d = mx::abs(mx::subtract(mx::astype(on_cpu, mx::float32),
                                       mx::astype(on_gpu, mx::float32)));
    mx::array mx_d = mx::max(d), mean_d = mx::mean(d);
    mx::eval(mx_d, mean_d);
    std::printf("MLX cpu vs gpu:  max|d| %.4e   mean|d| %.4e\n",
                mx_d.item<float>(), mean_d.item<float>());
    std::printf("wrote mlx_matmul_probe.safetensors\n");
    return 0;
}
