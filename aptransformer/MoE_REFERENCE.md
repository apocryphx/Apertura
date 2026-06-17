# Apertura MoE Reference — Gemma-4-26B-a4b (annotated)

A standing explanation of how the Mixture-of-Experts block works in Apertura, the three
execution paths, and the exact knobs that switch between them. Source of truth is the code in
`ESRouter.{h,mm}`, `ESExperts.{h,mm}`, `ESDecoderLayer.mm`; this file mirrors the key pieces with
step-by-step commentary. (Gemma-4-31B is *dense* — `enable_moe_block=false` — and uses none of this.)

---

## 0. The big picture

In the 26B-a4b, **each decoder layer runs a dense MLP AND a sparse Mixture-of-Experts block in
parallel and sums them.** The model is "a4b" = ~4B *active* params out of 25.8B total: every token
routes to only **top-8 of 128 experts**. The experts are ~88% of the model's weights, so how you
evaluate them dominates both correctness and speed.

```
                 ┌─ pre_ff_ln(x1) → MLP → post_ff_ln_1 ─┐         (dense path, always full)
   x1 (residual)─┤                                       +→ post_feedforward_layernorm → +x1 → ×layer_scalar
                 └─ router(x1) ┐                          │
                    pre_ff_ln_2(x1) → EXPERTS ┘→ post_ff_ln_2 ─┘   (sparse path)
```

Three ways to evaluate the EXPERTS, selected by config (see §5):
1. **dense**     — evaluate all 128 experts, mask by router weights. Correctness path. (reads all 25.8B/token)
2. **sparse**    — `gather_mm` only the top-k experts.                 (reads ~4B active/token)
3. **sparse+Q**  — `gather_qmm` only the top-k experts, quantized.     (reads ~4B active × bits/16)

Measured 26B decode (M4 Max, warmed): dense ~9 → sparse 38 → sparse+fused 44 → sparse+fused+Q8 **60 tok/s**.

---

## 1. The Router — who goes where  (ESRouter::routeTopK)

Picks each token's experts and their weights. Returns the top-k indices + weights (sparse) or a
dense `[seq, E]` weight matrix (dense path calls this then scatters).

```cpp
ESRouter::TopK ESRouter::routeTopK(const mx::array & x) const {   // x: [seq, hidden]
    const int seq = x.shape(0);

    // (a) Weightless RMSNorm in f32, then per-dim `scale` and the 1/sqrt(hidden) factor.
    //     (Gemma4TextRouter: norm has with_scale=false; scalar_root_size = hidden^-0.5.)
    mx::array xf  = mx::astype(x, mx::float32);
    mx::array ms  = mx::mean(mx::multiply(xf, xf), -1, /*keepdims=*/true);
    mx::array h   = mx::multiply(xf, mx::rsqrt(mx::add(ms, mx::array(eps_, mx::float32))));
    h = mx::multiply(mx::multiply(h, mx::astype(scale_, mx::float32)),
                     mx::array(scalarRoot_, mx::float32));

    // (b) Expert logits → probabilities over all E experts.
    mx::array scores = mx::matmul(h, mx::transpose(mx::astype(proj_, mx::float32)));  // [seq, E]
    mx::array probs  = mx::softmax(scores, -1, /*precise=*/true);

    // (c) Top-k by probability (descending). argsort(-probs) then take the first k columns.
    mx::array order  = mx::argsort(mx::negative(probs), -1);                           // [seq, E]
    mx::array topIdx = mx::astype(mx::slice(order, {0, 0}, {seq, topK_}), mx::int32);  // [seq, k]
    mx::array topP   = mx::take_along_axis(probs, topIdx, -1);                         // [seq, k]

    // (d) Renormalize the k weights to sum to 1 per token, then apply the per-expert scale.
    mx::array topW = mx::divide(topP, mx::sum(topP, -1, /*keepdims=*/true));
    topW = mx::multiply(topW, mx::take(mx::astype(perExpertScale_, mx::float32), topIdx, 0));

    return {topIdx, mx::astype(topW, computeDtype_)};   // idx [seq,k] int32, w [seq,k]
}
```

The dense weight matrix (`routeWeights`) just scatters those k weights into a `[seq, E]` zeros row
via `put_along_axis` — non-chosen experts get weight 0.

**The `topK_` here is the "number of experts" switch (§5).** Sweeping it 128→4 (the `--expert-ladder`)
showed the MoE is remarkably robust because the *dense MLP path runs regardless* — top-k only
perturbs half the feedforward.

---

## 2. Experts — DENSE path  (ESExperts::forward) — correctness

Evaluate **every** expert, combine by the dense router weights `W [seq, E]`. Exactly equals the
PyTorch scatter loop (unchosen experts contribute 0), but reads all expert weights → slow.

```cpp
mx::array ESExperts::forward(const mx::array & x, const mx::array & W) const {  // x [seq,hidden], W [seq,E]
    const int seq = x.shape(0);
    mx::array xb = mx::expand_dims(x, 0);                              // [1, seq, hidden]
    mx::array gu = mx::matmul(xb, mx::transpose(gateUp_, {0, 2, 1}));  // [E, seq, 2I]  (all experts!)
    auto halves  = mx::split(gu, 2, -1);                              // gate, up : [E, seq, I]
    mx::array y  = mx::multiply(geluTanh(halves[0]), halves[1]);       // SwiGLU [E, seq, I]
    y = mx::matmul(y, mx::transpose(down_, {0, 2, 1}));               // [E, seq, hidden]
    mx::array yt = mx::transpose(y, {1, 0, 2});                       // [seq, E, hidden]
    return mx::sum(mx::multiply(mx::expand_dims(W, -1), yt), 1);       // out[s]=Σ_e W[s,e]·y[e,s] → [seq,hidden]
}
```

---

## 3. Experts — SPARSE path  (ESExperts::sparseForward) — fast

Touch **only the top-k** experts via matrix-level gather (mlx-lm SwitchGLU pattern). The gather
reads just the selected experts' weight rows from DRAM — the actual MoE bandwidth win.

```cpp
mx::array ESExperts::sparseForward(const mx::array & x,            // [seq, hidden]
                                   const mx::array & idx,          // [seq, k] int32 (from router)
                                   const mx::array & w) const {    // [seq, k]   weights
    mx::array xe = mx::expand_dims(x, std::vector<int>{-2, -3});   // [seq, 1, 1, hidden]

    // gate_up: gather only the k experts per token, then matmul.
    //   bf16  → gather_mm on the swapaxed weight [E, hidden, 2I]
    //   quant → gather_qmm on the quantized weight (transpose=true expects [E, out, in] quant'd along in)
    mx::array gu = quant_
        ? mx::gather_qmm(xe, gateUpQ_, gateUpS_, gateUpB_, std::nullopt, idx, /*transpose=*/true, gs_, bits_)
        : mx::gather_mm(xe, mx::swapaxes(gateUp_, -1, -2), std::nullopt, idx);     // [seq, k, 1, 2I]
    auto halves = mx::split(gu, 2, -1);                            // gate, up : [seq, k, 1, I]
    mx::array y = mx::multiply(geluTanh(halves[0]), halves[1]);    // SwiGLU

    // down projection (same gather choice).
    mx::array out = quant_
        ? mx::gather_qmm(y, downQ_, downS_, downB_, std::nullopt, idx, /*transpose=*/true, gs_, bits_)
        : mx::gather_mm(y, mx::swapaxes(down_, -1, -2), std::nullopt, idx);        // [seq, k, 1, hidden]
    out = mx::squeeze(out, -2);                                    // [seq, k, hidden]
    return mx::sum(mx::multiply(mx::expand_dims(w, -1), out), 1);  // weighted sum over k → [seq, hidden]
}
```

Quantized weights are built once in the constructor:
```cpp
if (quant_) {                                  // quantBits>0
    auto gq = mx::quantize(gateUp_, gs_, bits_);  gateUpQ_=gq[0]; gateUpS_=gq[1]; gateUpB_=gq[2];
    auto dq = mx::quantize(down_,   gs_, bits_);  downQ_  =dq[0]; downS_  =dq[1]; downB_  =dq[2];
}                                              // bf16 gateUp_/down_ kept so the dense path still works
```
> Note: `gather_qmm(..., transpose=true)` wants the weight as `[E, out, in]` quantized along `in`
> (the last axis) — which is exactly `mx::quantize(gateUp_[E,2I,hidden])`. The bf16 `gather_mm`
> path instead needs the weight pre-`swapaxes`'d to `[E, in, out]`. That asymmetry is why the two
> branches look different.

---

## 4. The layer combine  (ESDecoderLayer::forward, MoE section)

```cpp
residual = x1;                                       // x1 = hidden after attention + residual
mx::array hmlp = mlp_.forward(preFFLN_.forward(x1));  // dense MLP path input = pre_ff_ln(x1)

mx::array ff = hmlp;                                  // dense (31B) layers stop here
if (enableMoe_) {
    mx::array dense = postFFLN1_->forward(hmlp);                  // dense MLP path, normed
    mx::array expIn = preFFLN2_->forward(x1);                     // experts read pre_ff_ln_2(x1)
    mx::array moe = moeSparse_                                    // ── the path switch ──
        ? [&]{ ESRouter::TopK tk = router_->routeTopK(x1);        //   sparse: top-k indices+weights
               return experts_->sparseForward(expIn, tk.idx, tk.w); }()
        : experts_->forward(expIn, router_->routeWeights(x1));    //   dense: all experts + W matrix
    ff = mx::add(dense, postFFLN2_->forward(moe));               // sum the two feedforward paths
}
h  = postFFLN_.forward(ff);                           // wrap the sum
x2 = mx::add(residual, h);                            // residual
return mx::multiply(x2, layerScalar_);                // × per-layer scalar
```
Router and experts both read the **pre-norm** residual `x1` (router gets it raw; experts get
`pre_ff_ln_2(x1)`). The dense MLP path is independent of every MoE switch.

---

## 5. THE SWITCHES — how to change experts & quantization

All MoE behavior is config (`ESModelConfig`) + driver flags. The model architecture
(`enable_moe_block`, `num_experts`, `top_k_experts`, `moe_intermediate_size`) is parsed from the
HF `config.json`; the rest are runtime knobs.

| Knob (`ESModelConfig`)   | Meaning                                              | Driver flag        | Default |
|--------------------------|------------------------------------------------------|--------------------|---------|
| `enableMoeBlock`         | run the MoE block at all (from config.json)          | —                  | per model |
| `topKExperts`            | **how many experts per token** (the 128↔8↔4 knob)    | `--expert-ladder` sweeps it | from config (8) |
| `moeSparse`              | dense (all experts) vs sparse gather                 | `--moe-sparse`     | false (dense) |
| `quantBits`              | weight quant bits (also quantizes the experts)       | `--quant N`        | 0 (bf16) |
| `quantGroupSize`         | quant group size                                     | —                  | 64 |

### Switch the NUMBER OF EXPERTS (128 vs 8 vs 4 …)
Set `config.topKExperts = k`. In the router, step (c) takes the first `k` of the descending-sorted
experts; the dense path still computes all 128 but only `k` get nonzero weight, the sparse path
gathers only those `k`. From the CLI, the ladder sweeps it automatically:
```
AperturaResearch <26b_dir> <fixtures> --expert-ladder <chat_ids> --decode 200
```
To pin a single value in code: `config.topKExperts = 4;` before constructing the LM. (The trained
value is 8; >8 over-selects with dilution, <8 under-selects — both stay coherent, see the ladder note.)

### Switch DENSE ↔ SPARSE (the speed knob)
- **dense** (default): every expert evaluated — exact, slow. `config.moeSparse = false`.
- **sparse**: only top-k gathered — fast. `config.moeSparse = true`  /  CLI `--moe-sparse`.
Same output (sparse = dense restricted to the nonzero experts), so argmax/greedy match.

### Switch QUANTIZATION
- `config.quantBits = 8` (CLI `--quant 8`) quantizes attention + dense MLP (via `ESLinear`) **and**
  the experts (3D `gather_qmm`). Experts are 88% of the weights, so this is the dominant lever.
- Only the **sparse** path uses quantized experts (`gather_qmm`); the dense path keeps bf16 experts.
  So quantization is meaningful with `--moe-sparse`.
- `--quant-embed 8` additionally quantizes the tied embedding / LM head.

### Recipes
```
# Exact reference (slow, all experts, bf16):
AperturaResearch <26b> <fixtures>                      # dense bf16

# Fast, exact-equivalent (top-8 sparse, bf16):
AperturaResearch <26b> <fixtures> --moe-sparse --fused

# Fastest (top-8 sparse, Q8 experts, fused attention)  ~60 tok/s decode:
AperturaResearch <26b> <fixtures> --moe-sparse --fused --quant 8

# Benchmark dense vs sparse (bf16) or sparse-Q8:
AperturaResearch <26b> <fixtures> --bench --fused [--quant 8] --decode 64

# Expert-count ablation, greedy, top_k 128→4:
AperturaResearch <26b> <fixtures> --expert-ladder <chat_ids> --decode 200
```

---

## 6. Why the speeds are what they are (one sentence)

Decode is memory-bandwidth-bound, so every lever is "read fewer weight bytes per token": **sparse**
reads 8 experts not 128 (≈6×), **Q8** halves each weight's bytes, **fused** trims the dispatch
overhead the savings expose. Prefill is compute-bound, so quantization can *slow* it (dequant cost
> bandwidth saved) — same wall, opposite sign.
