# Table 1: High-Level DSL Comparison

*Last updated: 2026-03-30*

> **Table 1 (this):** High-level language comparison
> **Table 2:** Syntax constructs & performance impact → `ascend-dsl-syntax-perf.md`

## Comparison Matrix

| | **Triton** | **TileLang** | **Pallas** | **cuTile** | **Helion** | **Gluon** | **Mojo** |
|---|---|---|---|---|---|---|---|
| **Developer** | OpenAI [1] | tile-ai (open-source) [2] | Google (JAX) [3] | NVIDIA [4] | Meta (PyTorch) [5] | OpenAI [6] | Modular [7] |
| **Year** | 2021 [1] | 2024 [2] | 2023 [3] | 2025 [4] | 2025 [5] | 2025 [6] | 2023 [7] |
| **Type** | Python DSL | Python DSL | Python DSL (JAX) | Python/Julia DSL | Python DSL (PyTorch) | Triton extension | Language (Python superset) |
| **Abstraction** | Block-level | Tile-level | Tile/Grid+Block | Tile-level | Tile-level (PyTorch ops) | Warp-level | Systems-level |
| **Target HW** | NVIDIA, AMD, Intel GPU | NVIDIA, AMD GPU, **Huawei Ascend (A2/A3)** [2] | Google TPU, NVIDIA/AMD GPU | NVIDIA GPU (Ampere+) [4] | NVIDIA, AMD, Intel GPU (via Triton) [5] | NVIDIA GPU (H100, B200) [6] | NVIDIA, AMD GPU, CPU [7] |
| **Backend/IR** | MLIR → LLVM → PTX/AMDGPU [1] | TVM-based; Ascend: AscendC & PTO, AscendNPU IR [2] | Mosaic (TPU) / Triton (GPU) [3] | Tile IR → GPU machine code [4] | Compiles to Triton → MLIR [5] | Triton GPU IR [6] | MLIR → LLVM [7] |
| **Perf vs peak** | Out-of-the-box ~80% peak [8]; Flash Attention without warp specialization: 45% compute throughput on H100, with WS: 69% [9]; 90%+ only with heavily tuned kernels (GEMM, FA3) | ~1.0–1.1x cuBLAS on H100 (≈85–95% peak) [2]; Ascend A2/A3: no published numbers (Mar 2026) | Up to 2x vs torch.compile on TPU v4/v5 [3]; GPU perf varies | Near-native on Ampere/Hopper (auto Tensor Cores) [4]; no third-party benchmarks | 1.21x vs torch.compile, 1.85x vs hand-written Triton on H100 GEMM [5] | FMHA on B200: still slower than cuDNN out-of-the-box [8]; better than stock Triton | ~87% vs CUDA compute-bound; ~100% mem-bound on A100 [10] |
| **Supported kernels** | GEMM, Flash Attention (FA2/FA3), softmax, layer norm, RMSNorm, MoE gating, element-wise fused ops, scan/reduction, dropout+bias fusions [1] | GPU: GEMM (incl. batch, int4 dequant, split-K), Flash Attention, MLA decoding, FlashMLA, softmax, layer norm, element-wise, reduction [2]; **Ascend:** GEMM, batch GEMM, conv, Flash Attention, softmax, normalization, activation, GEMV, grouped GEMM, sparse FA, positional embedding, cross-entropy, linear attention/RNN, top-k [11] | GEMM, Flash Attention (TPU/GPU), layer norm, softmax, element-wise ops, scan/reduction, custom attention masks [3] | GEMM, vector add; attention/softmax via tile abstractions — limited published examples (Blackwell-only, Dec 2025) [4] | GEMM (15 LOC), softmax (2.28x torch.compile), RMSNorm (bwd), JSD (6.22x torch.compile), int4-GEMM (4.5x Triton), Mamba-2 chunk-scan, Flash Attention [5] | FMHA (Flash Attention style, B200/H100) with warp specialization; MMA (WGMMA); TMA load/store; experimental [6] | GEMM, matmul, stencil, BabelStream (memory ops), element-wise; softmax/attention via custom fused kernels; primary focus: systems-level portability [10] |
| **Scheduling** | Automatic + hints | Automatic | Automatic + manual prefetch (TPU) | Automatic (compiler) | Autotuning engine | Explicit (warp-level) | Manual + autotune |
| **Memory mgmt** | Implicit (compiler) | Explicit tiling (DRAM↔SRAM) | Explicit (BlockSpec) | Implicit (compiler) | Implicit (PyTorch semantics) | Explicit (shared mem) | Manual |
| **LOC for GEMM** | ~30 [1] | ~25 [2] | ~40 [3] | ~15 [4] | ~15 [5] | ~40 [6] | ~50+ [7] |
| **Ascend support** | ✅ **Triton-Ascend** (Huawei fork, gitcode.com/Ascend/triton-ascend) [12] | ✅ **tilelang-ascend** (A2/A3, AscendC, Sep 2025) [11] | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Strengths** | Mature, large community, multi-vendor | Simple, fast, multi-vendor, **Ascend** | Native JAX/TPU integration | Minimal code, auto Tensor Cores | PyTorch-native, excellent perf | Max control, warp-level | Full language, not just kernels |
| **Weaknesses** | Block-level limits on non-NVIDIA; out-of-box ~80% peak [8] | Young, sparse docs | JAX lock-in, complex for GPU | NVIDIA only (Blackwell+); very early | Depends on Triton backend | NVIDIA only, not for general use [8] | Closed parts, no TPU/Ascend |

---

## Key Observations

### 1. Trend: Tile-based abstractions dominate
TileLang, cuTile, Helion, Pallas — all use tile as the core primitive. Natural fit for accelerator memory hierarchies (HBM → L2 → SRAM/shared → registers).

### 2. TileLang — only third-party DSL with Ascend support
**TileLang-Ascend** [11] (released Sep 2025) — only Python DSL with Ascend NPU backend (A2/A3). Generates AscendC code, requires CANN ≥8.2. Key competitor/reference for our DSL.

### 3. Triton-Ascend — Huawei's official fork
**Triton-Ascend** [12] — Huawei's own port of Triton to Ascend. Active development at gitcode.com/Ascend/triton-ascend.

### 4. Performance ceiling
- **Gluon** shows that exceeding Triton requires **warp/wave-level control** [8]
- **cuTile** demonstrates that **compiler automation** can deliver near-native with minimal code [4]
- **Helion** proves that **PyTorch-level abstraction + autotuning** can beat hand-written Triton (1.85x on H100 GEMM) [5]

### 5. For 90% peak on Ascend:
- Explicit **Cube Unit** (matrix) vs **Vector Unit** (elementwise) vs **Scalar Unit** management
- Control over **double buffering** and **pipeline** between units
- Awareness of **AI Core** topology (multiple cores, L0/L1/L2/HBM memory hierarchy)

### 6. Syntax patterns worth borrowing
| Pattern | Source | Why it matters |
|---------|--------|----------------|
| `tile` loop as core primitive | Helion [5], TileLang [2] | Intuitive, minimal code |
| Automatic scheduling + autotuning | Helion [5], cuTile [4] | Key to simplicity |
| Explicit memory hints | Pallas [3], Gluon [6] | Required for 90% peak |
| PyTorch-compatible syntax | Helion [5] | Low barrier to entry |
| Warp/wave-level escape hatch | Gluon [6] | For edge cases when auto isn't enough |

---

## Source Notes

| # | Source |
|---|--------|
| [1] | triton-lang/triton — github.com/triton-lang/triton |
| [2] | tile-ai/tilelang — github.com/tile-ai/tilelang |
| [3] | JAX Pallas docs — jax.readthedocs.io/en/latest/pallas |
| [4] | NVIDIA cuTile blog + docs — developer.nvidia.com/blog/…cutile… |
| [5] | Helion blog (Meta/PyTorch) — pytorch.org/blog/helion |
| [6] | Gluon in triton repo — github.com/triton-lang/triton/…/gluon |
| [7] | Modular/Mojo — modular.com/mojo; arxiv.org/abs/2509.21039 |
| [8] | Triton community meetup notes, Jul 9 2025 — github.com/triton-lang/triton/…/meetups/07-09-2025 |
| [9] | Triton community meetup notes, Mar 12 2025 — github.com/triton-lang/triton/…/meetups/03-12-2025 |
| [10] | Mojo portability paper — arxiv.org/abs/2509.21039 |
| [11] | tile-ai/tilelang-ascend examples — github.com/tile-ai/tilelang-ascend/…/examples |
| [12] | Triton-Ascend primary repo — gitcode.com/Ascend/triton-ascend |

---

## Full References

### DSLs Analyzed

#### Triton (OpenAI)
- Repo: https://github.com/triton-lang/triton
- Docs: https://triton-lang.org
- Community meetups: https://github.com/triton-lang/triton/tree/main/docs/meetups
- Paper: *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (Tillet et al., 2019) — https://dl.acm.org/doi/10.1145/3315508.3329973

#### TileLang (tile-ai)
- Repo: https://github.com/tile-ai/tilelang
- Docs: https://tilelang.readthedocs.io
- Paper: https://arxiv.org/abs/2504.17577

#### Pallas (Google / JAX)
- Docs: https://jax.readthedocs.io/en/latest/pallas/index.html
- Source (in JAX): https://github.com/google/jax/tree/main/jax/experimental/pallas

#### cuTile (NVIDIA)
- Blog: https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/
- Docs: https://docs.nvidia.com/cuda/cutile-python/
- Repo: https://github.com/nvidia/cutile-python

#### Helion (Meta / PyTorch)
- Repo: https://github.com/pytorch/helion
- Blog: https://pytorch.org/blog/helion

#### Gluon (OpenAI)
- Triton repo (tutorials): https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon
- Blog post: https://www.lei.chat/posts/gluon-explicit-performance/

#### Mojo (Modular)
- Site: https://www.modular.com/mojo
- Docs: https://docs.modular.com/mojo
- Repo (stdlib): https://github.com/modularml/mojo
- Portability paper: https://arxiv.org/abs/2509.21039

---

### Ascend-Specific Projects

#### Triton-Ascend (Huawei)
- **Primary repo (active development):** https://gitcode.com/Ascend/triton-ascend
- Mirror: https://github.com/Ascend/triton-ascend
- Docs: https://ascend.github.io/triton-ascend
- Related ops/tutorials: https://github.com/Ascend/triton-ascend-ops
- PyPI: https://pypi.org/project/triton-ascend/
- Requirements: torch==2.6.0, torch-npu==2.6.0rc1

#### TileLang-Ascend (tile-ai)
- Repo: https://github.com/tile-ai/tilelang-ascend
- Examples: https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples
- Released: September 2025
- Targets: Ascend A2/A3 NPU
- Requirements: CANN ≥ 8.2, torch-npu ≥ 2.6

#### AscendCraft
- Paper: *AscendCraft: LLM-Driven Automatic AscendC Kernel Generation* — https://arxiv.org/abs/2601.22760
- Type: LLM-driven DSL → AscendC auto-generation (not a hand-programming DSL)

---

### Huawei Ascend / CANN

- CANN (Compute Architecture for Neural Networks): https://www.hiascend.com/software/cann
- AscendC Programming Guide: https://www.hiascend.com/document/detail/en/canncommercial/80RC1/operatordev/tbeaicpudevg/atlasascendc_10_0001.html
- Ascend Community: https://www.hiascend.com/en/
- MindSpore: https://www.mindspore.cn/en

---

### Background & IR

- MLIR: https://mlir.llvm.org
- MLIR Dialects overview: https://mlir.llvm.org/docs/Dialects/
- Triton's MLIR pipeline (TritonGPU IR): https://triton-lang.org/main/dialects/dialects.html
