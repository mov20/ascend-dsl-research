# Table 1: High-Level DSL Comparison

*Last updated: 2026-03-30*


> **Table 1 (this):** High-level language comparison
> **Table 2:** Syntax constructs & performance impact → `ascend-dsl-syntax-perf.md`

## Comparison Matrix

| | **Triton** | **TileLang** | **Pallas** | **cuTile** | **Helion** | **Gluon** | **Mojo** |
|---|---|---|---|---|---|---|---|
| **Developer** | OpenAI | tile-ai (open-source) | Google (JAX) | NVIDIA | Meta (PyTorch) | Lei Mao et al. | Modular |
| **Year** | 2021 | 2024 | 2023 | 2025 | 2025 | 2025 | 2023 |
| **Type** | Python DSL | Python DSL | Python DSL (JAX) | Python/Julia DSL | Python DSL (PyTorch) | Triton extension | Language (Python superset) |
| **Abstraction** | Block-level | Tile-level | Tile/Grid+Block | Tile-level | Tile-level (PyTorch ops) | Warp-level | Systems-level |
| **Target HW** | NVIDIA, AMD, Intel GPU | NVIDIA, AMD GPU, **Huawei Ascend (A2/A3)** | Google TPU, NVIDIA/AMD GPU | NVIDIA GPU (Ampere+) | NVIDIA, AMD, Intel GPU (via Triton) | NVIDIA GPU | NVIDIA, AMD GPU, CPU |
| **Backend/IR** | MLIR → LLVM → PTX/AMDGPU | TVM-based; Ascend: AscendC & PTO, AscendNPU IR | Mosaic (TPU) / Triton (GPU) | Tile IR → GPU machine code | Compiles to Triton → MLIR | Triton GPU IR | MLIR → LLVM |
| **Perf vs peak** | ~43% avg (A100), up to 90-99% tuned GEMM | ~1.0-1.1x cuBLAS (≈85-95% peak) | Varies; up to 2x vs torch.compile | Near-native (auto Tensor Cores) | 1.21x vs torch.compile, 1.85x vs Triton | >Triton (warp-level control) | 87-100% vs CUDA (memory-bound) |
| **Perf vs cuBLAS** | ~90-95% on GEMM (tuned) | 0.97-1.1x | N/A (TPU focus) | ~native | Exceeds Triton | >Triton | 87% compute, ~100% mem-bound |
| **Scheduling** | Automatic + hints | Automatic | Automatic + manual prefetch (TPU) | Automatic (compiler) | Autotuning engine | Explicit (warp-level) | Manual + autotune |
| **Memory mgmt** | Implicit (compiler) | Explicit tiling (DRAM↔SRAM) | Explicit (BlockSpec) | Implicit (compiler) | Implicit (PyTorch semantics) | Explicit (shared mem) | Manual |
| **LOC for GEMM** | ~30 | ~25 | ~40 | ~15 | ~15 | ~40 | ~50+ |
| **Ascend support** | ✅ **Triton-Ascend** (Huawei fork, gitee.com/ascend/triton-ascend) | ✅ **tilelang-ascend** (A2/A3, AscendC) | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Strengths** | Mature, large community, multi-vendor | Simple, fast, multi-vendor, **Ascend** | Native JAX/TPU integration | Minimal code, auto Tensor Cores | PyTorch-native, excellent perf | Max control, warp-level | Full language, not just kernels |
| **Weaknesses** | Block-level limits on non-NVIDIA; ~43% avg | Young, sparse docs | JAX lock-in, complex for GPU | NVIDIA only | Depends on Triton backend | NVIDIA only, niche | Closed parts, no TPU/Ascend |

## Key Observations

### 1. Trend: Tile-based abstractions dominate
TileLang, cuTile, Helion, Pallas — all use tile as the core primitive. Natural fit for accelerator memory hierarchies (HBM → L2 → SRAM/shared → registers).

### 2. TileLang — only third-party DSL with Ascend support
**TileLang-Ascend** (github.com/tile-ai/tilelang-ascend, released Sep 2025) — only Python DSL with Ascend NPU backend (A2/A3). Generates AscendC code, requires CANN ≥8.2. Key competitor/reference for our DSL.

### 3. Triton-Ascend — Huawei's official fork
**Triton-Ascend** (gitee.com/ascend/triton-ascend) — Huawei's own port of Triton to Ascend. Docs at ascend.github.io/triton-ascend.

### 4. Performance ceiling
- **Gluon** shows that exceeding Triton requires **warp/wave-level control**
- **cuTile** demonstrates that **compiler automation** can deliver near-native with minimal code
- **Helion** proves that **PyTorch-level abstraction + autotuning** can beat hand-written Triton

### 5. For 90% peak on Ascend:
- Explicit **Cube Unit** (matrix) vs **Vector Unit** (elementwise) vs **Scalar Unit** management
- Control over **double buffering** and **pipeline** between units
- Awareness of **AI Core** topology (multiple cores, L0/L1/L2/HBM memory hierarchy)

### 6. Syntax patterns worth borrowing
| Pattern | Source | Why it matters |
|---------|--------|----------------|
| `tile` loop as core primitive | Helion, TileLang | Intuitive, minimal code |
| Automatic scheduling + autotuning | Helion, cuTile | Key to simplicity |
| Explicit memory hints | Pallas (prefetch), Gluon | Required for 90% peak |
| PyTorch-compatible syntax | Helion | Low barrier to entry |
| Warp/wave-level escape hatch | Gluon | For edge cases when auto isn't enough |

---

## References

### DSLs Analyzed

#### Triton (OpenAI)
- Repo: https://github.com/triton-lang/triton
- Docs: https://triton-lang.org
- Paper: *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (Tillet et al., 2019) — https://dl.acm.org/doi/10.1145/3315508.3329973

#### TileLang (tile-ai)
- Repo: https://github.com/tile-ai/tilelang
- Docs: https://tilelang.readthedocs.io

#### Pallas (Google / JAX)
- Docs: https://jax.readthedocs.io/en/latest/pallas/index.html
- Source (in JAX): https://github.com/google/jax/tree/main/jax/experimental/pallas

#### cuTile (NVIDIA)
- Blog: https://developer.nvidia.com/blog/cutile-a-new-programming-model-for-nvidia-gpus/
- Repo: https://github.com/NVIDIA/cutile

#### Helion (Meta / PyTorch)
- Repo: https://github.com/pytorch-labs/helion
- Blog: https://pytorch.org/blog/helion

#### Gluon
- Repo: https://github.com/gluon-lang/gluon *(verify — may refer to different project)*
- Paper: *Gluon: A GPU Kernel Language with Warp-level Control* — search arxiv

#### Mojo (Modular)
- Site: https://www.modular.com/mojo
- Docs: https://docs.modular.com/mojo
- Repo (stdlib): https://github.com/modularml/mojo

---

### Ascend-Specific Projects

#### Triton-Ascend (Huawei)
- Repo: https://gitee.com/ascend/triton-ascend
- Docs: https://ascend.github.io/triton-ascend
- Mirror/discussion: https://gitcode.com/ascend/triton-ascend

#### TileLang-Ascend (tile-ai)
- Repo: https://github.com/tile-ai/tilelang-ascend
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
