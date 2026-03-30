# Table 1: High-Level DSL Comparison

*Last updated: 2026-03-30*

> **Table 1 (this):** High-level language comparison
> **Table 2:** Syntax constructs & performance impact → `ascend-dsl-syntax-perf.md`

## Comparison Matrix

| | **Triton** | **TileLang** | **Pallas** | **cuTile** | **Helion** | **Gluon** | **Mojo** |
|---|---|---|---|---|---|---|---|
| **Developer** | OpenAI <sup>[[1]](#ref-1)</sup> | tile-ai (open-source) <sup>[[2]](#ref-2)</sup> | Google (JAX) <sup>[[3]](#ref-3)</sup> | NVIDIA <sup>[[4]](#ref-4)</sup> | Meta (PyTorch) <sup>[[5]](#ref-5)</sup> | OpenAI <sup>[[6]](#ref-6)</sup> | Modular <sup>[[7]](#ref-7)</sup> |
| **Year** | 2021 | 2024 | 2023 | 2025 | 2025 | 2025 | 2023 |
| **Type** | Python DSL | Python DSL | Python DSL (JAX) | Python/Julia DSL | Python DSL (PyTorch) | Triton extension | Language (Python superset) |
| **Abstraction** | Block-level | Tile-level | Tile/Grid+Block | Tile-level | Tile-level (PyTorch ops) | Warp-level | Systems-level |
| **Target HW** | NVIDIA, AMD, Intel GPU | NVIDIA, AMD GPU, **Huawei Ascend (A2/A3)** <sup>[[2]](#ref-2)</sup> | Google TPU, NVIDIA/AMD GPU | NVIDIA GPU (Ampere+) <sup>[[4]](#ref-4)</sup> | NVIDIA, AMD, Intel GPU (via Triton) <sup>[[5]](#ref-5)</sup> | NVIDIA GPU (H100, B200) <sup>[[6]](#ref-6)</sup> | NVIDIA, AMD GPU, CPU <sup>[[7]](#ref-7)</sup> |
| **Backend/IR** | MLIR → LLVM → PTX/AMDGPU <sup>[[1]](#ref-1)</sup> | TVM-based; Ascend: AscendC & PTO, AscendNPU IR <sup>[[2]](#ref-2)</sup> | Mosaic (TPU) / Triton (GPU) <sup>[[3]](#ref-3)</sup> | Tile IR → GPU machine code <sup>[[4]](#ref-4)</sup> | Compiles to Triton → MLIR <sup>[[5]](#ref-5)</sup> | Triton GPU IR <sup>[[6]](#ref-6)</sup> | MLIR → LLVM <sup>[[7]](#ref-7)</sup> |
| **Perf vs peak** | 1. Out-of-the-box: ~80% peak <sup>[[8]](#ref-8)</sup><br>2. Flash Attention (no WS): 45% compute throughput on H100 <sup>[[9]](#ref-9)</sup><br>3. Flash Attention (with WS): 69% <sup>[[9]](#ref-9)</sup><br>4. 90%+ only with heavily tuned kernels (GEMM, FA3) | 1. ~1.0–1.1x cuBLAS on H100 (≈85–95% peak) <sup>[[2]](#ref-2)</sup><br>2. Ascend A2/A3: no published numbers (Mar 2026) | 1. Up to 2x vs torch.compile on TPU v4/v5 <sup>[[3]](#ref-3)</sup><br>2. GPU: varies by kernel | 1. Near-native on Ampere/Hopper (auto Tensor Cores) <sup>[[4]](#ref-4)</sup><br>2. No third-party benchmarks available | 1. 1.21x vs torch.compile on H100 GEMM <sup>[[5]](#ref-5)</sup><br>2. 1.85x vs hand-written Triton on H100 GEMM <sup>[[5]](#ref-5)</sup> | 1. FMHA on B200: still slower than cuDNN out-of-the-box <sup>[[8]](#ref-8)</sup><br>2. Better than stock Triton | 1. ~87% vs CUDA (compute-bound) <sup>[[10]](#ref-10)</sup><br>2. ~100% vs CUDA (memory-bound) <sup>[[10]](#ref-10)</sup> |
| **Supported kernels** | 1. GEMM, Flash Attention (FA2/FA3) <sup>[[1]](#ref-1)</sup><br>2. Softmax, layer norm, RMSNorm<br>3. MoE gating<br>4. Element-wise fused ops<br>5. Scan/reduction<br>6. Dropout+bias fusions | **GPU** <sup>[[2]](#ref-2)</sup><br>1. GEMM (batch, int4 dequant, split-K)<br>2. Flash Attention, FlashMLA<br>3. MLA decoding<br>4. Softmax, layer norm<br>5. Element-wise, reduction<br><br>**Ascend** <sup>[[11]](#ref-11)</sup><br>1. GEMM, batch GEMM, GEMV<br>2. Grouped GEMM, conv<br>3. Flash Attention, sparse FA<br>4. Softmax, normalization, activation<br>5. Positional embedding<br>6. Cross-entropy<br>7. Linear attention/RNN<br>8. Top-k | 1. GEMM (matmul) <sup>[[3]](#ref-3)</sup><br>2. Flash Attention (TPU/GPU)<br>3. Layer norm, softmax<br>4. Element-wise ops<br>5. Scan/reduction<br>6. Custom attention masks | 1. GEMM <sup>[[4]](#ref-4)</sup><br>2. Vector add<br>3. Attention/softmax via tile abstractions<br>_(limited examples, Blackwell-only, Dec 2025)_ | 1. GEMM 15 LOC <sup>[[5]](#ref-5)</sup><br>2. Softmax (2.28x torch.compile)<br>3. RMSNorm (bwd)<br>4. JSD (6.22x torch.compile)<br>5. int4-GEMM (4.5x Triton)<br>6. Mamba-2 chunk-scan<br>7. Flash Attention | 1. FMHA / Flash Attention <sup>[[6]](#ref-6)</sup><br>2. MMA (WGMMA)<br>3. TMA load/store<br>_(experimental, B200/H100 only)_ | 1. GEMM, matmul <sup>[[10]](#ref-10)</sup><br>2. Stencil, BabelStream<br>3. Element-wise ops<br>4. Softmax/attention (custom, no official examples) |
| **AI Models / Frameworks** | 1. LLaMA 2/3, Llama 3.2 (training via Liger Kernel <sup>[[16]](#ref-16)</sup>)<br>2. Mistral, Gemma, Qwen2-VL, Phi (Liger Kernel / HF Trainer)<br>3. DeepSeek (FlashAttention-2/3 <sup>[[13]](#ref-13)</sup>, vLLM)<br>4. GPT-series (OpenAI internal, torch.compile)<br>5. Most open LLMs via vLLM / SGLang inference<br>6. **All models** using `torch.compile` (PyTorch Inductor → Triton) | 1. DeepSeek MLA (FlashMLA <sup>[[14]](#ref-14)</sup>)<br>2. General LLM inference via vLLM (2.91x latency vs baseline) | 1. Gemini (Google, TPU <sup>[[15]](#ref-15)</sup>)<br>2. Anthropic models (Google Cloud TPU)<br>3. xAI models (Google Cloud TPU)<br>4. Apple models (Google Cloud TPU)<br>5. RecurrentGemma, MaxText-based models | No production models yet (Dec 2025) | No production models yet (beta Oct 2025) | OpenAI production models (internal, undisclosed) <sup>[[8]](#ref-8)</sup> | 1. Modular MAX inference engine<br>2. No public LLM integrations yet |
| **Scheduling** | Automatic + hints | Automatic | Automatic + manual prefetch (TPU) | Automatic (compiler) | Autotuning engine | Explicit (warp-level) | Manual + autotune |
| **Memory mgmt** | Implicit (compiler) | Explicit tiling (DRAM↔SRAM) | Explicit (BlockSpec) | Implicit (compiler) | Implicit (PyTorch semantics) | Explicit (shared mem) | Manual |
| **LOC for GEMM** | ~30 <sup>[[1]](#ref-1)</sup> | ~25 <sup>[[2]](#ref-2)</sup> | ~40 <sup>[[3]](#ref-3)</sup> | ~15 <sup>[[4]](#ref-4)</sup> | ~15 <sup>[[5]](#ref-5)</sup> | ~40 <sup>[[6]](#ref-6)</sup> | ~50+ <sup>[[7]](#ref-7)</sup> |
| **Ascend support** | ✅ **Triton-Ascend** (Huawei fork, gitcode.com/Ascend/triton-ascend) <sup>[[12]](#ref-12)</sup> | ✅ **tilelang-ascend** (A2/A3, AscendC, Sep 2025) <sup>[[11]](#ref-11)</sup> | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Strengths** | Mature, large community, multi-vendor | Simple, fast, multi-vendor, **Ascend** | Native JAX/TPU integration | Minimal code, auto Tensor Cores | PyTorch-native, excellent perf | Max control, warp-level | Full language, not just kernels |
| **Weaknesses** | Block-level limits on non-NVIDIA; out-of-box ~80% peak <sup>[[8]](#ref-8)</sup> | Young, sparse docs | JAX lock-in, complex for GPU | NVIDIA only (Blackwell+); very early | Depends on Triton backend | NVIDIA only, not for general use <sup>[[8]](#ref-8)</sup> | Closed parts, no TPU/Ascend |

---

## Key Observations

### 1. Trend: Tile-based abstractions dominate
TileLang, cuTile, Helion, Pallas — all use tile as the core primitive. Natural fit for accelerator memory hierarchies (HBM → L2 → SRAM/shared → registers).

### 2. TileLang — only third-party DSL with Ascend support
**TileLang-Ascend** <sup>[[11]](#ref-11)</sup> (released Sep 2025) — only Python DSL with Ascend NPU backend (A2/A3). Generates AscendC code, requires CANN ≥8.2. Key competitor/reference for our DSL.

### 3. Triton-Ascend — Huawei's official fork
**Triton-Ascend** <sup>[[12]](#ref-12)</sup> — Huawei's own port of Triton to Ascend. Active development at gitcode.com/Ascend/triton-ascend.

### 4. Performance ceiling
- **Gluon** shows that exceeding Triton requires **warp/wave-level control** <sup>[[8]](#ref-8)</sup>
- **cuTile** demonstrates that **compiler automation** can deliver near-native with minimal code <sup>[[4]](#ref-4)</sup>
- **Helion** proves that **PyTorch-level abstraction + autotuning** can beat hand-written Triton (1.85x on H100 GEMM) <sup>[[5]](#ref-5)</sup>

### 5. For 90% peak on Ascend:
- Explicit **Cube Unit** (matrix) vs **Vector Unit** (elementwise) vs **Scalar Unit** management
- Control over **double buffering** and **pipeline** between units
- Awareness of **AI Core** topology (multiple cores, L0/L1/L2/HBM memory hierarchy)

### 6. Syntax patterns worth borrowing
| Pattern | Source | Why it matters |
|---------|--------|----------------|
| `tile` loop as core primitive | Helion <sup>[[5]](#ref-5)</sup>, TileLang <sup>[[2]](#ref-2)</sup> | Intuitive, minimal code |
| Automatic scheduling + autotuning | Helion <sup>[[5]](#ref-5)</sup>, cuTile <sup>[[4]](#ref-4)</sup> | Key to simplicity |
| Explicit memory hints | Pallas <sup>[[3]](#ref-3)</sup>, Gluon <sup>[[6]](#ref-6)</sup> | Required for 90% peak |
| PyTorch-compatible syntax | Helion <sup>[[5]](#ref-5)</sup> | Low barrier to entry |
| Warp/wave-level escape hatch | Gluon <sup>[[6]](#ref-6)</sup> | For edge cases when auto isn't enough |

---

## References

| # | Source | Notes |
|---|--------|-------|
| <a name="ref-1"></a>[1] | [triton-lang/triton](https://github.com/triton-lang/triton) | Main repo; includes tutorials, meetup notes, MLIR pipeline docs |
| <a name="ref-2"></a>[2] | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | GPU-side TileLang; TileLang-Ascend is a separate adapter repo |
| <a name="ref-3"></a>[3] | [JAX Pallas docs](https://jax.readthedocs.io/en/latest/pallas/index.html) | Experimental API; TPU backend via Mosaic, GPU backend via Triton |
| <a name="ref-4"></a>[4] | [NVIDIA cuTile blog](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/) · [docs](https://docs.nvidia.com/cuda/cutile-python/) | Released Dec 2025 with CUDA 13.1; Blackwell-only initially (sm_100/sm_120) |
| <a name="ref-5"></a>[5] | [Helion blog (Meta/PyTorch)](https://pytorch.org/blog/helion) | Beta Oct 2025; all perf numbers from official blog post |
| <a name="ref-6"></a>[6] | [Gluon tutorials in triton repo](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon) | OpenAI-internal tool; not accepting external contributions yet (as of Jul 2025) |
| <a name="ref-7"></a>[7] | [Modular/Mojo](https://www.modular.com/mojo) · [portability paper](https://arxiv.org/abs/2509.21039) | Perf numbers from independent portability benchmark paper (H100 + AMD MI300A) |
| <a name="ref-8"></a>[8] | [Triton meetup notes, Jul 9 2025](https://github.com/triton-lang/triton/blob/main/docs/meetups/07-09-2025/notes.md) | Jeff Niu (OpenAI): "out-of-the-box only ~80% peak"; Gluon FMHA on B200 still slower than cuDNN |
| <a name="ref-9"></a>[9] | [Triton meetup notes, Mar 12 2025](https://github.com/triton-lang/triton/blob/main/docs/meetups/03-12-2025/notes.md) | Meta WS case study: Flash Attention without WS = 45%, with WS = 69% compute throughput on H100 |
| <a name="ref-10"></a>[10] | [Mojo portability paper](https://arxiv.org/abs/2509.21039) | Independent benchmark: ~87% CUDA parity compute-bound, ~100% mem-bound on H100/MI300A |
| <a name="ref-11"></a>[11] | [tile-ai/tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) · [examples](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples) | Active development; kernel list sourced directly from examples/ directory in repo |
| <a name="ref-12"></a>[12] | [Triton-Ascend primary repo](https://gitcode.com/Ascend/triton-ascend) · [mirror](https://github.com/Ascend/triton-ascend) | gitcode.com = primary active development; github.com = mirror |
| <a name="ref-13"></a>[13] | [FlashAttention repo (Dao-AILab)](https://github.com/Dao-AILab/flash-attention) | FA2/FA3 written in Triton/CUDA; used by LLaMA, Mistral, DeepSeek, vLLM |
| <a name="ref-14"></a>[14] | [TileLang DeepSeek MLA example](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md) | ~80 LOC TileLang achieving FlashMLA-level perf on H100 |
| <a name="ref-15"></a>[15] | [JAX AI Stack production blog](https://developers.googleblog.com/building-production-ai-on-google-cloud-tpus-with-jax/) | Confirms Pallas used by Anthropic, xAI, Apple on Google Cloud TPUs |
| <a name="ref-16"></a>[16] | [Liger Kernel (LinkedIn)](https://github.com/linkedin/Liger-Kernel) | Triton kernels for LLM training; supports LLaMA, Mistral, Gemma, Qwen2-VL, Phi via HF/Axolotl/TRL |

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
