# AI Accelerator Landscape: Beyond GPU

*Last updated: 2026-03-30*

> **Context:** This document surveys the broader AI accelerator landscape relevant to our DSL research.
> The key question: which chips are gaining traction, what are their programming models,
> and what does this mean for designing a DSL that could eventually target more than just Ascend?

---

## Why This Matters for Our DSL

Current analysis (see `ascend-dsl-comparison.md`) shows that almost all existing Python DSLs
(Triton, TileLang, Helion, cuTile, Gluon, Mojo) are **GPU-first** and have poor or no support
for ASIC-class accelerators. Huawei Ascend is an outlier — Triton-Ascend and TileLang-Ascend
exist precisely because it needed custom backends.

If we design our DSL with a clean MLIR-based pipeline, targeting other ASICs in the future
becomes a matter of adding a new backend, not redesigning the language.

---

## Accelerator Overview

### 1. NVIDIA CUDA (GPU) — King of the Hill

| | |
|--|--|
| **Architecture** | SIMT GPU (Streaming Multiprocessor) |
| **Programming model** | CUDA C++, PTX; Triton, cuTile, Gluon, Helion, Mojo |
| **DSL support** | Best-in-class: Triton, cuTile, Helion, Gluon, Mojo all primary target |
| **Relevance** | Reference benchmark for all perf comparisons |
| **Key link** | https://developer.nvidia.com/cuda |

### 2. Huawei Ascend — China's Contender

| | |
|--|--|
| **Architecture** | ASIC NPU with separate Cube (matrix), Vector, Scalar units |
| **Programming model** | AscendC (C++-like), CANN runtime, MindSpore framework |
| **DSL support** | Triton-Ascend (Huawei fork), TileLang-Ascend, our target |
| **Status** | Growing fast in China; primary motivation for this project |
| **Key links** | https://gitcode.com/Ascend/triton-ascend · https://www.hiascend.com |

### 3. Google TPU — Years of Production Hardening

| | |
|--|--|
| **Architecture** | Systolic array ASIC; latest TPU v5e (2024), v6 (Trillium, 2025) |
| **Programming model** | JAX/XLA (primary), PyTorch/XLA; Pallas for custom kernels |
| **DSL support** | **Pallas** (JAX extension) — only major Python DSL with native ASIC support beyond GPU |
| **Status** | Production for Google (Gemini), Anthropic, xAI, Apple via Google Cloud |
| **Note** | XLA compiler does the heavy lifting; Pallas adds kernel-level escape hatch |
| **Key links** | https://jax.readthedocs.io/en/latest/pallas/ · https://cloud.google.com/tpu |

### 4. Meta MTIA v2 — Ranking & Recommendation Focus

| | |
|--|--|
| **Architecture** | Custom ASIC for inference (ranking, recommendation, ads) |
| **Programming model** | PyTorch 2.0 native; **Triton-MTIA** backend (Meta-developed) |
| **DSL support** | Triton (via triton-mtia backend); TorchDynamo/TorchInductor integration |
| **Status** | MTIA v2 announced April 2024; production at Meta for ranking systems |
| **Note** | Meta also built TritorX — agentic AI system that auto-generates Triton kernels for MTIA |
| **Key links** | https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/ |

### 5. Cerebras WSE — Wafer-Scale Integration

| | |
|--|--|
| **Architecture** | Wafer-Scale Engine (WSE); ~900K processing elements on a single wafer |
| **Programming model** | **CSL** (Cerebras Software Language) — C-like DSL with dataflow/wavelet model |
| **DSL support** | Proprietary CSL; no Triton/TileLang/Pallas support; isolated ecosystem |
| **Status** | WSE-3 in production (2024); niche HPC + LLM training use cases |
| **Note** | Completely different model: no concept of "tiles" or "blocks" — pure dataflow |
| **Key links** | https://sdk.cerebras.net |

### 6. Groq LPU — Deterministic Inference

| | |
|--|--|
| **Architecture** | LPU (Language Processing Unit); compiler-scheduled SIMD, no dynamic dispatch |
| **Programming model** | Compiler-first: static scheduling of all ops; no public DSL/SDK for custom kernels |
| **DSL support** | No public custom kernel API; workloads via ONNX/Groq API |
| **Status** | GroqCloud in production; acquired by NVIDIA (2025) |
| **Note** | Extreme inference latency but closed programming model |
| **Key links** | https://groq.com · https://wow.groq.com |

### 7. Intel Gaudi (Habana) — GPU Alternative for Training

| | |
|--|--|
| **Architecture** | ASIC with dedicated matrix multiply units (MME) + TPC vector processors |
| **Programming model** | SynapseAI runtime; PyTorch via **Optimum-Habana**; ONNX; TPC-C for custom kernels |
| **DSL support** | No Triton/Pallas support; own TPC-C kernel language (C-based); HF integration |
| **Status** | Gaudi 3 (2024); competitive with H100 on some LLM workloads |
| **Note** | Intel strategic product; Databricks, others running LLaMA-2-70B on Gaudi 2 |
| **Key links** | https://developer.habana.ai |

### 8. Tenstorrent Tensix — Open-Source RISC-V

| | |
|--|--|
| **Architecture** | Tensix RISC-V cores + 2D mesh NoC; tile-based (32×32 native tile size) |
| **Programming model** | **TT-Metalium** (open-source SDK): C++ device kernels, Python host; reader/compute/writer pattern |
| **DSL support** | No Triton/Pallas; TT-Metalium is their Triton equivalent; actively developed |
| **Status** | Commercial (Wormhole, Blackhole chips); backed by Jim Keller; growing ecosystem |
| **Note** | Interesting: native tile model (32×32) is conceptually close to our DSL design |
| **Key links** | https://tenstorrent.com · https://github.com/tenstorrent/tt-metal |

### 9. Amazon Inferentia / Trainium — Cloud-Optimized

| | |
|--|--|
| **Architecture** | Custom ASIC (NeuronCore); optimized for transformer inference/training |
| **Programming model** | **Neuron SDK** (NKI — Neuron Kernel Interface for custom kernels); PyTorch, JAX |
| **DSL support** | NKI (Neuron Kernel Interface) for custom kernels; Triton support **not** available |
| **Status** | Trainium2 (late 2024), Trainium3 (Dec 2025); Anthropic runs on 500K Trainium2 chips |
| **Note** | Anthropic uses Trainium2 at massive scale (Project Rainier) |
| **Key links** | https://awsdocs-neuron.readthedocs-hosted.com |

### 10. Cambricon MLU — China #2

| | |
|--|--|
| **Architecture** | MLU (Machine Learning Unit); IPU-like architecture |
| **Programming model** | **BANG C** — C-based DSL with SIMD model (`__nram__`, `__wram__` memory spaces) |
| **DSL support** | Proprietary BANG C; no Triton/Pallas; QiMeng-Xpiler (2025) auto-transpiles to BANG C |
| **Status** | Growing in China; MLU370 series in production; first profit 2024 |
| **Note** | China's #2 behind Ascend; BANG C similar in spirit to AscendC |
| **Key links** | https://github.com/Cambricon/mlu-ops |

### 11. D-Matrix Corsair — Digital In-Memory Computing

| | |
|--|--|
| **Architecture** | DIMC (Digital In-Memory Computing); weights stored in SRAM stash on-chip |
| **Programming model** | **Aviator SDK**; MLIR-based compiler; GPU-like parallel ops API |
| **DSL support** | Aviator (proprietary); MLIR backend — potentially extensible |
| **Status** | Q2 2025 broad release; targeting LLM inference (focus on token generation) |
| **Note** | **MLIR-based compiler** — closest in philosophy to our approach |
| **Key links** | https://www.d-matrix.ai |

### 12. AMD MI250X / MI300X (GPU) — Open Alternative

| | |
|--|--|
| **Architecture** | GPU (CDNA); very similar to NVIDIA but ROCm-based |
| **Programming model** | ROCm/HIP; Triton, TileLang (ROCm support); Mojo |
| **DSL support** | Full Triton support (AMD backend); TileLang AMD examples exist |
| **Status** | MI300X competitive for LLM inference; strong open-source push |
| **Key links** | https://rocm.docs.amd.com |

### 13. Intel GPU (Xe / PVC / GPU Max) — Data Center

| | |
|--|--|
| **Architecture** | Xe GPU; Intel GPU Max (PVC) for HPC/AI |
| **Programming model** | SYCL (OpenCL successor); Triton Intel backend |
| **DSL support** | Triton Intel backend (experimental); Helion (via Triton); Mojo |
| **Status** | Triton Intel backend in active development; still maturing |
| **Key links** | https://github.com/intel/intel-xpu-backend-for-triton |

### 14. Qualcomm Hexagon — Mobile/Edge AI

| | |
|--|--|
| **Architecture** | DSP with HVX (vector) and HTA (tensor) extensions |
| **Programming model** | Hexagon SDK, Qualcomm AI Stack; ONNX/TFLite for deployment |
| **DSL support** | No Triton/Pallas; proprietary Hexagon intrinsics |
| **Status** | Production in Snapdragon (mobile, edge); not data center AI |
| **Note** | Out of scope for server-side NPU DSL |

### 15. Apple Neural Engine (ANE) — Vertical Integration

| | |
|--|--|
| **Architecture** | Fixed-function NPU integrated in Apple Silicon (M/A series) |
| **Programming model** | Core ML, Metal; no public low-level programming API |
| **DSL support** | None (closed black box) |
| **Status** | Production in all Apple devices |
| **Note** | Completely closed; not relevant for our DSL |

### 16. Microsoft Azure Maia AI 200 — Hyperscaler Custom

| | |
|--|--|
| **Architecture** | Custom ASIC for Azure AI workloads (transformer inference) |
| **Programming model** | Not publicly documented; likely ONNX + custom compiler |
| **DSL support** | Unknown (closed) |
| **Status** | Announced 2023; limited public info as of 2026 |

### 17. SiFive Intelligence X390 — RISC-V Edge

| | |
|--|--|
| **Architecture** | RISC-V vector processor for edge AI |
| **Programming model** | ONNX runtime; standard RISC-V toolchain |
| **DSL support** | Standard compilers (no custom DSL) |
| **Status** | Edge/IoT focus; not relevant for server NPU DSL |

---

## DSL Support Matrix: Non-GPU ASICs

| DSL | Google TPU | Ascend | Meta MTIA | AWS Trainium | Gaudi | Tenstorrent | Cerebras |
|-----|-----------|--------|-----------|-------------|-------|-------------|---------|
| **Triton** | ❌ | ✅ Triton-Ascend | ✅ triton-mtia | ❌ | ❌ | ❌ | ❌ |
| **TileLang** | ❌ | ✅ TileLang-Ascend | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Pallas** | ✅ Native (Mosaic) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **cuTile** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Helion** | ❌ | ❌ | ✅ (via triton-mtia) | ❌ | ❌ | ❌ | ❌ |
| **Gluon** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Mojo** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Our DSL (goal)** | 🔮 possible via MLIR | ✅ primary target | 🔮 possible | 🔮 possible | 🔮 possible | 🔮 possible | ❌ (incompatible model) |

---

## Key Observations

### 1. MLIR is the common thread
D-Matrix (Aviator), our DSL, and Triton all use MLIR as IR. This is becoming the standard
for portable accelerator compilers. MLIR backends exist or are feasible for Ascend, Gaudi, MTIA.

### 2. Only Pallas and Triton cross the GPU/ASIC boundary
- **Pallas**: TPU only (via JAX/Mosaic)
- **Triton**: GPU-first, but has MTIA and Ascend backends from partners

### 3. Tile-based model fits multiple ASICs
Tenstorrent (32×32 native tiles), Google TPU (systolic = tiled), Ascend (Cube = matrix tile) —
tile abstraction is natural for ASIC memory hierarchies, not just GPUs.

### 4. China AI chip ecosystem is real
Ascend (Huawei) + Cambricon: both have C-based kernel DSLs (AscendC, BANG C) with similar
philosophy. Cambricon BANG C is essentially China's AscendC. Our DSL could target both
with different backends.

### 5. Accelerators with no viable custom kernel story
Groq, Apple ANE, Qualcomm Hexagon, Azure Maia — all closed or limited. Not viable targets.

### 6. Most promising future targets after Ascend (given MLIR pipeline)
1. **AWS Trainium** — NKI exists but no Python DSL; huge scale (Anthropic)
2. **Google TPU** — Pallas exists but complex; MLIR backend feasible
3. **Intel Gaudi** — MLIR backend from Intel in progress
4. **Tenstorrent** — open source, tile-native, RISC-V; philosophically aligned

---

## Implications for Our DSL Design

1. **Keep MLIR clean** — don't embed GPU-specific assumptions in the IR
2. **Tile abstraction is universal** — tile loop model works for Ascend, TPU, Tenstorrent
3. **Explicit memory hierarchy** — each ASIC has named memory levels (L0/L1/HBM on Ascend, NRAM/WRAM on Cambricon, SRAM stash on D-Matrix); our DSL should expose this abstractly
4. **Escape hatch per target** — hardware-specific ops should be isolated in backend, not in core DSL

---

## References

| # | Source |
|---|--------|
| Cerebras | [sdk.cerebras.net](https://sdk.cerebras.net) |
| Groq | [groq.com/blog/the-groq-lpu-explained](https://groq.com/blog/the-groq-lpu-explained) |
| Intel Gaudi | [developer.habana.ai](https://developer.habana.ai) |
| Tenstorrent | [github.com/tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) |
| Meta MTIA v2 | [ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) |
| Cambricon BANG C | [arxiv.org/html/2505.02146v1](https://arxiv.org/html/2505.02146v1) |
| D-Matrix Aviator | [d-matrix.ai/product/](https://www.d-matrix.ai/product/) |
| AWS Neuron SDK | [awsdocs-neuron.readthedocs-hosted.com](https://awsdocs-neuron.readthedocs-hosted.com) |
| Triton-MTIA | [ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) |
