# AI Accelerator Landscape: Beyond GPU

*Last updated: 2026-03-30*

> **Goal:** Evaluate existing AI accelerator chips and their programming models.
> Understanding the landscape of available hardware and native programming approaches
> informs design decisions for the Ascend DSL project.

---

---

## Accelerator Overview

### 1. NVIDIA CUDA (GPU) — King of the Hill

| | |
|--|--|
| **Architecture** | SIMT GPU (Streaming Multiprocessor) |
| **Programming model** | CUDA C++, PTX |
| **DSL support** | Best-in-class: Triton, cuTile, Helion, Gluon, Mojo all primary target |
| **Relevance** | Reference benchmark for all perf comparisons |
| **Key link** | https://developer.nvidia.com/cuda |

---

### 2. Huawei Ascend — China's AI Leader

| | |
|--|--|
| **Architecture** | ASIC NPU with separate Cube (matrix), Vector, Scalar units; multi-AI-core topology |
| **Programming model** | **AscendC** (C++-like DSL with explicit CopyIn→Compute→CopyOut pipeline) |
| **Memory hierarchy** | L0A/L0B/L0C (matrix buffers), L1 (input), UB (unified buffer), GM (global memory/HBM) |
| **DSL support** | ✅ Triton-Ascend (Huawei fork) · ✅ TileLang-Ascend (tile-ai) |
| **Production** | Deployed in Huawei data centers, China cloud providers (Alibaba Cloud, Baidu Cloud) |
| **Key links** | https://gitcode.com/Ascend/triton-ascend · https://www.hiascend.com |

---

### 3. Google TPU — Years of Production Hardening

| | |
|--|--|
| **Architecture** | Systolic array ASIC; TPU v4 (2023), v5e (2024), v6/Trillium (2025) |
| **Programming model** | **JAX/XLA** (primary); **Pallas** for custom kernels (Mosaic compiler) |
| **DSL support** | ✅ Pallas (JAX extension) — only major Python DSL with native non-GPU ASIC support |
| **Production** | Gemini (Google), Anthropic, xAI, Apple — all on Google Cloud TPUs |
| **Note** | XLA compiler handles most optimization; Pallas adds kernel-level escape hatch |
| **Key links** | https://jax.readthedocs.io/en/latest/pallas/ · https://cloud.google.com/tpu |

---

### 4. Meta MTIA v2 — AI Inference at Social Scale

| | |
|--|--|
| **Architecture** | Custom ASIC for inference (ranking, recommendation, ads); MIMD with SIMD tensor units |
| **Programming model** | **Triton-MTIA** (Meta-developed Triton backend) + PyTorch 2.0 / TorchDynamo |
| **DSL support** | ✅ Triton (via triton-mtia backend); Helion (via Triton) |
| **Production** | MTIA v2 in production at Meta for ranking/recommendation systems (announced Apr 2024) |
| **Note** | Meta built **TritorX** — agentic AI system that auto-generates Triton kernels for MTIA |
| **Key links** | https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/ |

---

### 5. Cerebras WSE — Wafer-Scale Compute

| | |
|--|--|
| **Architecture** | WSE-3: ~900K processing elements on a single wafer; 2D mesh dataflow |
| **Programming model** | **CSL** (Cerebras Software Language) — C-like DSL with wavelet/dataflow model |
| **DSL support** | ❌ None of our 7 DSLs — completely isolated ecosystem |
| **Production** | WSE-3 in production (2024); HPC + LLM training niche; Andromeda cluster |
| **Note** | Fundamentally incompatible model: no tile/block concept — pure dataflow on 2D PE mesh |
| **Key links** | https://sdk.cerebras.net |

---

### 6. Groq LPU — Deterministic Inference

| | |
|--|--|
| **Architecture** | LPU (Language Processing Unit); fully compiler-scheduled SIMD, no dynamic dispatch |
| **Programming model** | Compiler-first: static scheduling; no public custom kernel API; workloads via ONNX |
| **DSL support** | ❌ No custom kernel API; closed programming model |
| **Production** | GroqCloud in production; acquired by NVIDIA (2025) |
| **Note** | Extreme inference latency but no extensibility for custom kernels |
| **Key links** | https://groq.com |

---

### 7. Intel Gaudi 3 (Habana) — GPU Alternative for Training

| | |
|--|--|
| **Architecture** | ASIC: dedicated MME (Matrix Multiplication Engines) + TPC (Tensor Processor Cores, VLIW) |
| **Programming model** | **TPC-C** (C-like kernel language for TPC units); PyTorch via **Optimum-Habana**; ONNX |
| **DSL support** | ❌ No Triton/Pallas support; own TPC-C kernel language |
| **Production** | Gaudi 3 (2024); LLaMA-2-70B training at competitive H100 speed (Databricks) |
| **Note** | Intel MLIR-based compiler work in progress; potential future backend for our DSL |
| **Key links** | https://developer.habana.ai · https://github.com/huggingface/optimum-habana |

---

### 8. Tenstorrent Tensix — Open-Source RISC-V AI

| | |
|--|--|
| **Architecture** | Tensix RISC-V cores + 2D mesh NoC; **native 32×32 tile compute** |
| **Programming model** | **TT-Metalium** (open-source): C++ device kernels (reader/compute/writer pattern), Python host |
| **DSL support** | ❌ No Triton/Pallas; TT-Metalium is their own Triton equivalent |
| **Production** | Commercial (Wormhole, Blackhole); backed by Jim Keller; growing ecosystem 2024–2025 |
| **Note** | Philosophically closest to our DSL: native tile model, MLIR potential, fully open-source |
| **Key links** | https://tenstorrent.com · https://github.com/tenstorrent/tt-metal |

---

### 9. Amazon Inferentia2 / Trainium2 — Cloud at Massive Scale

| | |
|--|--|
| **Architecture** | Custom ASIC (NeuronCore v2); matrix multiply + vector + SRAM hierarchy |
| **Programming model** | **Neuron SDK** + **NKI** (Neuron Kernel Interface — custom kernel API); PyTorch, JAX |
| **DSL support** | ❌ No Triton; NKI is their own custom kernel interface |
| **Production** | Trainium2 (late 2024), Trainium3 (Dec 2025); Anthropic Project Rainier = 500K Trainium2 chips |
| **Note** | Huge strategic importance — Anthropic committed to AWS at massive scale |
| **Key links** | https://awsdocs-neuron.readthedocs-hosted.com |

---

### 10. D-Matrix Corsair — Digital In-Memory Computing

| | |
|--|--|
| **Architecture** | DIMC (Digital In-Memory Computing); weights stored in on-chip SRAM stash (2GB @ 150 TB/s) |
| **Programming model** | **Aviator SDK** — MLIR-based compiler; GPU-like parallel ops; quantization pipeline |
| **DSL support** | ❌ Proprietary Aviator; but MLIR-based = potentially extensible |
| **Production** | Q2 2025 broad release; targeting LLM inference (token generation focus) |
| **Note** | MLIR-based compiler philosophy is closest to our approach among novel ASICs |
| **Key links** | https://www.d-matrix.ai |

---

### 11. AMD MI300X (GPU) — Open Alternative to NVIDIA

| | |
|--|--|
| **Architecture** | GPU (CDNA3 architecture); multi-die with HBM3; very similar to NVIDIA GPU |
| **Programming model** | **ROCm/HIP** (CUDA-compatible); same kernel model as CUDA |
| **DSL support** | ✅ Triton (AMD backend, mature) · ✅ TileLang (AMD examples exist) · Mojo, Helion |
| **Production** | MI300X competitive for LLM inference; strong open-source push; DeepSeek trained on MI300X |
| **Key links** | https://rocm.docs.amd.com |

---

### 12. Intel GPU (Xe / GPU Max PVC) — Data Center

| | |
|--|--|
| **Architecture** | Xe GPU; Intel GPU Max 1100/1550 (Ponte Vecchio) for HPC/AI |
| **Programming model** | **SYCL** (C++ abstraction over OpenCL); Triton Intel backend |
| **DSL support** | ✅ Triton Intel XPU backend (experimental) · Helion (via Triton) · Mojo |
| **Status** | Triton Intel backend in active development; still maturing vs NVIDIA/AMD |
| **Key links** | https://github.com/intel/intel-xpu-backend-for-triton |

---

### 13. Cambricon MLU — China's Dedicated AI Chip Leader

| | |
|--|--|
| **Architecture** | MLU (Machine Learning Unit); MLUarch03 chiplet design (TSMC 7nm) |
| **Latest chips** | MLU370-X8 (256 TOPS INT8, 48GB LPDDR5, 250W) · MLU590 "Siyuan" (1 PFLOPS CNNS, HBM3) |
| **Programming model** | **BANG C** — C-like DSL with SIMD model; memory spaces: `__nram__` (neuron), `__wram__` (weight), `__mlu_shared__` |
| **Key primitives** | `__bang_mlp(...)`, `__bang_conv(...)`, `taskId`/`clusterId`/`coreId` for multi-core |
| **DSL support** | ❌ No Triton/Pallas; BANG C is their AscendC equivalent; PyTorch via CATCH extension |
| **Auto-transpilation** | QiMeng-Xpiler (2025): auto-transpiles tensor programs to BANG C, 95% accuracy, 2x perf over libs, 96x productivity gain |
| **Production** | China cloud providers; first profit 2024; top #1 in Hurun China AI Top 50 (2026) |
| **Note** | BANG C philosophy mirrors AscendC — our DSL could target both with separate backends |
| **Key links** | https://github.com/Cambricon/mlu-ops · https://github.com/Cambricon/catch |

---

### 14. Moore Threads MTT — China's GPU (CUDA Alternative)

| | |
|--|--|
| **Architecture** | SIMT GPU (China-designed); MTT S80 (gaming), MTT S3000/S5000 (AI/data center) |
| **Programming model** | **MUSA** (Moore Unified Supercomputing Architecture) — explicit CUDA-compatible interface |
| **DSL support** | ❌ No Triton/Pallas natively; MUSA ≈ CUDA-like — Triton port possible in theory |
| **Production** | #2 in Hurun China AI Top 50; DeepSeek V3 inference on MTT S5000 (1000 tok/s decode) |
| **Note** | CUDA-compatible API means existing CUDA code ports with minimal changes; strong alignment momentum |
| **Key links** | https://en.mthreads.com |

---

### 15. Biren BR100 — China's High-Performance GPU

| | |
|--|--|
| **Architecture** | Multi-die GPU (2 tiles); 512 EUs (V-Cores + T-Cores); 64GB HBM2E @ 1.6 TB/s; 77B transistors |
| **Programming model** | **BIRENSUPA** SDK; SIMT model; V-Cores (FP32/FP16/INT) + T-Cores (matrix/conv) |
| **DSL support** | ❌ No Triton/Pallas; BIRENSUPA is proprietary; SIMT model similar to CUDA |
| **Production** | Targeting training/HPC; preparing IPO at $21.9B valuation (2024); US blacklisted |
| **Note** | GPU-like SIMT model means Triton port is architecturally feasible (no custom backend needed) |
| **Key links** | https://chipsandcheese.com/p/hot-chips-34-birens-br100-a-machine-learning-gpu-from-china |

---

## DSL Support Matrix

| | **NVIDIA GPU** | **AMD GPU** | **Intel GPU (Xe)** | **Google TPU** | **Huawei Ascend** | **Meta MTIA** | **AWS Trainium** | **Intel Gaudi** | **Tenstorrent** | **Cerebras** | **Cambricon** | **Moore Threads** | **Biren** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Programming model** | CUDA/PTX | ROCm/HIP | SYCL | JAX/XLA + Pallas | AscendC + CANN | Triton-MTIA + PyTorch | NKI + Neuron SDK | TPC-C + SynapseAI | TT-Metalium (RISC-V) | CSL (dataflow) | BANG C | MUSA (CUDA-like) | BIRENSUPA (SIMT) |
| **Triton** | ✅ Native | ✅ AMD backend | ✅ Intel XPU (exp.) | ❌ | ✅ Triton-Ascend | ✅ triton-mtia | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ (feasible) | ❌ (feasible) |
| **TileLang** | ✅ Native | ✅ AMD support | ❌ | ❌ | ✅ TileLang-Ascend | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Pallas** | ❌ | ❌ | ❌ | ✅ Native (Mosaic) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Helion** | ✅ (via Triton) | ✅ (via Triton) | ✅ (via Triton, exp.) | ❌ | ❌ | ✅ (via triton-mtia) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **cuTile** | ✅ Blackwell+ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Gluon** | ✅ H100/B200 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Mojo** | ✅ Native | ✅ AMD | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

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

### 4. China AI chip ecosystem is real and growing
- **Ascend** (Huawei): most mature ecosystem, two Python DSLs exist
- **Cambricon**: BANG C ≈ AscendC in spirit; China #1 AI chip company by revenue
- **Moore Threads**: CUDA-compatible MUSA; DeepSeek V3 runs on it
- **Biren**: SIMT-like; GPU model means Triton port is architecturally natural

### 5. Most promising future targets after Ascend (given our MLIR pipeline)
1. **AWS Trainium** — NKI exists but no Python DSL; Anthropic at massive scale
2. **Google TPU** — Pallas exists but complex; MLIR backend feasible
3. **Intel Gaudi** — MLIR compiler work in progress
4. **Tenstorrent** — open source, tile-native, RISC-V; philosophically aligned
5. **Cambricon** — BANG C target similar to AscendC

---

## Implications for Our DSL Design

1. **Keep MLIR clean** — don't embed GPU-specific assumptions in the IR
2. **Tile abstraction is universal** — tile loop model works for Ascend, TPU, Tenstorrent
3. **Explicit memory hierarchy** — each ASIC has named memory levels; our DSL should expose this abstractly
4. **Escape hatch per target** — hardware-specific ops should be isolated in backend, not in core DSL
5. **BANG C is our "second target" candidate** — Cambricon's model is close enough that a backend is feasible

---

## References

| Source | Link |
|--------|------|
| Triton-Ascend | https://gitcode.com/Ascend/triton-ascend |
| TileLang-Ascend | https://github.com/tile-ai/tilelang-ascend |
| Google Pallas / JAX | https://jax.readthedocs.io/en/latest/pallas/ |
| Meta MTIA v2 | https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/ |
| Cerebras SDK | https://sdk.cerebras.net |
| Intel Gaudi | https://developer.habana.ai |
| Tenstorrent TT-Metalium | https://github.com/tenstorrent/tt-metal |
| AWS Neuron SDK | https://awsdocs-neuron.readthedocs-hosted.com |
| D-Matrix Aviator | https://www.d-matrix.ai/product/ |
| Cambricon mlu-ops | https://github.com/Cambricon/mlu-ops |
| Cambricon CATCH (PyTorch) | https://github.com/Cambricon/catch |
| QiMeng-Xpiler (BANG C auto-transpile) | https://arxiv.org/html/2505.02146v1 |
| Moore Threads MUSA | https://en.mthreads.com |
| Biren BR100 (Chips&Cheese) | https://chipsandcheese.com/p/hot-chips-34-birens-br100-a-machine-learning-gpu-from-china |
