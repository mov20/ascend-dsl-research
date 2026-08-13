# Python DSL Programming Models for AI Accelerators — Industry Trends & Strategic Outlook (2025–2026 H1)

*Last updated: 2026-08-05*

> Companion to [`asic-landscape.md`](asic-landscape.md) (accelerator hardware + DSL-support matrix) and [`../pyasc2-design.md`](../pyasc2-design.md) / upstream `compiler-team/pyasc@v2` design docs (per-DSL *mechanics*). This document operates one altitude up: **cross-DSL trend synthesis, adoption momentum, vendor strategy, and the AI-generated-kernel wave** — used to justify Ascend's continued investment in a native Python DSL (**PyAsc2 / `asc2`**). It does not repeat programming-model mechanics; it references them.

---

## Contents

- [1. Highlights](#1-highlights)
- [2. Industry Trends Review](#2-industry-trends-review)
  - [2.0 Framing: what a kernel DSL must deliver](#20-framing-what-a-kernel-dsl-must-deliver)
  - [2.1 What frontier SOTA models actually use](#21-what-frontier-sota-models-actually-use)
  - [2.2 Python as the universal kernel front-end](#22-python-as-the-universal-kernel-front-end)
  - [2.3 NVIDIA moves up-stack to Python DSLs](#23-nvidia-moves-up-stack-to-python-dsls)
  - [2.4 Triton as the de-facto cross-vendor standard](#24-triton-as-the-de-facto-cross-vendor-standard)
  - [2.5 Front-ends going higher-level and autotuned](#25-front-ends-going-higher-level-and-autotuned)
  - [2.6 Every non-NVIDIA datacenter vendor ships a Python DSL](#26-every-non-nvidia-datacenter-vendor-ships-a-python-dsl)
  - [2.7 Customer and production adoption](#27-customer-and-production-adoption)
  - [2.8 LLM / agentic kernel generation](#28-llm--agentic-kernel-generation)
  - [2.9 Standardization and interop pressure](#29-standardization-and-interop-pressure)
  - [2.10 Synthesis: where this lands in 1–2 years](#210-synthesis-where-this-lands-in-12-years)
- [3. Strategic Plan for Ascend (1–2 years)](#3-strategic-plan-for-ascend-12-years)
  - [3.1 Positioning & messaging — PyAsc2 vs Triton / TileLang](#31-positioning--messaging--pyasc2-vs-triton--tilelang)
  - [3.2 Strategic pillars](#32-strategic-pillars)
  - [3.3 Risks of inaction](#33-risks-of-inaction)
  - [3.4 Milestones](#34-milestones)
- [4. References](#4-references)
- [Appendix A. Open questions](#appendix-a-open-questions)

---

## 1. Highlights

<!-- Written LAST — distilled from §2 and §3. Placeholder. -->
_TODO (Stage 5): 6–8 cited highlight bullets + "why this matters for Ascend" + strategic-recommendation teaser._

---

## 2. Industry Trends Review

### 2.0 Framing: what a kernel DSL must deliver

**The strategic premise.** This document assumes Ascend becomes a mainstream AI accelerator — a platform frontier models are expected to run on, not a niche port target. That premise fixes the design goal: **a Python DSL that reaches peak *on Ascend*, rather than one that runs everywhere at a discount.** Portability — the property Triton and TileLang optimize for — is deliberately traded away. The rest of this section establishes the three axes that remain, and §2.10/§3 return to what the industry's own evidence says about that trade.

Three axes decide whether a kernel DSL is worth building:

| Axis | Question it answers | Measured by |
|------|--------------------|-------------|
| **Performance** | Does it beat third-party portable DSLs on our own hardware? | % of a hand-optimized native kernel |
| **Usability** | What does it cost a human to get there? | LOC, debuggability, tooling, error quality |
| **Extensibility** | Can it absorb what comes next? | New model architectures, megakernels, in-kernel collective communication |

Kernel programming has historically forced a choice between two bad corners — a framework/graph level that is effortless but capped at whatever operators the vendor shipped, and native intrinsics (Ascend C, CUDA C++) that define the performance ceiling but cost hundreds to thousands of lines and scarce expertise. **A kernel DSL exists to break that tradeoff.** Whether any given DSL actually does is an empirical question, and the answers below are less flattering than the marketing.

#### Performance: there is no single "% of peak" number

The frequently-quoted claim that a portable DSL delivers "~80% of peak out of the box" is not supported by measurements on current hardware. Against hand-tuned vendor kernels, Triton spans roughly **20% to 100%**, and the spread is driven by *kernel and architecture*, not by a single headline figure.

**GEMM is the good case.** On Hopper, Triton lands within ~10% of cuBLAS — a figure reported by Triton's own creator, whose GTC 2025 slide for dense FP16 8192² matmul on H100 and GB200 reads simply *"still slower!"* <sup>[[7]](#ref-7)</sup> On Blackwell that falls to roughly 62–70%.

**Attention is the bad case, and it is getting worse as hardware advances.** On H100, Triton reaches ~61% of FlashAttention-3. On B200, measured on *current* Triton 3.6 / CUDA 13.1 — so not an artifact of stale tooling: <sup>[[12]](#ref-12)</sup>

All four rows below run the **same algorithm** — flash attention — on the same hardware, at the same shape. The only variable is **what each implementation is written in**. That is what makes the comparison meaningful: it isolates the cost of the authoring technology from the cost of the algorithm.

| Implementation | Written in | TFLOP/s | vs hardware peak¹ | vs best kernel² |
|---|---|---|---|---|
| Triton 3.6 | portable tile DSL | 703 | 31% | 44% |
| **Gluon 3.6** | **low-level tile DSL, non-portable** | **1250** | **56%** | **77%** |
| cuDNN 9.19 | NVIDIA closed, hand-tuned library | 1613 | 72% | 100% |
| FlashAttention-4 | hand-written CUTLASS / CuTe C++ | 1613 | 72% | 100% |

*B200, forward, head-dim 128, non-causal, sequence length 32K.*
¹ Against B200 dense BF16 peak = 2250 TFLOP/s — the theoretical hardware ceiling.
² Against FlashAttention-4 — the fastest implementation measured, and the reference point for "what a skilled human achieves in C++." cuDNN ties it.

Both denominators are shown because they answer different questions: *vs hardware peak* is how much of the chip is left unused; *vs best kernel* is the gap to what a skilled human has actually achieved on that chip. Across sequence lengths 1K–32K, Triton spans **37–44% of FA4** and **23–31% of hardware peak**.

Triton's absolute attention throughput rose only ~1.78× from H100 to B200 while the hardware and hand-tuned kernels roughly 2.5×'d. A portable abstraction does not automatically inherit a new architecture's capability.

**Tuning closes the gap, but the starting point is low.** An independent IBM Research study of paged attention in Triton on H100 measured a naive kernel at **19.7% of FlashAttention-3**, reaching **98.6–105.9%** only after 5.9× of systematic tuning. <sup>[[13]](#ref-13)</sup> The DSL does not deliver peak; a skilled human using the DSL eventually does.

**What closes the gap is hardware control, not abstraction.** The same Flash Attention kernel moves from **45% → 69%** of H100 compute throughput once warp specialization is applied — a 24-point swing driven purely by how much hardware detail the programmer can reach. <sup>[[8]](#ref-8)</sup>

**The precedent.** Faced with this ceiling, OpenAI's response was not a better compiler but **Gluon** — a second, lower-level, deliberately non-portable language inside the Triton repository that exposes layouts, shared memory, and warp specialization directly. On the B200 attention above, Gluon reaches 1250 TFLOP/s against Triton's 703: a **1.8× gain obtained by surrendering portability.** The organization with the most to lose from fragmenting its own DSL nonetheless concluded that reaching peak requires hardware-specific control. §3 returns to this.

#### Usability: LOC is the floor, not the ceiling of the question

Lines of code is a crude but honest proxy for authoring effort. Flash Attention and GEMM are the conventional yardsticks:

> **TODO — this comparison is too narrow to generalize.** Two kernels cannot support a general claim, and these are the two every DSL optimizes hardest. A representative table needs the ordinary kernels that dominate real workloads — `layer_norm`, `softmax`, quantize/dequantize, MoE routing — and ideally pairs each with a measured performance number, so usability and performance can be read on the same axes. Kernel selection should follow the SOTA-model analysis. See [Appendix A.1](#a1-broaden-the-performance-vs-usability-comparison).


| DSL | Flash Attention fwd | GEMM |
|-----|--------------------|------|
| TileLang (GPU) | ~74 LOC <sup>[[1]](#ref-1)</sup> | ~25 |
| Helion | ~81 LOC <sup>[[2]](#ref-2)</sup> | ~15 |
| Triton | ~136 LOC <sup>[[3]](#ref-3)</sup> | ~30 |
| **TileLang (Ascend)** | **~208 LOC** <sup>[[4]](#ref-4)</sup> | — |
| Gluon (warp-specialized) | ~645 LOC <sup>[[5]](#ref-5)</sup> | ~40 |
| Pallas (TPU, production) | ~1718 LOC <sup>[[6]](#ref-6)</sup> | ~40 |

Two observations. The spread is **more than 20×** between tile DSLs — usability is the dominant differentiator, not a rounding error. And **the same DSL costs ~2.8× more code on Ascend than on GPU** (TileLang: ~208 vs ~74 LOC), because the Ascend port must express memory movement explicitly. <sup>[[4]](#ref-4)</sup> The authoring gap is measurably wider on NPU.

But LOC alone understates the problem. Real authoring cost also includes **debuggability** (can you inspect intermediate tile state, or only the final output?), **error quality** (does an oversized buffer fail at compile time with a clear message, or corrupt memory silently at runtime?), **tooling** (profilers that attribute time to source lines; autotuning that does not require hand-written search spaces), and **determinism** — which, as §2.1 shows, has become a hard production requirement rather than a nicety.

**Higher-level is not automatically slower.** Helion reports **1.85× over hand-written Triton** on H100 GEMM, because an autotuning compiler searches a space a human will not. <sup>[[9]](#ref-9)</sup> TileLang reaches FlashMLA-level performance on H100 in ~80 LOC. <sup>[[10]](#ref-10)</sup> When the compiler is strong, abstraction wins on both axes at once — fewer lines *and* more speed. That is why the industry keeps moving up-stack rather than down, and why a weak compiler, not a high abstraction level, is what costs performance.

#### Extensibility: the axis that decides the next two years

A DSL is a long-lived investment; model architectures are not. The question is whether the programming model can absorb demands that did not exist when it was designed. The 2026 frontier generation has produced four that most tile DSLs cannot express (evidenced in detail in §2.1):

- **Fused compute + collective communication.** Throughout this document, *communication* means **inter-device collectives — all-reduce, all-gather, reduce-scatter, all-to-all — across NPUs and across nodes**, the operations NCCL and HCCL provide. It does **not** mean intra-SoC data movement between Cube and Vector cores, which is a separate concern handled by on-chip synchronization. Large sparse MoE now fuses dispatch, GEMM, activation, and combine into a single pipelined megakernel, overlapping one wave of experts' compute with the next wave's *network* transfer — worth **1.50–1.73×** over non-fused baselines. <sup>[[14]](#ref-14)</sup> No mainstream tile DSL lets a kernel *contain* a collective operation.
- **Low precision as a first-class type.** FP4 weights with FP8 activations and block-wise scale factors, plus quantization fused into epilogues.
- **Heterogeneous memory layouts.** Compressed and sparse attention variants that place differently-shaped KV state in one paged pool.
- **Awkward shapes.** Novel residual topologies producing, in one real case, a GEMM with output dimension 24 — DSLs tuned for 128×128 tiles handle these badly.

**The gap is wider on Ascend.** The native tier costs more here than on GPU: Ascend C requires hand-placed synchronization barriers, hand-planned Unified Buffer layout within a hard 192–256 KB budget, and hand-structured ping-pong loops — with missing barriers causing silent data hazards and UB overflow corrupting memory at runtime. The wider the gap between the native and DSL tiers, the more a DSL that closes it is worth. PyAsc2 targets tile-level Python at **≈90% of hand-optimized Ascend C**, with synchronization, UB allocation, and pipelining automated by the compiler. <sup>[[11]](#ref-11)</sup>

> Per-DSL mechanics (how each handles sync insertion, ping-pong, and UB allocation) are analyzed in [`../pyasc2-design.md`](../pyasc2-design.md) §3 and are not repeated here. This document tracks where the industry is *moving*.

### 2.1 What frontier SOTA models actually use

Vendor announcements describe what a programming model *can* do. The kernel repositories published alongside frontier open-weights models show what teams with the strongest possible incentive to be fast actually *chose*. That evidence is more informative than any roadmap, and over the last eighteen months it moved.

#### DeepSeek migrated from Triton to TileLang, and documented why

The clearest signal is a dependency change across three consecutive releases of the same reference inference stack:

| Release | Date | Kernel dependency |
|---|---|---|
| DeepSeek-V3 | 2024-12 | `triton==3.0.0` <sup>[[15]](#ref-15)</sup> |
| DeepSeek-V3.2-Exp | 2025-09 | **`tilelang==0.1.6`** <sup>[[16]](#ref-16)</sup> |
| DeepSeek-V4-Pro | 2026-04 | **`tilelang==0.1.8`** <sup>[[17]](#ref-17)</sup> |

This was not incidental. The V4 technical report devotes a named section — §3.2, *"Flexible and Efficient Kernel Development with TileLang"* — to the decision. Across 58 pages the report mentions TileLang 13 times and Triton, PTX, and CUTLASS zero times each. <sup>[[14]](#ref-14)</sup> Its stated reasoning:

> "In practice, our elaborate model architecture would have resulted in hundreds of fine-grained Torch ATen operators. We adopt TileLang to develop **a set of fused kernels to replace the vast majority of them**, delivering optimal performance with minimal effort. It also allows us to quickly prototype operators like attention variants during validation. These kernels play critical roles in model architecture development, large-scale training, and **ultimately production deployment of inference services**."

Two details raise this above ordinary tool adoption. First, DeepSeek did not merely use the compiler — they **co-developed it**, contributing host-side code generation that cut per-invocation launch overhead from tens or hundreds of microseconds to under one, a Z3 SMT solver for layout inference and memory-hazard detection, and IEEE intrinsics enabling bitwise reproducibility against hand-written CUDA baselines. <sup>[[14]](#ref-14)</sup> Second, they shipped the resulting library: `deepseek-ai/TileKernels` contains **122 Python files and zero `.cu` files**, covering MoE gating and routing, FP8/FP4 quantization, fused SwiGLU-quant, and batched transpose. <sup>[[18]](#ref-18)</sup>

#### But the hot path stayed in hand-written C++

The migration has a sharp boundary, and mistaking its extent would be the easy error. DeepSeek runs a deliberate two-tier stack:

| Tier | Components | Language | Evidence |
|---|---|---|---|
| **Hand-written** | MLA + sparse attention (FlashMLA) | CUDA C++ / CUTLASS | 23 `.cu`, 20 `.cuh` <sup>[[19]](#ref-19)</sup> |
| | FP8/FP4 GEMM, MegaMoE (DeepGEMM) | CUDA C++ (JIT) | 42 `.cuh`, 46 `.hpp`; Python is bindings only <sup>[[20]](#ref-20)</sup> |
| | MoE all-to-all dispatch/combine (DeepEP) | CUDA C++ + inline PTX | **55 `asm volatile` sites in one header** <sup>[[21]](#ref-21)</sup> |
| **Python DSL** | MoE routing, quantization, transpose, norms, novel ops (TileKernels) | TileLang | 122 `.py`, 0 `.cu` <sup>[[18]](#ref-18)</sup> |

The split is visible even *within a single feature*: the small GEMM for the mHC residual topology lives in DeepGEMM as CUDA, while mHC's Sinkhorn normalization and mixing operations live in TileKernels as TileLang. <sup>[[18]](#ref-18)</sup> <sup>[[20]](#ref-20)</sup>

The operative rule appears to be: **Tensor-Core-bound GEMM, the flagship attention kernel, and network kernels stay hand-written; the long tail of fused element-wise, routing, quantization, and prototype operators moves to the DSL.** That long tail is large and growing, because each new architectural idea generates exactly this kind of operator.

#### Per-lab survey

| Lab / model | Kernel repositories | DSL used | Serves |
|---|---|---|---|
| **DeepSeek V4** | TileKernels; FlashMLA; DeepGEMM; DeepEP | **TileLang** (long tail) + CUDA/PTX (hot path) | Sparse attention, MoE EP, FP8/FP4 GEMM, quantization |
| **Zhipu GLM-5.2** | `THUDM/slime` <sup>[[22]](#ref-22)</sup> | **TileLang** (4 files, 823 LOC) *adapted from DeepSeek's examples* + borrowed CUDA forward | Sparse MLA + lightning indexer (training) |
| **Alibaba Qwen 3.8** | `QwenLM/FlashQLA` <sup>[[23]](#ref-23)</sup> | **TileLang only** — 41 `.py`, **0 `.cu`**, 16,149 LOC | Gated DeltaNet chunked prefill, fwd + bwd |
| **Moonshot Kimi K3** | FlashKDA; MoonEP <sup>[[24]](#ref-24)</sup> | **CUTLASS/CuTe C++** + **CuTe DSL** (Python) | KDA linear attention; MoE dispatch/combine |
| **MiniMax M3** | `MiniMax-AI/MSA` <sup>[[25]](#ref-25)</sup> | **CuTe DSL + CUDA**; no Triton, no TileLang | Block-sparse attention, FP8/FP4 paged KV |
| **Tencent Hy3** | `Tencent/hpc-ops` <sup>[[26]](#ref-26)</sup> | **CUDA/CuTe only** — 0 Triton, 0 TileLang | FP8 MoE, inference ops |
| **OpenAI gpt-oss** | in-repo `gpt_oss/triton/` <sup>[[27]](#ref-27)</sup> | **Triton** (OpenAI owns Triton — dogfooding, not selection) | MXFP4 MoE, attention with sinks |
| **Meta Llama 4** | none published | — | — |

Three patterns emerge:

**The choice splits along a China/US line.** TileLang has been adopted by the Chinese frontier labs — DeepSeek, Zhipu, and Alibaba — while CuTe DSL has been adopted where Blackwell-class control in Python is the priority (Moonshot, MiniMax). This is the most significant DSL shift of the period and is not widely reported.

**Triton has become the reference layer, not the performance layer.** Across this cohort its role is the baseline others benchmark against and retain as a portability fallback. The only lab in the table shipping Triton kernels is **OpenAI**, which created Triton — so `gpt-oss` is dogfooding rather than a competitive selection. Every lab that evaluated Triton as an outside option chose something else for its performance-critical kernels.

**The model repository is no longer a kernel artifact.** Every flagship model repository here ships zero kernels; they arrive separately, weeks to months later, under a different name. Serving is delegated to vLLM or SGLang.

#### Qwen 3.8: a frontier flagship with no C++ at all

The newest release in the set is also the most consequential for this document's argument. Alibaba's **Qwen 3.8** (2026-08-08) is a 2.4T-parameter / 95B-active model whose novel operator — Gated DeltaNet, used in 69 of 92 layers — ships as `QwenLM/FlashQLA`: **41 Python files, zero `.cu` files, 16,149 lines, one language.** <sup>[[23]](#ref-23)</sup>

This is not a high-level convenience wrapper over vendor libraries. A census of the TileLang primitives used shows hand-scheduled, architecture-specific work:

| Primitive | Uses | What it reaches |
|---|---|---|
| `T.barrier_arrive` / `T.barrier_wait` | 577 | explicit producer/consumer warp-group synchronization |
| `T.tcgen05_gemm` | 44 | Blackwell 5th-generation Tensor Cores |
| `T.tma_copy` | 43 | Tensor Memory Accelerator async copy |
| `T.set_max_nreg` | 43 | per-warp register budget control |
| `T.alloc_tmem` | 24 | Blackwell tensor memory allocation |
| `T.fence_proxy_async` | 82 | async proxy memory ordering |

These are the same hardware capabilities that Gluon was created to expose and that stock Triton withholds (§2.0) — reached here from Python, with no escape to C++. The repository even carries a commit titled *"Add fence.proxy.async to SM90 and SM100 kernels."* Alibaba's own description is explicit about why: they *"use TileLang to build several key fused kernels, and **manually implement warpgroup specialization** to overlap data movement, Tensor Core computation, and CUDA Core computation."* <sup>[[23]](#ref-23)</sup>

The reported result is **2–3× forward and 2× backward speedup over the FLA Triton kernel** across H200, GB200, RTX 5090, and RTX Pro 6000. <sup>[[23]](#ref-23)</sup>

**Why this matters more than the DeepSeek migration.** DeepSeek's two-tier stack is consistent with a reading where DSLs handle only the easy long tail. Qwen refutes that reading: a frontier lab shipped its most performance-critical novel operator, at 2.4T scale, on the newest silicon, **entirely in a Python DSL** — because the DSL exposed the same scheduling primitives CUDA does. The constraint on tile DSLs is *what they expose*, not that they are Python.

Two caveats worth recording. The architecture-specific trees (`chunk/hopper/`, `chunk/blackwell/`, `chunk/blackwell_sm120/`) are three separate implementations of the same mathematics — the DSL delivered portability of *source*, not of *performance*. And Qwen 3.8 shipped with **no technical report**, only a blog post, alongside a licence change away from Apache-2.0 — a disclosure regression relative to earlier Qwen releases.

#### Kimi K3: CuTe throughout, and a revealing Ascend story

Moonshot's **Kimi K3** (2026-07-27, 2.8T / 104B active; 69 KDA layers to 24 Gated-MLA) takes the opposite path from Qwen at every layer: <sup>[[24]](#ref-24)</sup>

| Component | Implementation | Language |
|---|---|---|
| KDA linear attention | `FlashKDA` | CUTLASS / CuTe **C++** |
| MoE dispatch / combine | `MoonEP` | CuTe **DSL** (Python) |

Moonshot's stated reason for CUTLASS is performance — FlashKDA *"substantially outperforms the Triton reference implementation"* — with Triton retained only as a fallback inherited from the community FLA project. <sup>[[24]](#ref-24)</sup> Kimi K3 also ships **no technical report on arXiv**, only a PDF in the repository.

**The Ascend port is the finding.** vLLM-Ascend support for K3 was opened on **K3's release day** and merged four days later — implying pre-release access — but it was written by Huawei and community engineers, not by Moonshot. <sup>[[46]](#ref-46)</sup> What that port required is the point:

> The KDA forward kernel alone is **~2,584 lines of hand-written Ascend C** (`chunk_kda_fwd.cpp`), plus separate hand-written kernels for gate cumulative-sum, layout swap, and MX quantization. <sup>[[46]](#ref-46)</sup>

On NVIDIA, the same operator is ~2,300 lines of CuTe C++ and its communication layer is Python. On Ascend, a new attention architecture required **thousands of lines of hand-written Ascend C within two weeks of the model's release, with no DSL in the path at all.** That is precisely the gap PyAsc2 exists to close, documented with dated file-level evidence.

Two further details sharpen it. First, K3's native **MXFP4** weights cannot be used directly on Ascend A2/A3: the community checkpoint dequantizes MXFP4 to BF16 — its own manifest states the conversion *"is lossy and does not reconstruct the original pre-quantization BF16 weights"* — then re-quantizes to INT W4A8. <sup>[[47]](#ref-47)</sup> Second, **Ascend A5 (950) is the only non-NVIDIA silicon with a native MXFP4 path** matching K3's training format. The quantization gap on Ascend is generational, not architectural — and it closes with A5.

#### Linear attention: one architectural trend, two DSL answers

The clearest natural experiment in the table is linear and hybrid attention, which the Chinese labs have pushed hardest. Two of them shipped production kernels for it within months of each other — and chose differently:

| Model | Mechanism | Kernel | Written in |
|---|---|---|---|
| **Kimi K3** | KDA (Kimi Delta Attention) — 69 KDA layers to 24 Gated-MLA layers | `FlashKDA` | **CUTLASS / CuTe C++** <sup>[[24]](#ref-24)</sup> |
| **Qwen** | Gated DeltaNet | `QwenLM/FlashQLA` | **TileLang** (41 `.py`, 0 `.cu`) <sup>[[23]](#ref-23)</sup> |

Same architectural direction, opposite tooling. Moonshot's stated reason for CUTLASS is performance — FlashKDA "substantially outperforms the Triton reference implementation" — while Qwen's stated reason for TileLang is access to **warpgroup specialization**. <sup>[[23]](#ref-23)</sup> <sup>[[24]](#ref-24)</sup> Both are arguments about control over scheduling, resolved in opposite directions.

This matters for Ascend because linear attention is where the *portable* layer has actually reached the platform: the community `fla-org/flash-linear-attention` project ships a `triton_ascend` backend family covering KDA, gated-delta-rule, and related kernels, with Ascend-specific machinery — a Unified Buffer manager and an AI-Core task-time block budget. <sup>[[32]](#ref-32)</sup> It is community work rather than a lab's own port, and its Ascend-specific extensions are further evidence that "portable Triton" does not survive contact with the NPU unmodified.

#### Why each lab chose what it chose

The stated reasons are technical and consistent, and they centre on **control over scheduling** rather than syntax or ergonomics:

- **Qwen** is the most explicit: FlashQLA uses TileLang because they "take CP and backward requirements into account… and **manually implement warpgroup specialization** to overlap data movement, Tensor Core computation, and CUDA Core computation." <sup>[[23]](#ref-23)</sup> They chose the DSL that *exposes* warp specialization — which Triton does not.
- **Moonshot** reports FlashKDA, a CUTLASS-based kernel, "substantially outperforms the Triton reference implementation," and retains Triton only as a fallback. <sup>[[24]](#ref-24)</sup>
- **Meta**, at the Triton developer conference: "Most people start with Triton… Some customers will go directly to CUTLASS/CuTe DSL. **Scheduling is usually a question that drives this choice.**" <sup>[[28]](#ref-28)</sup>

This is the same conclusion §2.0 reached from the performance data, arrived at independently by practitioners: the gap that matters is *access to the hardware's scheduling primitives*.

#### Counter-evidence

Two findings cut against a simple pro-DSL reading, and both should be carried forward.

**Nondeterminism cost a training run at Zhipu — and it indicted CUDA first.** The GLM-5 report (§3.2, "DSA RL insights") records that the **nondeterministic CUDA-based top-k implementation used in SGLang's DSA indexer** created a training/inference mismatch; "other non-deterministic top-k operators (e.g., CUDA or TileLang implementations) caused drastic performance degradation during RL after only a few steps, accompanied by a sharp drop in entropy." They reverted to `torch.topk`, which is slower but deterministic. <sup>[[29]](#ref-29)</sup>

It would be a misreading to file this against DSLs. The implementation that failed was **hand-written CUDA**; TileLang appears in an "e.g." list alongside it; the scope is one operator during RL; and Zhipu kept TileLang for sparse MLA and the indexer's own forward and backward in the same codebase. Their actual remedy was to write **more** DSL kernels — a Triton module for route-permutation gradients whose docstring states the contract directly: *"They do not use atomics, and every visible output element has exactly one writer."* <sup>[[22]](#ref-22)</sup>

The defensible conclusion is about **determinism as a contract, not about DSLs**: at RL scale, bitwise reproducibility becomes a correctness property rather than a nicety, and neither CUDA nor TileLang offered a guarantee — so Zhipu chose, operator by operator, whichever implementation it could prove deterministic. A kernel programming model that can *guarantee* determinism has an advantage neither incumbent currently provides.

**Zhipu is a consumer of TileLang, not an author of it.** All four of its GLM-5 TileLang kernels carry `Adapted from` headers pointing at DeepSeek V3.2 examples in the TileLang repository, pinned to specific upstream commits. <sup>[[22]](#ref-22)</sup> Its sparse-MLA module also defines a second path, `SGLangSparseMLA`, documented as *"SGLang FlashMLA forward with the trainable TileLang backward"* — a hand-written CUDA forward borrowed from SGLang, with TileLang supplying the backward pass that no inference library provides. GLM's adoption is therefore downstream of DeepSeek's rather than independent corroboration of it, and its own two-tier split falls on the training/inference boundary.

**Fragmentation is now measurable inside a single file tree.** vLLM's DeepSeek-V4 mHC operator ships **five backends** — CUDA, AITER, TileLang, Triton, and torch — and Qwen's Gated DeltaNet has four independent implementations across three DSLs, with the model author's own TileLang kernel not being the one vLLM serves. <sup>[[23]](#ref-23)</sup> Convergence on tiles as a *model* has not produced convergence on a single implementation.

#### What these architectures demand of a programming model

These are the concrete demands behind the extensibility axis in §2.0. Each is drawn from a shipped 2026 model, not speculation.

| Demand | Evidence | What it requires of the DSL |
|---|---|---|
| **Fused compute + communication** | V4 fuses MoE dispatch, GEMM, activation, and combine into one pipelined megakernel, overlapping one expert wave's compute with the next wave's transfer: **1.50–1.73×** over non-fused, up to **1.96×** for RL rollout <sup>[[14]](#ref-14)</sup> | A kernel must be able to *contain* a network operation and schedule against it |
| **Low precision as a type** | V4 routed-expert weights in FP4 with FP8 elsewhere and **ue8m0 scales at 128×128 blocks**; Kimi K3 uses MXFP4 weights / MXFP8 activations with QAT <sup>[[14]](#ref-14)</sup> <sup>[[24]](#ref-24)</sup> | Block scaling as a first-class type; mixed FP4×FP8 operands; quantization fused into epilogues |
| **Heterogeneous KV layouts** | V4's compressed + sparse attention reaches 1M context at **27% of the FLOPs and 10% of the KV cache** of V3.2 <sup>[[14]](#ref-14)</sup> | Differently-shaped state coexisting in one paged pool; context parallelism surviving non-contiguous partitioning |
| **Bitwise determinism** | V4 abandoned split-KV attention and split-k GEMM, replaced cuBLAS with DeepGEMM, and replaced `atomicAdd` in sparse-attention backward with per-SM buffers and a deterministic global sum — all to obtain batch-invariance <sup>[[14]](#ref-14)</sup> | Reproducibility controls, pinned reduction order, opt-in fast-math |
| **Awkward shapes** | The mHC topology requires a GEMM with **output dimension 24** <sup>[[14]](#ref-14)</sup> | Tiny and irregular tiles handled as well as 128×128 |

The first row is the important one. **No mainstream tile DSL can express a kernel that performs network communication**, which is precisely why DeepSeek's MegaMoE is raw CUDA and Moonshot's MoonEP is CuTe DSL. It is a capability gap, not a performance gap.

#### The Ascend angle

Three facts here bear directly on this document's argument.

**Demand for Ascend already exists at the frontier — and was not met by published code.** The V4 report states plainly: *"We validated the fine-grained EP scheme on **both NVIDIA GPUs and HUAWEI Ascend NPUs platforms**."* <sup>[[14]](#ref-14)</sup> This is the report's only mention of Ascend. The corresponding open-sourced MegaMoE implementation is CUDA-only; the Ascend implementation was not released.

**Ascend support is real, fast, and written by other people.** vLLM-Ascend ships tutorials for DeepSeek-V4-Pro and V4-Flash, GLM-5.2, Kimi K3, and MiniMax on Atlas A2/A3, and MindSpeed-LLM added GLM-5.2 pretraining scripts two days after that model's release. <sup>[[30]](#ref-30)</sup> <sup>[[31]](#ref-31)</sup> GLM-5.2's full 1M context **is** validated on Ascend — on Atlas 800 A3 with decode context parallelism (`DCP16`); the caveat is hardware tier and feature maturity (the A2 series is not validated at 1M, and DCP with Sparse Flash Attention C8 is marked experimental), not a context ceiling. <sup>[[30]](#ref-30)</sup>

The consistent pattern is that **model labs do not write the Ascend kernels — Huawei and the community do**, quickly and by hand. Alibaba declined outright: asked to support Ascend in FlashQLA, the maintainer replied that they *"do not have sufficient bandwidth in the short term to develop kernels for Ampere, Ada, Ascend NPU, or PPU."* <sup>[[23]](#ref-23)</sup> In the same window, FlashQLA added Blackwell SM103 support in eight days and SM121 in under twenty-four hours. The revealed ordering places every NVIDIA SKU, down to a desktop workstation part, ahead of any NPU.

**The Ascend Triton path is not portable Triton.** vLLM-Ascend's Triton kernels import `triton.language.extra.cann.extension` — Ascend-specific intrinsics, not portable Triton source. <sup>[[30]](#ref-30)</sup> Portability across the GPU/NPU boundary is being asserted more often than it is achieved.

> **Claim to avoid.** Reporting that GLM-5/5.2 was *trained* on ~100k Ascend 910B with MindSpore is unsubstantiated: the GLM-5 report never states its training hardware, does not mention MindSpore, and benchmarks its architecture against an H800 roofline. That these models *run* on Ascend is confirmed; that they were *trained* on it is not. <sup>[[29]](#ref-29)</sup>

### 2.2 Python as the universal kernel front-end

_TODO (Stage 1)._

### 2.3 NVIDIA moves up-stack to Python DSLs

_TODO (Stage 1)._

### 2.4 Triton as the de-facto cross-vendor standard

_TODO (Stage 1)._

### 2.5 Front-ends going higher-level and autotuned

_TODO (Stage 2)._

### 2.6 Every non-NVIDIA datacenter vendor ships a Python DSL

_TODO (Stage 2)._

### 2.7 Customer and production adoption

_TODO (Stage 3)._

### 2.8 LLM / agentic kernel generation

_TODO (Stage 3)._

### 2.9 Standardization and interop pressure

_TODO (Stage 3)._

### 2.10 Synthesis: where this lands in 1–2 years

_TODO (Stage 3)._

---

## 3. Strategic Plan for Ascend (1–2 years)

### 3.1 Positioning & messaging — PyAsc2 vs Triton / TileLang

_TODO (Stage 4): portability-vs-peak thesis, candidate one-line messages, positioning table._

### 3.2 Strategic pillars

_TODO (Stage 4)._

### 3.3 Risks of inaction

_TODO (Stage 4)._

### 3.4 Milestones

_TODO (Stage 4)._

---

## 4. References

| # | Description | URL |
|---|-------------|-----|
| <a name="ref-1"></a>[1] | TileLang GPU Flash Attention fwd kernel — ~74 LOC (lines 23–96) | https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_fwd_bshd.py |
| <a name="ref-2"></a>[2] | Helion Flash Attention example — ~81 LOC (lines 35–115); Paged Attention 133L vs 295L in Triton | https://github.com/pytorch/helion/blob/main/examples/attention.py · https://pytorch.org/blog/portable-paged-attention-in-helion/ |
| <a name="ref-3"></a>[3] | Triton fused-attention tutorial — fwd ~136 LOC (65L inner helper + 71L kernel) | https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py |
| <a name="ref-4"></a>[4] | TileLang-**Ascend** Flash Attention fwd — ~208 LOC (lines 8–215); explicit memory management vs ~74 LOC GPU variant | https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/flash_attention/flash_attn_bhsd.py |
| <a name="ref-5"></a>[5] | Gluon warp-specialized attention tutorial — 645 LOC | https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py |
| <a name="ref-6"></a>[6] | JAX Pallas TPU Flash Attention — 1718 LOC (production-grade, highly parameterized) | https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py |
| <a name="ref-7"></a>[7] | Phil Tillet (Triton creator), GTC 2025 session S72876 — dense FP16 8192² matmul ~10% behind cuBLAS 12.8 on H100 and GB200; slide text "still slower!" | https://www.nvidia.com/en-us/on-demand/session/gtc25-s72876/ |
| <a name="ref-8"></a>[8] | Triton community meetup notes, 2025-03-12 — Meta warp-specialization case study: Flash Attention 45% → 69% compute throughput on H100 | https://github.com/triton-lang/triton/blob/main/docs/meetups/03-12-2025/notes.md |
| <a name="ref-9"></a>[9] | Helion announcement (Meta / PyTorch), 2025-10-22 — 1.85× vs hand-written Triton on H100 GEMM (vendor-reported) | https://pytorch.org/blog/helion |
| <a name="ref-10"></a>[10] | TileLang DeepSeek MLA example — ~80 LOC matching FlashMLA-level performance on H100 (project-reported) | https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md |
| <a name="ref-11"></a>[11] | PyAsc2 design overview — goal: ≈90% of hand-optimized Ascend C; automate sync insertion, UB allocation, ping-pong | https://gitcode.com/compiler-team/pyasc/blob/v2/docs/design/design-overview.md |
| <a name="ref-12"></a>[12] | FlashAttention-4 (MLSys 2026), arXiv 2603.05451 — B200 fwd hd128 non-causal TFLOP/s vs Triton 3.6 and Gluon 3.6, measured on CUDA 13.1 / PyTorch 2.10 | https://arxiv.org/abs/2603.05451 |
| <a name="ref-13"></a>[13] | "The Anatomy of a Triton Attention Kernel," IBM Research Zurich, 2025-10-07, arXiv 2511.11581 — independent; H100 80GB paged attention: naive 19.7% of FA3 → 98.6–105.9% after systematic tuning | https://arxiv.org/abs/2511.11581 |
| <a name="ref-14"></a>[14] | DeepSeek-V4 technical report, arXiv 2606.19348 — §3.1 fused MoE megakernel (1.50–1.73×, up to 1.96× RL rollout), §3.2 TileLang adoption and compiler contributions, §3.3 bitwise batch-invariance, FP4/ue8m0 quantization, Ascend EP validation. All performance figures self-reported | https://arxiv.org/abs/2606.19348 |
| <a name="ref-15"></a>[15] | DeepSeek-V3 reference inference `requirements.txt` (2024-12) — `triton==3.0.0` | https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/requirements.txt |
| <a name="ref-16"></a>[16] | DeepSeek-V3.2-Exp `inference/kernel.py` (2025-09) — `import tilelang.language as T`; `tilelang==0.1.6` | https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py |
| <a name="ref-17"></a>[17] | DeepSeek-V4-Pro reference inference `requirements.txt` (2026-04) — `tilelang==0.1.8` | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/requirements.txt |
| <a name="ref-18"></a>[18] | `deepseek-ai/TileKernels` — TileLang kernel library, MIT, created 2026-04-22; 122 `.py` / 0 `.cu`; MoE routing, FP8/FP4 quantization, fused SwiGLU-quant, mHC Sinkhorn ops | https://github.com/deepseek-ai/TileKernels |
| <a name="ref-19"></a>[19] | `deepseek-ai/FlashMLA` — MLA and sparse attention in CUDA C++/CUTLASS (23 `.cu`, 20 `.cuh`) | https://github.com/deepseek-ai/FlashMLA |
| <a name="ref-20"></a>[20] | `deepseek-ai/DeepGEMM` — FP8/FP4/BF16 GEMM, MegaMoE, mHC prenorm GEMM in CUDA C++ (42 `.cuh`, 46 `.hpp`); Python files are bindings | https://github.com/deepseek-ai/DeepGEMM |
| <a name="ref-21"></a>[21] | `deepseek-ai/DeepEP` — MoE all-to-all dispatch/combine; CUDA C++ with inline PTX (55 `asm volatile` sites in `csrc/kernels/legacy/utils.cuh`) | https://github.com/deepseek-ai/DeepEP |
| <a name="ref-22"></a>[22] | `THUDM/slime` — GLM-5/5.2 sparse-MLA and lightning-indexer training kernels in TileLang (4 files, ~823 LOC fwd+bwd); note the repo is under THUDM, not zai-org | https://github.com/THUDM/slime/tree/main/slime_plugins/models/glm5/ops |
| <a name="ref-23"></a>[23] | `QwenLM/FlashQLA` — Gated DeltaNet kernels in TileLang (41 `.py`, 0 `.cu`); README states manual warpgroup specialization; issue #14 (2026-06-30) declines Ascend support | https://github.com/QwenLM/FlashQLA |
| <a name="ref-24"></a>[24] | Moonshot AI Kimi K3 — repo and `k3_tech_report.pdf` (2026-07-27; no arXiv ID). FlashKDA in CUTLASS/CuTe "substantially outperforms the Triton reference"; MoonEP in CuTe DSL; `minitriton` tile DSL on MLIR | https://github.com/MoonshotAI/Kimi-K3 |
| <a name="ref-25"></a>[25] | `MiniMax-AI/MSA` (2026-06-11) — block-sparse attention, FP8/FP4 paged KV in CuTe DSL + CUDA; no Triton, no TileLang | https://github.com/MiniMax-AI/MSA |
| <a name="ref-26"></a>[26] | `Tencent/hpc-ops` — FP8 MoE and inference ops in CUDA/CuTe/CUTLASS; 0 Triton, 0 TileLang; benchmarks against vLLM Triton as baseline | https://github.com/Tencent/hpc-ops |
| <a name="ref-27"></a>[27] | OpenAI `gpt-oss` — in-repo Triton kernels for MXFP4 MoE and attention with sinks | https://github.com/openai/gpt-oss |
| <a name="ref-28"></a>[28] | Triton Developer Conference, Meta session (2025-11-05) — "Scheduling is usually a question that drives this choice" (Triton vs CUTLASS/CuTe DSL) | https://github.com/triton-lang/triton/tree/main/docs/meetups |
| <a name="ref-29"></a>[29] | GLM-5 technical report, arXiv 2602.15763 — §3.2: nondeterministic top-k in "CUDA or TileLang" implementations caused "drastic performance degradation during RL"; reverted to `torch.topk`. Report does not state training hardware and does not mention MindSpore | https://arxiv.org/abs/2602.15763 |
| <a name="ref-30"></a>[30] | `vllm-project/vllm-ascend` — tutorials for DeepSeek-V4-Pro/Flash, GLM-5.2, Kimi-K2.6, MiniMax on Atlas A2/A3; pins `triton-ascend==3.2.1`; imports `triton.language.extra.cann.extension` (Ascend-specific intrinsics) | https://github.com/vllm-project/vllm-ascend |
| <a name="ref-31"></a>[31] | `Ascend/MindSpeed-LLM` — GLM-5.2 pretraining scripts added 2026-06-18, two days after model release | https://gitee.com/ascend/MindSpeed-LLM |
| <a name="ref-46"></a>[46] | vLLM-Ascend Kimi K3 enablement — PR #12950 opened 2026-07-27 (K3 release day), merged 2026-07-31; adds `csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp` at ~2,584 lines of hand-written Ascend C, plus KDA gate-cumsum, layout-swap and MX quant kernels. SGLang NPU path merged 2026-08-12 (PR #33465) | https://github.com/vllm-project/vllm-ascend |
| <a name="ref-47"></a>[47] | `Eco-Tech/Kimi-K3-w4a8` — community (not official Huawei) Ascend checkpoint; `k3_bf16_conversion_manifest.json` states "MXFP4 dequantization is lossy and does not reconstruct the original pre-quantization BF16 weights"; 247,296 converted tensors | https://huggingface.co/Eco-Tech/Kimi-K3-w4a8 |
| <a name="ref-32"></a>[32] | `fla-org/flash-linear-attention` — community linear-attention kernel library with a `triton_ascend` backend family (KDA, gated-delta-rule); Ascend-specific machinery incl. `ascend_ub_manager.py` and an AI-Core task-time block budget; CI on CANN 9.0.0 | https://github.com/fla-org/flash-linear-attention |

---

## Appendix A. Open questions

Items deliberately left unresolved, to be closed before this document is final.

### A.1 Broaden the performance-vs-usability comparison

The LOC table in §2.0 covers only Flash Attention and GEMM — the two kernels every DSL optimizes hardest, and therefore the least representative. A general claim about usability needs the ordinary kernels that dominate real workloads:

| Kernel | Why it belongs |
|---|---|
| `layer_norm` / `rms_norm` | Ubiquitous; reduction + elementwise; tests whether cross-lane reductions are expressed cleanly |
| `softmax` | Numerically delicate (max-subtraction, online algorithms); a fair test of expressiveness |
| quantize / dequantize | Performance-critical at FP8/FP4; block-scaled formats stress the type system |
| MoE routing / gather-scatter | Irregular access patterns — the long tail a DSL is supposed to absorb |

Two open decisions: **(a)** whether to pair each kernel with a measured performance number so usability and performance sit on the same axes — more useful, but requires a benchmark run rather than source inspection; **(b)** which implementation counts as the reference, given that a single operator now ships with several backends across different DSLs — §2.1 found vLLM's mHC operator shipping five.

§2.1 also suggests the set should extend beyond textbook kernels to the operators frontier models actually generate: **MoE gating/routing**, **block-scaled FP8/FP4 quantize-dequantize**, and **fused quantize-plus-activation** — the categories that make up DeepSeek's TileKernels library, and therefore the clearest evidence of what the long tail now consists of.

### A.2 Dynamic shapes

**Unresolved — no position taken yet.** Static tile shapes are what make aggressive compile-time scheduling possible; PyAsc2 requires tile shapes known at JIT time, though tensor shapes may be runtime values. Real serving workloads are dynamic: variable sequence length, variable batch, ragged and paged attention, per-token MoE expert loads.

The open question is where the boundary belongs:

| Option | Cost |
|---|---|
| Recompile per shape bucket | Simple; risks JIT thrash and cache pressure at serving time |
| Pad to fixed tiles | Wastes compute at low occupancy |
| Symbolic tile dimensions in the IR | Most general; most expensive to build, and may forfeit the scheduling advantages static shapes provide |

What competing DSLs do, and what the 2026 model architectures actually require, should be established in the SOTA-model and cross-cutting-trends sections before this is decided.
