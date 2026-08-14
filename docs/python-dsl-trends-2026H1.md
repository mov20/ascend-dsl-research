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

> KDA support on Ascend is **written in Triton, not Ascend C**: `vllm_ascend/ops/triton/kda/` contains seven modules with sixteen `@triton.jit` kernels, and their headers attribute the code to the community flash-linear-attention project under MIT licence. <sup>[[46]](#ref-46)</sup>

This cuts against the assumption that reaching a new NPU requires hand-written native code. On NVIDIA, Moonshot wrote KDA in CuTe C++ and its communication layer in CuTe DSL. On Ascend, nobody wrote it in Ascend C at all — a **portable DSL carried a brand-new attention architecture onto the platform within days**, by way of code copied from a community project. Whatever performance that path leaves on the table (§2.0 quantifies the general case), it is the path that actually shipped, and it is evidence for Triton's role as the portability layer rather than for a native DSL.

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

Two properties are now shared by essentially every kernel language introduced since 2021: the source is **embedded in Python**, and the unit of work is a **tile** rather than a thread. This did not happen by coordination. It happened because two independent pressures — one from hardware, one from the developer population — pointed at the same design.

#### The convergence, enumerated

Every significant kernel DSL of the period, with the two properties marked:

| DSL | Author | First public | Python-embedded | Tile-based | Primary target |
|---|---|---|---|---|---|
| Triton | OpenAI | 2021 | ✅ | ✅ | NVIDIA, AMD |
| Pallas | Google / JAX | 2023 | ✅ | ✅ | TPU, GPU <sup>[[33]](#ref-33)</sup> |
| NKI | AWS | 2024-09 | ✅ | ✅ | Trainium <sup>[[30]](#ref-30)</sup> |
| TileLang | PKU + MSR | 2025-01 | ✅ | ✅ | GPU, Ascend, MetaX <sup>[[34]](#ref-34)</sup> |
| CuTe DSL | NVIDIA | 2025-06 | ✅ | ✅ (layout algebra) | NVIDIA <sup>[[35]](#ref-35)</sup> |
| Gluon | OpenAI | 2025-07 | ✅ | ✅ (explicit layouts) | NVIDIA, AMD |
| Helion | Meta | 2025-10 | ✅ | ✅ | Triton, Pallas, CuTe, TileIR, Metal <sup>[[9]](#ref-9)</sup> |
| cuTile | NVIDIA | 2025-12 | ✅ | ✅ | NVIDIA <sup>[[36]](#ref-36)</sup> |
| TT-Lang | Tenstorrent | 2026 | ✅ | ✅ | Tensix <sup>[[37]](#ref-37)</sup> |

Nine languages, four continents of origin, competing vendors — and no exceptions on either column. The pattern is strong enough that the interesting question is no longer *whether* a new accelerator gets a Python tile DSL, but how quickly.

**Two honest qualifications.** First, "tile-based" spans a real range of abstraction: Triton and cuTile hide the thread hierarchy entirely, while Gluon and CuTe DSL expose layouts, shared memory, and warp specialization — they are tile-shaped but deliberately low-level. Second, the pattern is not universal across *all* accelerator programming: Cerebras ships CSL, a C-like language with Python only as host runtime, and Groq and SambaNova expose no kernel language at all, by design. Those are covered in §2.6. The claim here is about kernel DSLs that exist, not about every vendor choosing to have one.

#### Why tiles: the hardware moved first

The tile abstraction is not a convenience invented by compiler authors. It is a description of what the hardware became. Modern accelerators execute matrix operations on dedicated units that consume and produce blocks — NVIDIA's Tensor Cores, Google's MXU, Ascend's Cube unit — and the surrounding memory hierarchy is organized around staging those blocks in on-chip SRAM. A programming model expressed in scalar threads has to be pattern-matched back into that shape by the compiler; a model expressed in tiles states it directly.

The hardware has continued moving in this direction rather than away from it. Blackwell introduced **TMEM**, a 256 KB per-SM memory dedicated to Tensor Core accumulators and separate from shared memory — a memory space that only makes sense in tile terms. NVIDIA shipped **CUDA Tile IR**, an MLIR-based virtual ISA whose unit of computation is the tile, and then built a **Tile IR backend for OpenAI Triton**. <sup>[[36]](#ref-36)</sup> Their own framing, from that announcement: *"As GPU programming continues to evolve beyond traditional SIMT models toward tile-based abstractions…"* <sup>[[38]](#ref-38)</sup>

That NVIDIA — whose SIMT model *defined* GPU programming for two decades — now describes the field as evolving past it is the strongest available statement that tiles are the destination rather than a detour.

#### Why Python: the developer population moved too

Tiles explain the execution model. They do not explain why the source language is Python, given that every performance argument favours C++.

The reason is that kernel authoring stopped being a specialist activity performed in isolation. Kernels are now written by the same people who write the model, in the same file tree, against tensors that arrive from PyTorch or JAX. `torch.compile` generates Triton by default, which means every user of the dominant training framework is already producing Python-embedded tile kernels whether or not they read them. <sup>[[39]](#ref-39)</sup> A kernel language outside Python pays a permanent tax in glue code, type conversion, and build integration — and, increasingly, in tooling: the profilers, autotuners, and test harnesses all assume a Python object.

There is now a second-order effect worth naming. LLM-based kernel generation (§2.8) trains on public code, and public kernel code is overwhelmingly Python-embedded. Tenstorrent's TT-Lang states the goal explicitly — it is designed for AI agents to auto-translate Triton, CUDA, cuTile, and TileLang kernels into it. <sup>[[37]](#ref-37)</sup> A DSL that is not Python-shaped is harder for a model to write, which is becoming a real adoption cost rather than a hypothetical one.

#### What this settles, and what it does not

The convergence settles the *shape* of the answer: a new datacenter accelerator that wants third-party kernel authors will expose a Python tile DSL, because every other vendor has and because the alternative costs ecosystem access. For Ascend that question is closed — Triton-Ascend, TileLang-Ascend, PyPTO, and PyAsc2 all take this shape (§2.6).

It settles nothing about the two questions that follow, which the rest of §2 addresses: **whose** DSL, and **how much hardware detail** it exposes. Nine languages sharing a syntax family are still nine languages, and §2.1 showed frontier labs choosing among them on grounds of scheduling control, not ergonomics. Convergence on Python and tiles is the floor of the discussion, not its conclusion.

### 2.3 NVIDIA moves up-stack to Python DSLs

NVIDIA has the least incentive of any vendor to make kernel authoring portable or easy. CUDA C++ is the moat: two decades of accumulated code, tooling, and expertise that competitors must replicate before their silicon is usable. An abstraction layer above CUDA erodes exactly that advantage.

NVIDIA built two anyway, plus a Python-first rework of the surrounding stack. When the incumbent invests against its own lock-in, the pressure driving it is worth understanding.

#### Six years of NVIDIA Python — and almost none of it was a kernel DSL

The two DSLs read as a sudden move only if the preceding years are skipped. NVIDIA shipped Python continuously from 2021 onward; what it did *not* ship, until 2025, was a Python language for authoring AI kernels. Package-registry first-release dates make the sequence exact: <sup>[[56]](#ref-56)</sup>

| Date | What shipped | Kind | Kernels in Python? |
|---|---|---|---|
| 2021-10-21 | `cuda-python` — driver/runtime/NVRTC bindings <sup>[[41]](#ref-41)</sup> | Bindings | No |
| 2021-11-09 | cuNumeric public alpha — distributed NumPy drop-in <sup>[[58]](#ref-58)</sup> | Array library | No |
| 2022-05-06 | **Warp** — `@wp.kernel`, JIT to C++/CUDA <sup>[[42]](#ref-42)</sup> | **Kernel DSL** | Yes — for graphics/simulation |
| 2023-11-14 | `nvidia-cutlass` — Python interface to CUTLASS C++ <sup>[[56]](#ref-56)</sup> | Emitter/bindings | No |
| 2024-06-25 | `numba-cuda` — NVIDIA takes over Numba's CUDA target <sup>[[59]](#ref-59)</sup> | Kernel JIT | Yes — but SIMT, not tiles |
| 2024-06-29 | `nvmath-python` — cuBLAS/cuFFT/cuSOLVER APIs <sup>[[43]](#ref-43)</sup> | Library bindings | Device-side calls only |
| 2024-12-02 | Warp 1.5.0 — tile primitives on cuBLASDx/cuFFTDx <sup>[[57]](#ref-57)</sup> | Tiles reach Warp | Yes |
| 2025-02-27 | `cuda.cooperative`, `cuda.parallel` <sup>[[41]](#ref-41)</sup> | Algorithm primitives | Partly |
| **2025-06-06** | **CuTe DSL** (CUTLASS 4.0) <sup>[[35]](#ref-35)</sup> | **AI kernel DSL** | **Yes** |
| **2025-12-04** | **cuTile Python** (CUDA 13.1) <sup>[[36]](#ref-36)</sup> | **AI kernel DSL** | **Yes** |

Three things follow from the shape of that list.

**The gap is four years wide.** Triton was public in 2021. NVIDIA's first Python DSL aimed at AI kernels arrived in mid-2025 — and only after `torch.compile` had made Triton the default code generator for the dominant training framework (§2.2). Warp, the one earlier kernel DSL, was announced at GTC 2022 for *differentiable graphics and physics simulation*, and did not acquire a tile API until December 2024. <sup>[[42]](#ref-42)</sup> <sup>[[57]](#ref-57)</sup> It was not built for this fight and was not positioned for it.

**The nearest thing to a Python kernel language on NVIDIA hardware was not NVIDIA's.** Numba's CUDA target — SIMT, thread-indexed, a direct Python transcription of the CUDA model — was a community project for a decade. NVIDIA assumed maintainership only in June 2024, and the built-in Numba target is now deprecated in its favour. <sup>[[59]](#ref-59)</sup>

**Everything before 2025 pointed away from kernels.** Bindings, array drop-ins, and library wrappers all serve the user who wants Python *without* writing a kernel. That is a coherent strategy — keep the kernel layer in C++, where the moat is — and NVIDIA held it for four years before reversing.

The reversal was resourced as a programme, not a product. GTC 2025 was framed as the "Year of CUDA Python": `cuda-python` was restructured into `cuda.core` / `cuda.bindings` / `cuda.cooperative` / `cuda.parallel` (with `cuda.core` still explicitly stabilizing), Warp's tile API became first-class through the 1.10–1.15 series, nvmath-python reached 1.0 GA, and cuPyNumeric went fully open source including the Legate runtime. <sup>[[41]](#ref-41)</sup> <sup>[[42]](#ref-42)</sup> <sup>[[43]](#ref-43)</sup> <sup>[[44]](#ref-44)</sup> The two kernel DSLs are the visible tip of that.

#### Two DSLs, not one — and they are constantly confused

The single most common error in this area is treating cuTile and CuTe DSL as versions of the same thing. They are different products, from different teams, at opposite ends of the abstraction range, released six months apart:

| | **cuTile** | **CuTe DSL** |
|---|---|---|
| Abstraction | High-level — compiler assigns threads | Low-level — you control the thread/data hierarchy |
| Analogue | Triton | CUTLASS/CuTe in C++ |
| Ships with | CUDA Toolkit | CUTLASS 4.x (`nvidia-cutlass-dsl`) |
| Underlying IR | CUDA Tile IR (MLIR-based) | CuTe layout algebra |
| First public | 2025-12-04 (CUDA 13.1) <sup>[[36]](#ref-36)</sup> | 2025-06-06 (CUTLASS 4.0) <sup>[[35]](#ref-35)</sup> |
| Status (2026-08) | Stable since 13.2; Hopper added in 13.3; C++ front-end added in 13.3 <sup>[[36]](#ref-36)</sup> | **Still Beta** at 4.7.0 (2026-08-05) <sup>[[52]](#ref-52)</sup> |

That NVIDIA needed *both* is the substantive point. A single abstraction level did not cover the range between "write a fused operator quickly" and "extract the last 20% from a new architecture" — the same split visible at DeepSeek (§2.1), at Meta, and inside Triton itself, where the answer to the same gap was Gluon. **No vendor has yet shipped one Python DSL that spans the whole range.**

#### What cuTile is for: 80% of peak, in a week

The facts above say what cuTile *is*. The intent is stated plainly enough in NVIDIA's own material to quote.

**The audience is everyone who is not writing a compiler.** NVIDIA's framing: *"Most developers will interface with CUDA tile programming through software like NVIDIA cuTile Python,"* while *"for developers looking to build their own DSL compiler or library, CUDA Tile IR is where you'll interface."* <sup>[[48]](#ref-48)</sup> Two audiences, two entry points, stated as such.

**The performance target is explicitly not peak.** Stephen Jones, NVIDIA distinguished engineer, put the productivity trade in numbers: *"If you can come to market [with] 80% of performance in a week instead of a month… then you're spending the rest of your time just optimizing."* <sup>[[49]](#ref-49)</sup> That is the cuTile pitch in one sentence — and it is the same bargain Triton offers, which is why the comparison is unavoidable.

**It does not replace SIMT, by design.** *"It's not an either/or situation… you don't have to choose between SIMT and tile programming; they coexist,"* and *"when you need SIMT, you write your kernels as you always have."* <sup>[[48]](#ref-48)</sup> cuTile is an added tier, not a migration.

**The durable selling point is forward compatibility, not speed.** *"CUDA Tile abstracts away tensor cores… so that code using CUDA Tile is compatible with current and future tensor core architectures."* <sup>[[48]](#ref-48)</sup> With TMEM, block-scaled FP4, and new tensor-core shapes arriving each generation, a kernel pinned to one architecture's intrinsics is a liability. This is the argument that survives even if a hand-written kernel is faster today.

**And the competitive read is not subtle.** Nicholas Wilt, a founding member of the CUDA team: *"It's hard not to suspect that cuTile was developed directly to counter Triton."* <sup>[[50]](#ref-50)</sup> NVIDIA has not said this. But cuTile occupies Triton's exact abstraction level, ships in the CUDA Toolkit where Triton must be installed separately, and arrived four years after Triton with the same 80%-fast value proposition. The obvious reading is the correct one: this is the productivity tier being brought in-house.

#### What CuTe DSL actually buys — and what it does not

CuTe DSL is easier to misread, because "Python DSL" invites the assumption that the benefit is brevity. It is not. Measured on NVIDIA's own examples, a Blackwell dense GEMM in CuTe DSL is **1,797 lines of Python (1,378 excluding blanks and comments)**, against **486 lines (291 code) for the C++ Blackwell GEMM example**. <sup>[[54]](#ref-54)</sup> The comparison is not apples-to-apples in NVIDIA's favour either — the C++ example *calls* `CollectiveBuilder`, assembling a kernel from pre-written library templates, while the Python example builds the mainloop from CuTe primitives. That is exactly the point: **CuTe DSL replaces the layer where CUTLASS collectives are written, not the layer where they are called.** Nobody adopts it to write less code.

What it does buy, from NVIDIA's own statements and release notes:

| Benefit | Evidence | Strength |
|---|---|---|
| **Compile time** | *"orders of magnitude faster compile times"* than C++ template instantiation <sup>[[51]](#ref-51)</sup> | Stated, not benchmarked publicly |
| **No C++ metaprogramming** | *"much more intuitive metaprogramming that does not require deep C++ expertise"*; compile-time configuration in Python instead of nested templates <sup>[[51]](#ref-51)</sup> | Structural — verifiable by reading either codebase |
| **Performance parity** | *"without any performance compromises"* <sup>[[51]](#ref-51)</sup> — but the docs concede kernels *aim* to match C++ and *"some performance gaps may exist due to missing optimizations"* <sup>[[35]](#ref-35)</sup> | Claimed; qualified by NVIDIA itself |
| **Framework integration** | *"native integration with DL frameworks without writing glue code"* <sup>[[51]](#ref-51)</sup> | Structural |
| **Compiler diagnostics** | 4.7 reports register spills and local-memory use with source line numbers; new Task Scheduling framework does static analysis of warp-specialized schedules and *"compilation stops when known concurrency issues are detected"* <sup>[[52]](#ref-52)</sup> | Shipped 2026-08-05 |

The diagnostics line is the one most relevant to PyAsc2. Warp specialization and multi-stage pipelining are where hand-written kernels break, and NVIDIA's answer was not to hide the schedule but to **verify it at compile time and refuse to build when it is wrong**. That is a DSL earning its keep as a correctness tool, not just an ergonomics one — and it is a capability an Ascend DSL could offer against the same class of bug (UB double-buffering, cross-pipe sync).

**Adoption is real, which the Beta label understates.** CUTLASS 4.7 lists its tested-against set as FlashAttention, FlashInfer, cuDNN-Frontend, and Quack. <sup>[[52]](#ref-52)</sup> Quack — *"A Quirky Assortment of CuTe Kernels"* from Dao-AILab, ~1.1k stars — is a full kernel library (RMSNorm, softmax, cross-entropy, layernorm, Hopper/Blackwell GEMM, forward and backward) written entirely in CuTe DSL, with a published account of driving memory-bound kernels to speed-of-light *"right in the comfort of Python."* <sup>[[55]](#ref-55)</sup> Fourteen months of Beta has not stopped the FlashAttention authors from building a kernel library on it.

#### The strategy: nested DSLs over a published IR contract

The four comment-worthy facts above — two DSLs, a C++ tile front-end, a Triton backend, a new sub-CuTe primitives layer — are not four separate moves. They are one: **NVIDIA is not shipping a DSL, it is shipping a stack of nested abstraction tiers over a stable IR contract, and inviting others to enter at whichever tier they can afford.**

| Tier | Surface | NVIDIA's stated audience | Lowers to |
|---|---|---|---|
| No kernels at all | cuPyNumeric, nvmath-python <sup>[[43]](#ref-43)</sup> <sup>[[44]](#ref-44)</sup> | Users who never write a kernel | Libraries |
| Tile, high-level | **cuTile Python**; **CUDA Tile C++** (13.3) <sup>[[36]](#ref-36)</sup> | *"Most developers"* <sup>[[48]](#ref-48)</sup> | CUDA Tile IR |
| Tile, third-party | **OpenAI Triton** (NVIDIA-built backend, 2026-01) <sup>[[38]](#ref-38)</sup> | Existing Triton users | CUDA Tile IR |
| **Contract** | **CUDA Tile IR** — a virtual ISA | *"developers looking to build their own DSL compiler or library"* <sup>[[48]](#ref-48)</sup> | NVVM → PTX |
| Layout algebra, ergonomic | `cute.experimental` (4.4) — fragment-free copy/dot, automatic TMA descriptors <sup>[[53]](#ref-53)</sup> | CuTe users wanting less boilerplate | CuTe DSL |
| Layout algebra, full control | **CuTe DSL** <sup>[[35]](#ref-35)</sup> | Kernel and library authors | NVVM → PTX |
| Below CuTe | **Primitives API** (4.7) — *"a stable, thin wrapper over NVVM operations to use where CuTe abstractions reduce development velocity"* <sup>[[52]](#ref-52)</sup> | Authors CuTe is slowing down | NVVM |
| Escape hatch | Inline PTX from Python <sup>[[52]](#ref-52)</sup> | Last resort | PTX |
| Unchanged | **SIMT CUDA C++** | *"when you need SIMT, you write your kernels as you always have"* <sup>[[48]](#ref-48)</sup> | PTX |

Read down the table and the design is legible. Every tier has a named audience. No tier deprecates the one below it. And each is a strict escape hatch for the one above — a cuTile user who hits a wall drops to CuTe DSL; a CuTe DSL user who hits a wall drops to Primitives, then to inline PTX, without leaving Python.

The load-bearing piece is the middle row. **CUDA Tile IR is published as a target, not kept as an implementation detail** — NVIDIA's stated purpose is to let others *"build higher-level hardware-specific compilers, frameworks, and domain-specific languages (DSLs) for NVIDIA hardware."* <sup>[[48]](#ref-48)</sup> Retargeting Triton onto it was the first demonstration, by NVIDIA itself, that the contract works for a front-end NVIDIA does not own. <sup>[[38]](#ref-38)</sup>

Two consequences worth stating separately. First, **the moat moves from the language to the IR.** A competitor can clone cuTile's syntax; what it cannot clone is a virtual ISA with a compiler, an autotuner, and a debugger behind it. Second, **the tiers keep multiplying in both directions** — `cute.experimental` was added *above* CuTe in February 2026 and the Primitives API *below* it in August 2026, the latter documented as *"a transitional API until a CUDA Python-like solution is available."* <sup>[[52]](#ref-52)</sup> <sup>[[53]](#ref-53)</sup> Even NVIDIA is still searching for the right number of layers.

#### Maturity: both are younger than the announcements suggest

The gap between announcement and production readiness has been substantial in both cases, which matters when judging how fast this space actually moves.

**cuTile** was unveiled at GTC 2025 (March) but shipped **experimental** in CUDA 13.1 (2025-12-04), became a stable feature in 13.2, and only gained **Hopper (sm_90) support in CUDA 13.3 (2026-05-26)** — meaning the flagship H100/H200 generation was unsupported for roughly the first five months of availability. <sup>[[36]](#ref-36)</sup> The `cutile-python` README still states that Hopper support is forthcoming, contradicting the shipped compiler.

**CuTe DSL** launched with CUTLASS 4.0 in June 2025 and is still marked **Beta** at 4.7.0 (2026-08-05), fourteen months later; the README as of 2026-08-14 still states it "will graduate out of beta by end of summer 2026." <sup>[[51]](#ref-51)</sup> <sup>[[52]](#ref-52)</sup> Its own docs state that generated kernels *aim* to match CUTLASS C++ while acknowledging that "some performance gaps may exist due to missing optimizations." <sup>[[35]](#ref-35)</sup>

**Performance claims should be treated carefully.** NVIDIA has published no primary cuTile-versus-cuBLAS benchmark; the launch material's quantified speedup figures are for cuBLAS itself, not cuTile. Independent evaluation exists but is third-party. <sup>[[40]](#ref-40)</sup> Any statement that cuTile matches vendor libraries is currently a roadmap claim, not a measured one. The same caution applies to the Beta label in the other direction: it has not deterred FlashAttention, FlashInfer, or Quack from shipping on CuTe DSL. <sup>[[52]](#ref-52)</sup>

#### NVIDIA adopting a competitor's front-end

The clearest signal is not any of NVIDIA's own languages. In January 2026 NVIDIA published a **CUDA Tile IR backend for OpenAI Triton** — engineering effort spent making a language they did not write, and which explicitly targets AMD hardware as well, run better on their own. <sup>[[38]](#ref-38)</sup>

A vendor builds a backend for someone else's front-end when that front-end has become the interface its customers already use. NVIDIA's GTC 2025 session on the topic was titled *"Blackwell Programming for the Masses With OpenAI Triton."* <sup>[[45]](#ref-45)</sup>

#### Reading this from Ascend

Five implications carry over.

**The thesis is validated by the least likely party.** If Python tile DSLs were merely a convenience for vendors with weak toolchains, NVIDIA — with the strongest toolchain and the most to protect — would have no reason to build two of them and a Triton backend besides. That it did is evidence the pressure is structural.

**One abstraction level is not enough, and NVIDIA proved it at cost.** cuTile and CuTe DSL exist as a pair because neither alone spans the range, and NVIDIA has since added tiers *above* CuTe (`cute.experimental`) and *below* it (Primitives API). An Ascend strategy that positions a single DSL as covering everything from rapid prototyping to peak extraction is claiming something no vendor has yet demonstrated — including the vendor that has tried hardest.

**The unit of strategy is the stack, not the language.** NVIDIA's durable asset is CUDA Tile IR: a published contract that cuTile, CUDA Tile C++, and a competitor's Triton all lower onto. PyAsc2 faces the identical question, and Ascend already has the raw material in AscendNPU-IR. Whether that IR is treated as an internal detail or published as a target others can compile to decides whether PyAsc2 is one language or an ecosystem position. This is a §3 decision, and it is the most consequential one in this section.

**Positioning should name the performance tier out loud.** NVIDIA does: cuTile is *"80% of performance in a week,"* CuTe DSL is peak-with-full-control, and neither pretends to be the other. <sup>[[49]](#ref-49)</sup> A DSL that declines to say which tier it occupies gets judged against the wrong baseline by everyone who evaluates it.

**Announcement-to-maturity is measured in years, not quarters.** cuTile took roughly fourteen months from unveiling to supporting NVIDIA's own flagship generation; CuTe DSL has been Beta for fourteen months and counting — with NVIDIA's resources. This is the realistic calibration for PyAsc2's roadmap in §3.

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
| <a name="ref-18"></a>[18] | `deepseek-ai/TileKernels` — TileLang kernel library, MIT, created 2026-04-22; 122 `.py` / 0 `.cu`. Kernels: [`engram/engram_gate_kernel.py`](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/engram/engram_gate_kernel.py) · [`engram/engram_hash_kernel.py`](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/engram/engram_hash_kernel.py) · [`mhc/expand_kernel.py`](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/mhc/expand_kernel.py) | [repo root](https://github.com/deepseek-ai/TileKernels/tree/main/tile_kernels) |
| <a name="ref-19"></a>[19] | `deepseek-ai/FlashMLA` — MLA and sparse attention, CUDA C++/CUTLASS. Kernels: [`sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu`](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu) · [`prefill/sparse/fwd/head128/instantiations/phase1_k512.cu`](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu) · [`kerutils/device/sm100/gemm.cuh`](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/kerutils/include/kerutils/device/sm100/gemm.cuh) | [csrc/](https://github.com/deepseek-ai/FlashMLA/tree/main/csrc) |
| <a name="ref-20"></a>[20] | `deepseek-ai/DeepGEMM` — FP8/FP4/BF16 GEMM in CUDA C++; Python files are bindings. The mHC prenorm GEMM cited in §2.1: [`impls/sm100_tf32_hc_prenorm_gemm.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh) · [`jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp`](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp) | [deep_gemm/include/](https://github.com/deepseek-ai/DeepGEMM/tree/main/deep_gemm/include/deep_gemm/impls) |
| <a name="ref-21"></a>[21] | `deepseek-ai/DeepEP` — MoE all-to-all dispatch/combine; CUDA C++ with inline PTX. **[`csrc/kernels/legacy/utils.cuh`](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/legacy/utils.cuh) contains exactly 55 `asm volatile` sites** (verified 2026-08-13); backends: [`kernels/backend/nvshmem.cu`](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/backend/nvshmem.cu) · [`kernels/backend/nccl.cu`](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/backend/nccl.cu) | [csrc/kernels/](https://github.com/deepseek-ai/DeepEP/tree/main/csrc/kernels) |
| <a name="ref-22"></a>[22] | `THUDM/slime` — GLM-5/5.2 training kernels in TileLang (repo is under THUDM, not zai-org). The four files, 823 LOC total: [`tilelang_sparse_mla_fwd.py`](https://github.com/THUDM/slime/blob/main/slime_plugins/models/glm5/ops/tilelang_sparse_mla_fwd.py) (209) · [`tilelang_sparse_mla_bwd.py`](https://github.com/THUDM/slime/blob/main/slime_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py) (309) · [`tilelang_indexer_fwd.py`](https://github.com/THUDM/slime/blob/main/slime_plugins/models/glm5/ops/tilelang_indexer_fwd.py) (140) · [`tilelang_indexer_bwd.py`](https://github.com/THUDM/slime/blob/main/slime_plugins/models/glm5/ops/tilelang_indexer_bwd.py) (165). Each carries an `Adapted from` header pinned to a DeepSeek V3.2 example commit. The two-tier split is in [`sparse_mla.py`](https://github.com/THUDM/slime/blob/main/slime_plugins/models/glm5/ops/sparse_mla.py) — `SGLangSparseMLA` = CUDA forward + TileLang backward | [glm5/ops/](https://github.com/THUDM/slime/tree/main/slime_plugins/models/glm5/ops) |
| <a name="ref-23"></a>[23] | `QwenLM/FlashQLA` — Gated DeltaNet kernels in TileLang, 41 `.py` / 0 `.cu`. Per-architecture trees: [`chunk/blackwell/fused_fwd.py`](https://github.com/QwenLM/FlashQLA/blob/main/flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py) · [`chunk/blackwell/fused_bwd.py`](https://github.com/QwenLM/FlashQLA/blob/main/flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py) · [`chunk/blackwell/cp_fwd.py`](https://github.com/QwenLM/FlashQLA/blob/main/flash_qla/ops/gated_delta_rule/chunk/blackwell/cp_fwd.py) · [`chunk/hopper/`](https://github.com/QwenLM/FlashQLA/tree/main/flash_qla/ops/gated_delta_rule/chunk/hopper). README states manual warpgroup specialization; issue [#14](https://github.com/QwenLM/FlashQLA/issues/14) (2026-06-30) declines Ascend support | [flash_qla/ops/](https://github.com/QwenLM/FlashQLA/tree/main/flash_qla/ops) |
| <a name="ref-24"></a>[24] | Moonshot AI Kimi K3 — [repo](https://github.com/MoonshotAI/Kimi-K3) + in-repo `k3_tech_report.pdf` (2026-07-27; no arXiv ID). **FlashKDA** (KDA linear attention) is CUDA/CuTe C++: [`csrc/smxx/fwd_kernel1.cuh`](https://github.com/MoonshotAI/FlashKDA/blob/master/csrc/smxx/fwd_kernel1.cuh) · [`csrc/smxx/fwd_kernel2.cuh`](https://github.com/MoonshotAI/FlashKDA/blob/master/csrc/smxx/fwd_kernel2.cuh) · [`csrc/smxx/fwd_launch.cu`](https://github.com/MoonshotAI/FlashKDA/blob/master/csrc/smxx/fwd_launch.cu). **MoonEP** (MoE dispatch/combine) is CuTe **DSL** in Python — [`moonep/dispatch.py`](https://github.com/MoonshotAI/MoonEP/blob/master/moonep/dispatch.py) and [`moonep/combine.py`](https://github.com/MoonshotAI/MoonEP/blob/master/moonep/combine.py) both open with `import cutlass.cute as cute` | [FlashKDA](https://github.com/MoonshotAI/FlashKDA) · [MoonEP](https://github.com/MoonshotAI/MoonEP) |
| <a name="ref-25"></a>[25] | `MiniMax-AI/MSA` (2026-06-11) — block-sparse attention, FP8/FP4 paged KV. CuTe **DSL** (Python): [`cute/interface.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/interface.py) · [`cute/quantize.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/quantize.py) · [`cute/sparse_index_utils.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/sparse_index_utils.py). CUDA: [`csrc/fmha_sm100_plan.cu`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/csrc/fmha_sm100_plan.cu) · [`csrc/include/fmha_cutlass_sm100.cuh`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/csrc/include/fmha_cutlass_sm100.cuh). No Triton, no TileLang | [python/fmha_sm100/](https://github.com/MiniMax-AI/MSA/tree/main/python/fmha_sm100) |
| <a name="ref-26"></a>[26] | `Tencent/hpc-ops` — FP8 MoE and inference ops in CUDA/CuTe/CUTLASS; 0 Triton, 0 TileLang. Kernels: [`allreduce/fuse_allreduce_rmsnorm_low_latency.cu`](https://github.com/Tencent/hpc-ops/blob/main/src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu) · [`allreduce/fuse_allreduce_rmsnorm_high_throughput.cu`](https://github.com/Tencent/hpc-ops/blob/main/src/allreduce/fuse_allreduce_rmsnorm_high_throughput.cu) · [`attention/decode/assign_task.cu`](https://github.com/Tencent/hpc-ops/blob/main/src/attention/decode/assign_task.cu) · [`activation/activation.cu`](https://github.com/Tencent/hpc-ops/blob/main/src/activation/activation.cu) | [src/](https://github.com/Tencent/hpc-ops/tree/main/src) |
| <a name="ref-27"></a>[27] | OpenAI `gpt-oss` — in-repo Triton kernels: [`gpt_oss/triton/attention.py`](https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/attention.py) · [`gpt_oss/triton/model.py`](https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/model.py) | [gpt_oss/triton/](https://github.com/openai/gpt-oss/tree/main/gpt_oss/triton) |
| <a name="ref-28"></a>[28] | Triton Developer Conference, Meta session (2025-11-05) — "Scheduling is usually a question that drives this choice" (Triton vs CUTLASS/CuTe DSL) | https://github.com/triton-lang/triton/tree/main/docs/meetups |
| <a name="ref-29"></a>[29] | GLM-5 technical report, arXiv 2602.15763 — §3.2: nondeterministic top-k in "CUDA or TileLang" implementations caused "drastic performance degradation during RL"; reverted to `torch.topk`. Report does not state training hardware and does not mention MindSpore | https://arxiv.org/abs/2602.15763 |
| <a name="ref-30"></a>[30] | `vllm-project/vllm-ascend` — tutorials for DeepSeek-V4-Pro/Flash, GLM-5.2, Kimi-K2.6, MiniMax on Atlas A2/A3; pins `triton-ascend==3.2.1`; imports `triton.language.extra.cann.extension` (Ascend-specific intrinsics) | https://github.com/vllm-project/vllm-ascend |
| <a name="ref-31"></a>[31] | `Ascend/MindSpeed-LLM` — GLM-5.2 pretraining scripts added 2026-06-18, two days after model release | https://gitee.com/ascend/MindSpeed-LLM |
| <a name="ref-46"></a>[46] | vLLM-Ascend Kimi K3 / KDA support — **implemented in Triton, not Ascend C**: [`vllm_ascend/ops/triton/kda/`](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops/triton/kda) holds 7 modules with 16 `@triton.jit` kernels ([`chunk_delta_h.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/kda/chunk_delta_h.py) · [`kda.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/kda/kda.py) · [`solve_tril.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/kda/solve_tril.py)), headers attributing the code to flash-linear-attention (MIT). Verified 2026-08-13 | [vllm-ascend](https://github.com/vllm-project/vllm-ascend) |
| <a name="ref-47"></a>[47] | `Eco-Tech/Kimi-K3-w4a8` — community (not official Huawei) Ascend checkpoint; [`k3_bf16_conversion_manifest.json`](https://huggingface.co/Eco-Tech/Kimi-K3-w4a8/blob/main/k3_bf16_conversion_manifest.json) states "MXFP4 dequantization is lossy and does not reconstruct the original pre-quantization BF16 weights" | [model card](https://huggingface.co/Eco-Tech/Kimi-K3-w4a8) |
| <a name="ref-32"></a>[32] | `fla-org/flash-linear-attention` — community linear-attention library with a dedicated **`triton_ascend` backend family**: [`fla/modules/backends/triton_ascend/`](https://github.com/fla-org/flash-linear-attention/tree/main/fla/modules/backends/triton_ascend) (layernorm, rotary, l2norm, causal_conv1d, fused cross-entropy) and [`fla/ops/attnres/backends/triton_ascend/`](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/attnres/backends/triton_ascend). Ascend CI: [`ascend-a2-ci.yml`](https://github.com/fla-org/flash-linear-attention/blob/main/.github/workflows/ascend-a2-ci.yml) | [repo](https://github.com/fla-org/flash-linear-attention) |
| <a name="ref-33"></a>[33] | JAX Pallas design doc — "On GPUs, we lower Pallas to Mosaic GPU (formerly Triton), and on TPUs, we lower Pallas to Mosaic"; tile/BlockSpec programming model | https://docs.jax.dev/en/latest/pallas/design/design.html |
| <a name="ref-34"></a>[34] | TileLang, arXiv 2504.17577 (2025-04-24) — "a composable tiled programming model"; repo public since 2025-01-20 | https://arxiv.org/abs/2504.17577 · https://github.com/tile-ai/tilelang |
| <a name="ref-35"></a>[35] | NVIDIA CuTe DSL — Python DSL shipping with CUTLASS 4.x; `nvidia-cutlass-dsl` first release 4.0.0 on 2025-06-06, latest 4.6.1 (2026-07-13), still marked Beta | https://pypi.org/project/nvidia-cutlass-dsl/#history · https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html |
| <a name="ref-36"></a>[36] | NVIDIA CUDA Tile / cuTile — tile-based Python DSL + CUDA Tile IR (MLIR-based); experimental in CUDA 13.1 (2025-12-04), stable in 13.2, Hopper (sm_90) support added in 13.3 (2026-05-26) | https://developer.nvidia.com/blog/nvidia-cuda-13-3-enhances-gpu-development-with-tile-programming-in-c-compiler-autotuning-and-python-updates/ · https://github.com/NVIDIA/cutile-python |
| <a name="ref-37"></a>[37] | Tenstorrent TT-Lang — "a Python-based Domain-Specific Language (DSL) for authoring high-performance custom kernels"; ships `tt-lang` and `tt-lang-sim` (hardware-free simulator); designed for AI-agent translation of Triton/CUDA/cuTile/TileLang kernels | https://github.com/tenstorrent/tt-lang |
| <a name="ref-38"></a>[38] | NVIDIA, "Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton" (2026-01-30) — "As GPU programming continues to evolve beyond traditional SIMT models toward tile-based abstractions…" | https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/ |
| <a name="ref-39"></a>[39] | PyTorch 2.x / TorchInductor — "uses a define-by-run loop-level IR to automatically map PyTorch models into generated Triton code on GPUs"; Triton is the default GPU codegen backend for `torch.compile` | https://pytorch.org/get-started/pytorch-2-x/ |
| <a name="ref-40"></a>[40] | "Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs," arXiv 2604.23466 (~2026-04) — independent (non-NVIDIA) evaluation; NVIDIA has published no primary cuTile-vs-cuBLAS benchmark | https://arxiv.org/abs/2604.23466 |
| <a name="ref-41"></a>[41] | NVIDIA CUDA Python — `cuda.core`, `cuda.bindings`, `cuda.cooperative`, `cuda.parallel`; framed as the "Year of CUDA Python" at GTC 2025. Repo notes the stack is undergoing an overhaul; `cuda.core` still stabilizing | https://developer.nvidia.com/cuda/python · https://github.com/NVIDIA/cuda-python |
| <a name="ref-42"></a>[42] | NVIDIA Warp — Python kernel framework; tile API (`wp.tile_*`) became first-class across v1.10–1.15; current release v1.15.0 (2026-07-07) | https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md |
| <a name="ref-43"></a>[43] | `nvmath-python` 1.0 GA (2026-07-16) — stable Python APIs over cuBLAS/cuFFT/cuSOLVER/cuDSS, host- and device-side calls | https://docs.nvidia.com/cuda/nvmath-python/latest/ |
| <a name="ref-44"></a>[44] | NVIDIA cuPyNumeric 25.03 (2026-03) — fully open source incl. the Legate runtime; distributed multi-GPU NumPy drop-in (renamed from cuNumeric) | https://developer.nvidia.com/blog/nvidia-cupynumeric-25-03-now-fully-open-source-with-pip-and-hdf5-support/ |
| <a name="ref-45"></a>[45] | NVIDIA GTC 2025 session S72876, "Blackwell Programming for the Masses With OpenAI Triton" (Phil Tillet); same deck reports dense FP16 8192² matmul ~10% behind cuBLAS 12.8 on H100/GB200 | https://www.nvidia.com/en-us/on-demand/session/gtc25-s72876/ |
| <a name="ref-48"></a>[48] | NVIDIA, "Focus on Your Algorithm — NVIDIA CUDA Tile Handles the Hardware" — audience split: *"Most developers will interface with CUDA tile programming through software like NVIDIA cuTile Python"*; *"for developers looking to build their own DSL compiler or library, CUDA Tile IR is where you'll interface"*; coexistence: *"it's not an either/or situation… you don't have to choose between SIMT and tile programming; they coexist"*; forward compatibility: *"CUDA Tile abstracts away tensor cores… compatible with current and future tensor core architectures"*; Tile IR published so users can *"build higher-level hardware-specific compilers, frameworks, and domain-specific languages (DSLs) for NVIDIA hardware"* | https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware/ |
| <a name="ref-49"></a>[49] | Stephen Jones (NVIDIA distinguished engineer), interview on tile-based programming (2025-12-08) — *"If you can come to market [with] 80% of performance in a week instead of a month… then you're spending the rest of your time just optimizing"*; *"Python's the language of AI"* | https://www.marktechpost.com/2025/12/08/interview-from-cuda-to-tile-based-programming-nvidias-stephen-jones-on-building-the-future-of-ai/ |
| <a name="ref-50"></a>[50] | Nicholas Wilt (founding member of the CUDA team), quoted on cuTile — *"It's hard not to suspect that cuTile was developed directly to counter Triton"* | https://hyper.ai/en/news/47715 |
| <a name="ref-51"></a>[51] | `NVIDIA/cutlass` [README](https://github.com/NVIDIA/cutlass/blob/main/README.md) (main, retrieved 2026-08-14) — CUTLASS DSLs are *"Python native interfaces for writing high-performance CUDA kernels… without any performance compromises,"* giving *"a much smoother learning curve, orders of magnitude faster compile times, native integration with DL frameworks without writing glue code, and much more intuitive metaprogramming that does not require deep C++ expertise"*; *"CuTe DSL is currently in public beta and will graduate out of beta by end of summer 2026"* | [README](https://github.com/NVIDIA/cutlass/blob/main/README.md) |
| <a name="ref-52"></a>[52] | CUTLASS 4.7.0 (`nvidia-cutlass-dsl`, 2026-08-05) — **Primitives API**, *"a lower-level abstraction beneath CuTe enabling Tensor Core programming through SIMT… a stable, thin wrapper over NVVM operations to use where CuTe abstractions reduce development velocity,"* noted as *"a transitional API until a CUDA Python-like solution is available"*; **Task Scheduling framework** — static analysis of warp-specialized schedules where *"compilation stops when known concurrency issues are detected"*; register-spill/local-memory reporting with source line numbers; release tested against FlashAttention, Quack, FlashInfer, cuDNN-Frontend. Inline-PTX escape hatch: [`dsl_tutorials/inline_ptx.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/dsl_tutorials/inline_ptx.py) | https://pypi.org/project/nvidia-cutlass-dsl/#history · [release notes](https://github.com/NVIDIA/cutlass/blob/main/README.md) |
| <a name="ref-53"></a>[53] | CUTLASS 4.4.0 release notes (2026-02-26) — `cute.experimental`: *"a higher-level, composable layer on top of existing CuTe DSL APIs (not a separate abstraction), which can be mixed with existing CuTe DSL building blocks"*; fragment-free copy/dot on memrefs, automatic TMA descriptor generation, automatic vectorization | https://github.com/NVIDIA/cutlass/releases/tag/v4.4.0 |
| <a name="ref-54"></a>[54] | Lines-of-code comparison, measured 2026-08-14 on `NVIDIA/cutlass@main`: CuTe DSL [`cute/blackwell/kernel/dense_gemm/dense_gemm.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py) = 1,797 lines total / 1,378 excluding blanks and comments; C++ [`70_blackwell_gemm/70_blackwell_fp16_gemm.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu) = 486 / 291. The C++ example assembles a kernel via `CollectiveBuilder`; the Python example builds the mainloop from CuTe primitives — different layers, which is the finding | [examples/python/CuTeDSL/](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) |
| <a name="ref-55"></a>[55] | `Dao-AILab/quack` — *"A Quirky Assortment of CuTe Kernels,"* ~1.1k stars, created 2025-05-20, active 2026-08; RMSNorm/softmax/cross-entropy/layernorm fwd+bwd and Hopper/Blackwell GEMM entirely in CuTe DSL ([`gemm_sm100.py`](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm100.py) · [`cross_entropy.py`](https://github.com/Dao-AILab/quack/blob/main/quack/cross_entropy.py)); blog post on reaching memory-bound speed-of-light *"right in the comfort of Python"* | [repo](https://github.com/Dao-AILab/quack) · [memory-bound blog](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md) |
| <a name="ref-56"></a>[56] | PyPI first-release dates for NVIDIA Python packages (queried 2026-08-14): [`cuda-python`](https://pypi.org/project/cuda-python/#history) 11.5.0 2021-10-21 · [`warp-lang`](https://pypi.org/project/warp-lang/#history) 0.2.0 2022-05-06 · [`nvidia-cutlass`](https://pypi.org/project/nvidia-cutlass/#history) 3.3.0.0 2023-11-14 · [`numba-cuda`](https://pypi.org/project/numba-cuda/#history) 2024-06-25 · [`nvmath-python`](https://pypi.org/project/nvmath-python/#history) 0.1.0 2024-06-29, 1.0.0 2026-07-09 · [`cuda-cooperative`](https://pypi.org/project/cuda-cooperative/#history) and [`cuda-parallel`](https://pypi.org/project/cuda-parallel/#history) 2025-02-27 · [`nvidia-cutlass-dsl`](https://pypi.org/project/nvidia-cutlass-dsl/#history) 4.0.0 2025-06-06, 4.7.0 2026-08-05 | https://pypi.org/ |
| <a name="ref-57"></a>[57] | `NVIDIA/warp` CHANGELOG, v1.5.0 (2024-12-02) — first tile primitives: *"Support for cooperative tile-based primitives using cuBLASDx and cuFFTDx"*; original 2022 positioning was differentiable graphics and physics simulation | [CHANGELOG](https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md) · [2022 announcement](https://developer.nvidia.com/blog/creating-differentiable-graphics-and-physics-simulation-in-python-with-nvidia-warp/) |
| <a name="ref-58"></a>[58] | NVIDIA, "NVIDIA Announces Availability for cuNumeric Public Alpha" (2021-11-09) — distributed NumPy drop-in, *"requires zero code changes"*; an array library, not a kernel DSL | https://developer.nvidia.com/blog/nvidia-announces-availability-for-cunumeric-public-alpha/ |
| <a name="ref-59"></a>[59] | `NVIDIA/numba-cuda` — NVIDIA-maintained CUDA target for Numba, first release 2024-06-25; Numba's built-in CUDA target is *"deprecated, with further development moved to the NVIDIA numba-cuda package"* | https://github.com/NVIDIA/numba-cuda · https://nvidia.github.io/numba-cuda/ |

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
