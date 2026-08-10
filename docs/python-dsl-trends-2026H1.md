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

_TODO (next stage)._ What programming model the 2026 open-weights frontier models use for their performance-critical kernels — DeepSeek V4, GLM 5.2, Kimi K3, Qwen, MiniMax. Covers: the documented Triton→TileLang migration; the two-tier split (hand-written CUDA/PTX for GEMM, flagship attention, and network kernels; Python DSL for the long tail); a per-lab table; each lab's stated reason for its choice; a counter-example where DSL nondeterminism cost model quality; and the architectural demands that define the extensibility axis in §2.0. Supplies the kernel list for [Appendix A.1](#a1-broaden-the-performance-vs-usability-comparison).

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
| <a name="ref-14"></a>[14] | DeepSeek-V4 technical report, arXiv 2606.19348 — §3.1 fused MoE dispatch+GEMM+activation+combine megakernel: 1.50–1.73× over non-fused (up to 1.96× for RL rollout); self-reported | https://arxiv.org/abs/2606.19348 |

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

Two open decisions: **(a)** whether to pair each kernel with a measured performance number so usability and performance sit on the same axes — more useful, but requires a benchmark run rather than source inspection; **(b)** which implementation counts as the reference, given that a single operator now ships with several backends across different DSLs. Kernel selection should follow the SOTA-model analysis, so the set reflects what frontier models actually run.

### A.2 Dynamic shapes

**Unresolved — no position taken yet.** Static tile shapes are what make aggressive compile-time scheduling possible; PyAsc2 requires tile shapes known at JIT time, though tensor shapes may be runtime values. Real serving workloads are dynamic: variable sequence length, variable batch, ragged and paged attention, per-token MoE expert loads.

The open question is where the boundary belongs:

| Option | Cost |
|---|---|
| Recompile per shape bucket | Simple; risks JIT thrash and cache pressure at serving time |
| Pad to fixed tiles | Wastes compute at low occupancy |
| Symbolic tile dimensions in the IR | Most general; most expensive to build, and may forfeit the scheduling advantages static shapes provide |

What competing DSLs do, and what the 2026 model architectures actually require, should be established in the SOTA-model and cross-cutting-trends sections before this is decided.
