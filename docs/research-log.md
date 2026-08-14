# Research Log

*Chronological summary of all research discussions and decisions.*

---

## 2026-03-30 — Project Kickoff

### Project Definition

**Context:** Oleg initiated a research project to design an alternative Python DSL for Huawei Ascend NPU.

**Goals established:**
- Goal 1 — **Simplicity:** minimal lines of code, clean syntax, low barrier to entry
- Goal 2 — **Performance:** ≥90% of peak hardware potential on Ascend NPU

### Architecture Decisions

**DSL level:** Triton-like kernel DSL (operator/kernel level), with a roadmap toward graph-level later.
- Rationale: Modern neural network optimization requires more than just operator-level control (graph-level needed for fusion/scheduling), but kernel DSL is the right starting point.

**IR strategy:** MLIR (Python DSL → MLIR → AscendC codegen)
- Considered: direct CANN CCE-C generation, TVM TIR → Ascend target
- Chosen: MLIR — more flexible, reusable optimizations, extensible with custom Ascend dialect
- Key insight: Triton itself moved to MLIR (Triton IR → TritonGPU IR → LLVM)

**Code generation target:** AscendC (not abstract CANN backend)
- AscendC is Huawei's official C++-like API for custom operators on Ascend NPU
- Generates verifiable, debuggable code
- Full pipeline: Python DSL → MLIR → AscendC → CANN compiler → NPU binary

### Languages Selected for Analysis

| Language | Author | Why included |
|----------|--------|--------------|
| Triton | OpenAI | Primary reference; industry standard kernel DSL |
| Triton-Ascend | Huawei | Direct Ascend port of Triton — key competitor/reference |
| TileLang | tile-ai | Tile-based, Ascend support — key competitor/reference |
| TileLang-Ascend | tile-ai | TileLang adapter for Ascend A2/A3, AscendC codegen |
| Pallas | Google/JAX | Tile/grid model, TPU + GPU |
| Gluon | OpenAI | Warp-level control, exceeds Triton performance |
| cuTile | NVIDIA | Minimal LOC, compiler-automated performance |
| Mojo | Modular | Python superset, systems-level |
| AscendCraft | Research | LLM-driven DSL → AscendC auto-generation |
| Helion | Meta/PyTorch | PyTorch-native tile DSL, beats hand-written Triton |

### Key Findings from Language Analysis

**Dominant trend:** Tile-based abstractions dominate all new DSLs. Tile naturally fits accelerator memory hierarchies (HBM → L2 → SRAM → registers).

**Performance ceiling insights (from Triton community meetups):**
- Triton out-of-the-box: only ~80% peak (Jeff Niu, OpenAI, Jul 2025 meetup)
- Flash Attention without warp specialization: 45% compute throughput on H100; with WS: 69% (Meta, Mar 2025 meetup)
- Gluon (warp-level) needed to exceed Triton performance, but FMHA on B200 still slower than cuDNN
- Helion achieves 1.85x vs hand-written Triton on H100 GEMM with PyTorch-level abstraction

**For 90% peak on Ascend — unique requirements not present in GPU DSLs:**
1. Explicit **Cube Unit** (matrix) vs **Vector Unit** (elementwise) vs **Scalar Unit** routing
2. **L0→L1→L2→HBM** pipeline — deeper memory hierarchy than GPU
3. **Multi-AI-core scheduling** — analogous to warp specialization but at core level
4. **CopyIn→Compute→CopyOut** — explicit pipeline model specific to Ascend

**Syntax patterns identified for our DSL:**
- `tile` loop as core primitive (Helion, TileLang) — intuitive, minimal code
- Automatic scheduling + autotuning (Helion, cuTile) — key to simplicity
- Explicit memory hints (Pallas, Gluon) — required for 90% peak
- PyTorch-compatible syntax (Helion) — low barrier to entry
- Warp/wave-level escape hatch (Gluon) — for edge cases

### Ascend-Specific Findings

**Triton-Ascend** (gitcode.com/Ascend/triton-ascend):
- Huawei's official Triton fork for Ascend NPU
- Primary active development on gitcode.com (not gitee.com)
- Requirements: torch==2.6.0, torch-npu==2.6.0rc1
- Related ops repo: github.com/Ascend/triton-ascend-ops

**TileLang-Ascend** (github.com/tile-ai/tilelang-ascend):
- Released September 2025, open source
- Only third-party Python DSL with Ascend NPU backend (A2/A3)
- Two backends: AscendC & PTO route, AscendNPU IR route
- Active: pip install support added March 2026, T.Parallel added Dec 2025
- No published TFLOPS benchmark numbers vs AscendC (as of Mar 2026)

**AscendCraft** (arxiv 2601.22760):
- LLM-driven: compact DSL → LLM generates DSL code → transpile to AscendC
- Results: 98% compilation, 90% correctness, only 46% reach PyTorch eager perf
- **Not a competitor** — different niche (auto-generation, not manual programming)
- Relevant as: validation of intermediate DSL concept; host/kernel split design reference

### Process Rules Established

1. **Plan first** — write action plan, get approval before executing
2. **Never push to main directly** — always branch + PR
3. **Batch work** — avoid one-by-one token-heavy iterations
4. **Documents in English**
5. **Cite sources** — every data claim needs an inline reference

### Repository Setup

- GitHub repo: https://github.com/mov20/ascend-dsl-research (public)
- Files: PROJECT.md, README.md, ascend-dsl-comparison.md, ascend-dsl-syntax-perf.md
- PR workflow established for review before merge

---

## 2026-08-07 — CATLASS TLA DSL Analysis

**Context:** Oleg requested a full analysis of the `dsl` branch of Huawei's CATLASS repository
(`gitcode.com/cann/catlass`), which carries a Python frontend for Ascend kernels. Result:
[`docs/catlass-dsl-analysis.md`](catlass-dsl-analysis.md). Static source and git-history analysis
only — no build or execution (requires CANN ≥ 9.1.0 and Ascend 950 hardware).

### New Project Identified

**CATLASS TLA DSL** is a first-party Huawei Python DSL for Ascend, not previously in our tracking
tables. ~45k LOC (26k Python + 19k C++/MLIR), 13 authors, 151 commits in 12 weeks with accelerating
cadence. It belongs alongside Triton-Ascend and TileLang-Ascend in the Ascend Python-DSL landscape.

### Notable Findings

**1. It does not target AscendC.** Pipeline is Python → TLA MLIR dialect (79 ops) → ~23 lowering
passes → HIVM/HACC (AscendNPU-IR) → LLVM → device binary, bypassing AscendC source generation
entirely. Notable contrast with TileLang-Ascend, which does generate AscendC — the vendor's own DSL
team chose the AscendNPU-IR path instead.

**2. Explicit synchronization is a usability dead end, and they know it.** The DSL Flash Attention
example needs ~60 hand-declared sync flags and comes out *longer* than the C++ template version
(1,527 vs 1,401 LOC). Basic matmul is 334 LOC vs 148 LOC in C++. Their answer is
`@tla.kernel(auto_sync="v0")` — compiler-inferred synchronization via `TlaInsertAutoMutexPass`,
which cuts basic matmul from 334 → 259 LOC and removes all 15 flags.

**3. SIMT-on-AIV is viable.** `tla.vec.func(mode="simt", thread_block_dim=N)` with `thread_idx()`
compiles a CUDA-shaped kernel onto Ascend vector cores — `gm_c[i] = gm_a[i] + gm_b[i]`, no UB
staging, no tiles, no flags. If this holds up, the "Ascend cannot do SIMT" assumption behind much
tile-first DSL design deserves re-examination. Caveat: landed 2026-08-07 with exactly 3 supporting
ops (`simt_add`, `simt_load`, `simt_store`).

**4. Cross-core AIC↔AIV sync is the hard part nobody abstracts.** Both CATLASS DSL and
TileLang-Ascend expose it explicitly (`cross_flag` / `cross_core_set_flag` / `cross_core_wait_flag`,
with sync topology `mode` 1/2/4). Whichever DSL hides it first while keeping performance wins the
usability argument on Ascend.

**5. No communication primitives at all.** Zero HCCL / all-reduce / all-gather / reduce-scatter
matches across the entire repository, both branches. Single-device kernel DSL only — anything
distributed needs a separate layer.

### Maturity Assessment

**Recommendation: TRACK, do not adopt.** Reasons:

- **Never released** — no git tag (v1.0.0 through v1.6.3) contains `python/tla_dsl`.
- **Ascend 950PR/950DT only** — `SUPPORTED_ARCH_SCOPES = ("aiv.c310", "aic.c310")`. Nothing runs on
  A2/A3, which is what most hardware access looks like today.
- **No CI** — ~23,500 LOC of tests with no automated gate. "DSL CI" is a Q3 2026 roadmap goal.
- **~3 months old**, self-labeled beta (first commit 2026-05-16).
- **License** is CANN Open Software License Agreement v2.0 — Huawei-authored, not OSI-approved.
- **Branch is diverging** — `dsl` is 154 ahead / 143 behind `master`, forked 2026-05-13, while
  mainline restructures for v2.0.0.

Counterweight: engineering quality is genuinely good. Test LOC ≈ source LOC, MLIR-based architecture,
healthy bus factor (top contributor 18% of commits), and a well-written English syntax-constraints
document added during the analysis window.

### Their Q3 2026 Roadmap (issue #399, opened 2026-08-04)

Still outstanding by their own account: 40+ SIMD ops needed for Matmul/FA, NZ data format,
MxFP8/MxFP4/FP8/int8/int32 dtypes, tensor subscript access, L0C2UB and UB2L1 paths, cross-core sync
modes 1/2/4, DSL CI, and a `dsl-gen` backend for torch.inductor. Planned operators: SplitK, StreamK,
fullLoad, GroupMatmulSliceM, PFA, KDA, BSA — all Ascend 950.

### Follow-ups Added

See Open TODOs below. The highest-value experiment is measuring what `auto_sync="v0"` costs in
performance versus hand-placed flags — that number bounds how much synchronization any Ascend DSL
can hide without paying for it.

---

## 2026-08-07 — CATLASS DSL: Automation Audit

**Context:** Follow-up question on the CATLASS TLA DSL — are double buffering, UB management, and
synchronization insertion automated, or explicit programmer work? Result:
[`docs/catlass-dsl-automation.md`](catlass-dsl-automation.md). Read from the MLIR pass source and all
44 end-to-end examples; no build or execution.

### Answer

| Concern | Automated? | Detail |
|---|---|---|
| Double buffering | **No** | No pass creates or manages it. Programmer allocates each half, names them, maintains the toggle index, writes the select |
| UB / on-chip memory | **Partial** | Compiler bump-allocates static byte offsets. No liveness, no reuse, no capacity fitting, no spilling. Sizes must be compile-time constant |
| Sync insertion | **Opt-in, intra-core only** | `auto_sync="v0"` infers intra-core mutexes. `cross_core_*` remains explicit, by the pass's own diagnostic |

### Supporting Detail

**`auto_sync="v0"` is narrower than it sounds.** Four limits from the pass source
(`TlaInsertAutoMutexPass`, 967 LOC, 23 error paths): cross-core sync is never automated (two separate
mutex ID spaces, Cube and Vector); it is all-or-nothing, so it cannot be mixed with any hand-placed
flag or mutex, ruling out incremental adoption; it bails on non-static buffer roots — "changing
loop-carried pointers are unsupported", which is exactly the deep-pipelining idiom; and it caps at 32
mutex IDs. Adoption reflects this: **2 of 44 examples** use it, versus 31 using manual flags. The
flagship Flash Attention kernel does not use it.

**Memory "management" is a bump allocator.** `planTlaScratchAllocations` assigns monotonically
increasing byte offsets per address space and stops there. Two buffers with disjoint live ranges
still occupy distinct bytes for the whole kernel. `LocalmemAllocator`, despite the name, is
stateless (`__slots__ = ()`) and emits the same `alloc_ptr` op — it takes bytes instead of elements,
nothing more.

**Scale of the remaining manual burden.** Flash Attention: 33 allocations, 33 flags, 7 cross-core
flags, 200 `set_flag`/`wait_flag` calls, 56 lines of buffer-index bookkeeping, ping/pong encoded in
identifier names — in 1,400 lines.

**SIMT is the one exception.** `tla.vec.func(mode="simt")` needs no allocation, no flags, no
buffering — operations lower onto GM memrefs directly. Three ops old, vector-only.

### Reading

The consistent pattern across all three concerns: the compiler analyzes what the programmer wrote and
fills in bookkeeping; it does not make scheduling decisions. Closer to a typed Python notation for
Ascend C than to a scheduling compiler — which matches the project's own statement that
template-style higher-level wrapping remains future work. Nothing on the Q3 2026 roadmap changes the
buffering or allocation picture.

Reinforces the existing TODO on benchmarking `auto_sync="v0"` overhead: if inferred mutexes are
conservative, the 75-line saving on basic matmul buys a performance regression, and that trade
decides whether the feature grows past 2 examples.

---

## Open TODOs

- [ ] Deep dive into AscendCraft paper — DSL design, host/kernel split, UB/L1 buffer model
- [ ] Deep dive into TileLang-Ascend — get actual benchmark numbers vs AscendC
- [ ] Start designing syntax for our DSL
- [ ] Determine access to Ascend hardware for benchmarks
- [ ] Decide target audience: ML engineers vs kernel developers
- [ ] Decide licensing/open-source strategy
- [ ] Compare codegen targets across Ascend DSLs: AscendC (TileLang-Ascend) vs AscendNPU-IR/HIVM (CATLASS DSL)
- [ ] **Benchmark `auto_sync="v0"` overhead** in CATLASS DSL vs hand-placed flags (needs 950 hardware)
- [ ] Study `TlaInsertAutoMutexPass` as a reference implementation for automatic sync insertion
- [ ] Add CATLASS TLA DSL to `ascend-dsl-comparison.md` (Table 1) and `ascend-dsl-syntax-perf.md` (Table 2)
- [ ] Re-evaluate CATLASS DSL when: it ships in a tagged release, A2/A3 support lands, or CI goes live
