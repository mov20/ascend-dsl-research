# CATLASS TLA DSL — Project Analysis (Huawei CANN)

*Last updated: 2026-08-07*

> Analysis of the `dsl` branch of **CATLASS**, Huawei's Ascend operator template library, which
> carries a Python frontend ("TLA DSL") built on MLIR. Companion to
> [`python-dsl-trends-2026H1.md`](python-dsl-trends-2026H1.md) (cross-DSL trend synthesis) and
> [`asic-landscape.md`](asic-landscape.md) (hardware + DSL-support matrix). This document covers a
> single project in depth: what it is, how it compiles, what its programming model looks like for
> vector / cube / mixed kernels, what synchronization and communication primitives it offers, and
> whether it is stable and active enough to depend on.
>
> **Why it belongs in this survey:** CATLASS DSL is a *first-party Huawei* Python DSL for Ascend,
> built on MLIR. That places it alongside Triton-Ascend and TileLang-Ascend in the Ascend Python-DSL
> landscape — a project not previously covered by our tracking tables.
>
> **See also:** [`catlass-dsl-automation.md`](catlass-dsl-automation.md) — a focused audit of whether
> double buffering, UB management, and synchronization insertion are compiler-automated or manual.

**Analyzed at commit `c511c43`** ("docs: add DSL Syntax Constraints"), 2026-08-07, branch `dsl`. <sup>[[1]](#ref-1)</sup>

**Method:** static source and git-history analysis only. Nothing was built or executed — the
toolchain requires CANN ≥ 9.1.0 and Ascend 950 hardware, which was unavailable. Every claim below
is a claim about source code or git history, not about observed runtime behavior.

---

## Contents

- [0. Executive Summary](#0-executive-summary)
- [1. Hard Facts](#1-hard-facts)
- [2. What CATLASS Is](#2-what-catlass-is)
- [3. What the `dsl` Branch Adds](#3-what-the-dsl-branch-adds)
- [4. Worked Examples: Vector, Cube, Mixed](#4-worked-examples-vector-cube-mixed)
- [5. Communication & Synchronization Primitives](#5-communication--synchronization-primitives)
- [6. `dsl` vs `master`](#6-dsl-vs-master)
- [7. Stability Assessment](#7-stability-assessment)
- [8. Activity Assessment](#8-activity-assessment)
- [9. Risks & Open Questions](#9-risks--open-questions)
- [10. How to Evaluate Further](#10-how-to-evaluate-further)
- [11. Appendix](#11-appendix)
- [12. References](#12-references)

---

## 0. Executive Summary

**What it is.** CATLASS ("CANN Templates for Linear Algebra Subroutines") is Huawei's open-source
operator template library for Ascend NPUs — the structural analogue of NVIDIA CUTLASS. It ships C++
templates for high-performance matmul and matmul-adjacent fused operators (Flash Attention,
grouped/quantized GEMM, conv2d). Open source since September 2025, currently at release v1.6.3,
with brisk mainline development (~30–70 commits/month). <sup>[[2]](#ref-2)</sup>

**What the `dsl` branch adds.** A Python frontend, `python/tla_dsl/`, for writing Ascend kernels in
Python instead of C++. It is *not* a wrapper over the C++ templates — it is an independent
compiler. Python source is traced into a custom **TLA MLIR dialect** (79 ops), run through ~23 TLA
lowering passes, converted to Huawei's **HIVM/HACC** IR via the AscendNPU-IR toolchain, then to LLVM
and a device binary — with JIT compilation, an on-disk artifact cache, and zero-copy PyTorch interop
via DLPack. Roughly 26k LOC of Python plus 19k LOC of C++/MLIR, with 23k LOC of
tests. <sup>[[3]](#ref-3)</sup> <sup>[[4]](#ref-4)</sup>

**Vector, cube, and mixed kernels.** All three are first-class, with working end-to-end examples.
Core regions open with `with tla.cube():` or `with tla.vector():`; a kernel containing both is
automatically classified MIX and split by a compiler pass. Vector code has two modes:
`tla.vec.func(mode="simd")` (explicit tile/register model) and `mode="simt"` (CUDA-like
`thread_idx()`, one thread per element). The SIMT path landed 2026-08-07 and currently has exactly
three ops. The cube path exposes the memory hierarchy explicitly: hand-allocated L1/L0A/L0B/L0C
buffers, hand-managed double buffering, explicit `tla.mmad()`. <sup>[[5]](#ref-5)</sup> <sup>[[6]](#ref-6)</sup> <sup>[[7]](#ref-7)</sup>

**Communication primitives.** There are **no inter-device or collective communication primitives
whatsoever** — `grep -ri hccl` returns zero hits across the entire repository on both `dsl` and
`master`. No all-reduce, all-gather, reduce-scatter, or all-to-all. This is a single-device kernel
DSL. What *does* exist is a rich set of **on-chip synchronization** primitives: pipe-to-pipe flags,
cross-core AIC↔AIV flags, named mutexes with an RAII-style guard, pipeline barriers, and GM atomic
accumulation. <sup>[[8]](#ref-8)</sup>

**Stability verdict: pre-release, unstable, but well-engineered.** No git tag contains
`python/tla_dsl` — the DSL has never shipped in a release. It is ~3 months old (first commit
2026-05-16, labeled "beta"). Hardware support is **Ascend 950PR/950DT only** — it does **not**
support the Atlas A2/A3 parts the C++ library targets. The project's own Q3 roadmap lists as *not
yet done*: 40+ SIMD ops needed for matmul/FA, NZ data format, FP8/MxFP8/MxFP4/int8 dtypes, tensor
subscript access, and **CI for the DSL**. <sup>[[9]](#ref-9)</sup> <sup>[[10]](#ref-10)</sup>

**Activity verdict: very high and accelerating.** 151 commits to `python/tla_dsl` from **13 distinct
authors**, no bus-factor concentration (top contributor 18%). Weekly commits climbed 2 → 8 → 18 → 31
over the last twelve weeks. A new commit landed on the branch *while the repository was being cloned
for this analysis*. <sup>[[11]](#ref-11)</sup>

**Recommendation: TRACK, do not adopt.** Engineering quality is real — MLIR-based architecture, test
LOC ≈ source LOC, a good English syntax-constraints document committed this week. But it is
unreleased, single-generation-hardware, has no CI, and delivers no code-size benefit yet. Revisit
when the DSL appears in a tagged release or when Atlas A2/A3 support lands.

**Top 3 risks**

1. **Hardware lock-in to an unreleased chip.** Ascend 950PR/950DT only. Without 950-series access,
   nothing here runs. The A2/A3 fleet is unsupported. <sup>[[9]](#ref-9)</sup>
2. **License is not OSI-approved.** "CANN Open Software License Agreement Version 2.0" — a
   Huawei-authored license (hence gitcode's `NOASSERTION`). Requires legal review before any
   commercial dependency. <sup>[[12]](#ref-12)</sup>
3. **No code-size win today.** DSL Flash Attention is 1,527 LOC versus 1,401 LOC for the equivalent
   C++ TLA example; basic matmul is 334 LOC (DSL) versus 148 LOC (C++). The DSL is currently *more*
   verbose than the thing it fronts. <sup>[[7]](#ref-7)</sup> <sup>[[13]](#ref-13)</sup>

---

## 1. Hard Facts

| Fact | Evidence |
|---|---|
| DSL is a separate subproject, not a branch-wide refactor | `git diff --numstat master...dsl` → 376 files, **+93,267 / −0**, all under `python/tla_dsl/` <sup>[[11]](#ref-11)</sup> |
| DSL never shipped in a release | `git cat-file -e vX:python/tla_dsl` fails for v1.4.0 through v1.6.3 <sup>[[11]](#ref-11)</sup> |
| Ascend 950 only | `SUPPORTED_ARCH_SCOPES = ("aiv.c310", "aic.c310")` <sup>[[9]](#ref-9)</sup> |
| Zero collective communication | `grep -ril hccl` → 0 hits, whole repo, both branches <sup>[[8]](#ref-8)</sup> |
| ~3 months old | first commit touching `python/tla_dsl`: `b22e826`, 2026-05-16, "ascend-catlass-DSL beta版本发布" <sup>[[14]](#ref-14)</sup> |
| `dsl` has fallen behind mainline | 154 ahead / **143 behind** `master`; merge-base `cc8edbd` (2026-05-13) <sup>[[11]](#ref-11)</sup> |
| No breaking change to the C++ API | only 4 files in `include/` touched; largest is a purely *additive* partial specialization <sup>[[15]](#ref-15)</sup> |
| 13 authors, healthy distribution | top contributor 27/151 commits = 18% <sup>[[11]](#ref-11)</sup> |
| No CI exists for the DSL | `.gitcode/` has only issue/PR templates; roadmap lists "支持DSL的CI" as a Q3 *goal* <sup>[[10]](#ref-10)</sup> |

---

## 2. What CATLASS Is

CATLASS is Huawei's Ascend equivalent of CUTLASS: a layered, template-based library for composing
high-performance GEMM-class kernels, with layering deliberately "white-box" so developers can
replace or locally modify individual stages rather than treating the operator as a monolith. <sup>[[2]](#ref-2)</sup>

- **Positioning.** The README claims 0.98×–1.2× of reference-operator performance on customized
  shapes. 66 numbered examples on `master` cover basic/batched/grouped matmul, quantized variants
  (W8A8, W4A8, W4A4), sparse matmul, StreamK, conv2d, and Flash Attention. <sup>[[2]](#ref-2)</sup>
- **TLA** is the layer the DSL is named after — CATLASS's tensor-layout algebra, the structural
  counterpart of CUTLASS's CuTe. It provides `Shape`/`Stride`/`Layout`/`Coord` composition and tiled
  tensor views, which is what makes the memory-hierarchy staging (GM → L1 → L0A/L0B → L0C → UB)
  expressible generically.
- **Target hardware.** `CATLASS_ARCH` selects the target: `2201` for Atlas A2/A3 training and
  inference parts, `3510` for next-generation Ascend 950PR/950DT. The 950 line was formally adopted
  as a mainline target at the March 2026 community meeting. <sup>[[2]](#ref-2)</sup>
- **Where the CUTLASS analogy breaks.** Ascend has two physically distinct core types — **AIC**
  (cube/matrix) and **AIV** (vector) — that must be explicitly programmed and explicitly
  synchronized with each other. There is no CUDA-style unified SIMT model at the hardware level.
  Data movement between GM, L1, L0A/L0B, L0C, and UB is issued through named pipes
  (MTE1/MTE2/MTE3/FIX/CUBE/VECTOR), and correctness depends on the programmer placing pipe-to-pipe
  flags correctly. This is the fundamental complexity the DSL inherits, and the central problem any
  Ascend Python DSL has to decide whether to expose or hide.

---

## 3. What the `dsl` Branch Adds

### 3.1 Architecture

`python/tla_dsl/` is a standalone Python project (own `pyproject.toml`, `setup.py`, `build.sh`,
`Dockerfile`, `mkdocs.yml`) with four parts: <sup>[[3]](#ref-3)</sup>

| Component | Size | Role |
|---|---|---|
| `catlass/` | ~25,900 LOC Python | Frontend: AST tracing, type system, MLIR building, runtime |
| `csrc/mlir/` | ~18,700 LOC C++ | TLA dialect definition, 23 lowering passes, `tla-compile` driver |
| `tests/` | ~23,500 LOC | 64 pytest modules + 113 lit (FileCheck) tests |
| `examples/end_to_end/` | ~13,600 LOC, 44 files | Runnable operator examples |
| `3rdparty/AscendNPU-IR` | submodule | Huawei's HIVM/HACC MLIR toolchain — the backend |

Largest single files: `catlass/core_api.py` (6,336 LOC, 57 public functions) and
`catlass/base_dsl/ast_preprocessor.py` (3,967 LOC — the Python AST rewriter). <sup>[[4]](#ref-4)</sup>

### 3.2 Programming model

Two decorators, documented in an English syntax-constraints guide: <sup>[[16]](#ref-16)</sup>

- `@tla.kernel` — device entry point; calling it returns a launcher.
- `@tla.jit` — a device sub-function callable from a kernel.

The frontend distinguishes **static** (compile-time) from **dynamic** (device-data-dependent)
values, which drives three different `for` constructs:

| Form | Semantics |
|---|---|
| `for i in tla.range(...)` | dynamic loop → real device loop instructions |
| `for i in range(...)` | compile-time unrolling (bound must be constant) |
| `for i in tla.range_constexpr(...)` | explicit compile-time unrolling |

Ordinary Python functions called inside a kernel are *not* compiled — they are evaluated at lowering
time as host code, producing a result only if their arguments are compile-time constants. Builtins
enter compilation via a whitelist (`any`/`all`/`bool`/`min`/`max`/`abs`/`pow`/`range`). This is
documented precisely, with file:line references into the implementation — a notably high-quality
document that landed in commit `c511c43`. <sup>[[16]](#ref-16)</sup>

### 3.3 The TLA dialect — 79 ops

```
Layout/tensor:  make_shape make_stride make_layout make_coord make_tensor
                make_tensor_like tile_view tensor_desc tensor_ptr squeeze
Memory:         alloc_ptr ptr_add recast_ptr inttoptr copy load store
                scalar_load scalar_store
Compute (cube): mmad
Compute (vec):  add sub mul div adds subs muls divs min max mins maxs
                abs neg exp log sqrt cast cmp where reduce arange full
                bitwise_{and,or,xor,not} gather interleave deinterleave
                create_mask update_mask
SIMT:           simt_add simt_load simt_store          ← only 3 ops
Regions:        cube vector func return
Sync:           flag set_flag wait_flag cross_flag cross_core_set_flag
                cross_core_wait_flag pipe_barrier local_mem_bar
                mutex mutex_lock mutex_unlock
Debug:          debug_print print_tensor
```

The SIMT op count (3) is the clearest single measure of how early that feature is. <sup>[[17]](#ref-17)</sup>

### 3.4 Compilation pipeline

Traced through `buildTlaPipeline` and the LLVM handoff: <sup>[[5]](#ref-5)</sup> <sup>[[18]](#ref-18)</sup>

```
Python source
  └─ AST rewrite            catlass/base_dsl/ast_preprocessor.py
  └─ trace to TLA dialect   catlass/core_api.py, catlass/_mlir_bindings/
       ↓
  TlaLowerFuncPass          ← classifies each function AIC / AIV / MIX from the
                              tla.cube / tla.vector regions it contains
  TlaInsertAutoMutexPass    ← the auto_sync feature
  TlaLowerPtrPass
  TlaSplitMixedFuncPass     ← splits a MIX kernel into separate AIC and AIV functions
  TlaLowerTensorDescPass, TlaLowerScalarAccessPass
  TlaVectorRegionPass, TlaCubeRegionPass
  TlaFinalizeMemrefPass, TlaLowerDebugPrintPass, TlaLowerBlockIdxPass
  TlaLowerFlagBarrierToHivmPass    ← flags become HIVM sync ops
  TlaLowerMutexToStdPass
  TlaPrologueEpiloguePass, CSE
       ↓
  HIVM / HIVM-AVE           (AscendNPU-IR: CombineAVEOPs, HIVMDecomposeOp,
                             ConvertHIVMToStandard, ConvertHIVMAVEToAVEIntrin)
       ↓
  LLVM dialect              (FuncToLLVM, FinalizeMemRefToLLVM, ArithToLLVM)
       ↓
  device binary (kernel.o)  — cached on disk, keyed by artifact.cache_key
```

Note this pipeline goes **TLA → HIVM → LLVM**, *not* TLA → AscendC source. That is a notable
contrast with TileLang-Ascend, which generates AscendC: Huawei's own DSL team bypasses AscendC as a
codegen target and lowers through AscendNPU-IR instead.

Host side: <sup>[[19]](#ref-19)</sup>

```python
tla.initialize(device=0)
t = from_dlpack(torch_tensor.contiguous(), layout_tag=tla.arch.RowMajor)
t = t.mark_compact_shape_dynamic(0)          # optional: reuse binary across shapes
artifact = tla.compile(kernel, t, ..., arch_scope="aiv.c310", cache=True)
artifact(t, ..., block_dim=block_dim)
torch.npu.synchronize()
```

DLPack conversion is **zero-copy** — the `tla.Tensor` shares device memory with the torch tensor,
and the source tensor must outlive the launch. Without `mark_*_dynamic`, shapes and strides are
burned into the compiled type; with it, one binary serves multiple shapes. <sup>[[19]](#ref-19)</sup>

---

## 4. Worked Examples: Vector, Cube, Mixed

### 4.1 Vector kernel

SIMD mode. The programmer stages GM → UB explicitly, tiles by vector length, and moves data through
registers: <sup>[[20]](#ref-20)</sup>

```python
@tla.kernel
def basic_vadd(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    n_ele = gm_a.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done  = tla.flag("vec_done",  tla.arch.VECTOR, tla.arch.MTE3)

    ub_ptr_a = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ...
    ub_a = tla.make_tensor_like(ub_ptr_a, gm_a, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub_a, gm_a); tla.copy(ub_b, gm_b)
        tla.set_flag(ub_loaded); tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                ub_vl_a = tla.tile_view(ub_a, tla.make_shape(VL_ELE), tla.make_coord(i))
                ...
                reg_c = tla.add(ub_vl_a.load(), ub_vl_b.load())
                ub_vl_c.store(reg_c)
        tla.set_flag(vec_done); tla.wait_flag(vec_done)
        tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)
```

`VL_ELE` is set per dtype by the host (64 for f32, 128 for f16, 256 for i8) — the vector length is
not abstracted away.

**SIMT mode**, from the commit that landed 2026-08-07, is dramatically simpler and looks like
CUDA: <sup>[[6]](#ref-6)</sup>

```python
@tla.kernel
def basic_vadd_simt(gm_a, gm_b, gm_c) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=VECTOR_ELE):
            tid, _, _ = tla.arch.thread_idx()
            thread_block_dim, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, VECTOR_ELE, thread_block_dim):
                gm_c[i] = gm_a[i] + gm_b[i]
        tla.pipe_barrier(tla.pipes.ALL)
```

No UB staging, no tiles, no flags — the compiler handles it. This is the most promising direction in
the project, and it is one day old with three supporting ops. SIMT buffers must be statically shaped
(only a pointer crosses the launch ABI), so `mark_compact_shape_dynamic` is unavailable there.

### 4.2 Cube kernel

The full memory hierarchy is the programmer's responsibility. The kernel declares **15 flags** by
hand to sequence a double-buffered GM→L1→L0→L0C pipeline: <sup>[[13]](#ref-13)</sup>

```python
l1a0_data_ready = tla.flag("l1a0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
l1a0_available  = tla.flag("l1a0_available",  tla.arch.MTE1, tla.arch.MTE2)
l0a0_available  = tla.flag("l0a0_available",  tla.arch.CUBE, tla.arch.MTE1)
l0_ab_data_ready= tla.flag("l0_ab_data_ready",tla.arch.MTE1, tla.arch.CUBE)
l0c_data_ready  = tla.flag("l0c_data_ready",  tla.arch.CUBE, tla.arch.FIX)
...   # 15 total, one per producer→consumer pipe pair per buffer

l1a0_ptr = tla.allocate(l1_tm*l1_tk, DTYPE_A, tla.AddressSpace.l1,  512)
l0a0_ptr = tla.allocate(l0_tm*l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
l0c_ptr  = tla.allocate(l0_tm*l0_tn, tla.Float32, tla.AddressSpace.l0c, 512)

with tla.cube():
    for block_linear in tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num()):
        ...
        l1_a = tla.make_tensor_like(l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1)
        tla.wait_flag(l1a0_available); tla.copy(l1_a, gm_a_by_l1); tla.set_flag(l1a0_data_ready)
```

Double buffering is manual (`l1a0_ptr` / `l1a1_ptr` selected by `l1_buf_idx`). Block scheduling is
manual (`tla.range(block_idx(), total, block_num())`).

**The escape hatch:** `@tla.kernel(auto_sync="v0")` removes all 15 flag declarations and every
set/wait, cutting the file from 334 → 259 lines. `TlaInsertAutoMutexPass` infers the dependencies.
The `"v0"` version string is a fair signal of maturity. <sup>[[13]](#ref-13)</sup>

### 4.3 Mixed vector+cube kernel

Both regions in one `@tla.kernel`. `TlaLowerFuncPass` classifies the kernel as MIX and
`TlaSplitMixedFuncPass` splits it into separate AIC and AIV device functions: <sup>[[21]](#ref-21)</sup>

```python
@tla.kernel
def basic_mixed(lhs, rhs, out, addend) -> None:
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    fix_done  = tla.cross_flag("fix_done")          # ← AIC → AIV, crosses cores

    with tla.cube():
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)   # zN / nZ layouts
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)
        tla.set_flag(mmad_done); tla.wait_flag(mmad_done)
        ub_c = tla.make_tensor_like(c_ub_ptr, l0_c, tla.arch.RowMajor)
        tla.copy(ub_c, l0_c, tla.params.CopyL0C2DstParams(
            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M))
        tla.cross_core_set_flag(fix_done, tla.arch.FIX)      # ── signal AIV
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()
        ub_c = tla.make_tensor_like(c_ub_ptr, gm_result, tla.arch.RowMajor)  # same UB buffer
        tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)  # ── wait for AIC
        for row_tile_idx in tla.range(0, VECTOR_TILE_M // VECTOR_REG_TILE_M, 1):
            with tla.vec.func(mode="simd"):
                result_chunk.store(c_chunk.load() + addend_chunk.load())
        tla.copy(gm_result, ub_result)
```

The AIC→AIV handoff: cube writes the mmad result L0C→UB, raises a cross-core flag on the FIX pipe;
vector waits on that flag, then reads **the same UB pointer** (`c_ub_ptr`) that cube wrote. Buffer
handoff is by shared pointer, ordering is by cross-core flag — no copy. `tla.arch.sub_block_idx()`
distinguishes the two AIV sub-blocks that pair with one AIC.

### 4.4 Boilerplate: DSL vs C++ — the DSL does not currently win

| Operator | C++ template path (`master`) | DSL path (`dsl`) |
|---|---|---|
| Basic matmul | 148 LOC <sup>[[22]](#ref-22)</sup> | 334 LOC manual sync / **259 LOC** with `auto_sync="v0"` <sup>[[13]](#ref-13)</sup> |
| Flash Attention (infer, TLA) | **1,401 LOC** across 4 files <sup>[[22]](#ref-22)</sup> | **1,527 LOC** <sup>[[7]](#ref-7)</sup> |

The DSL Flash Attention kernel declares roughly **60 synchronization flags by hand**, including
seven `tla.cross_flag(..., mode=4)` cross-core flags for the QK → softmax → PV pipeline. The DSL
exposes the same hardware complexity as the C++ path; it does not abstract it. This matches the
project's own README, which states that CATLASS-template-style higher-level wrapping is *future*
work. <sup>[[3]](#ref-3)</sup>

**So what is the DSL for, today?** Not brevity. Its value is Python-native iteration with JIT and
artifact caching, zero-copy PyTorch interop, `tla.print`/`print_tensor` on-device debugging,
msProf/Profiling integration for quick functional and performance validation before porting to a
production project, and — per the roadmap — a future `dsl-gen` backend for
torch.inductor. <sup>[[23]](#ref-23)</sup> <sup>[[10]](#ref-10)</sup>

---

## 5. Communication & Synchronization Primitives

### 5.1 What does not exist

**No collective or inter-device communication of any kind.** <sup>[[8]](#ref-8)</sup>

```
$ grep -ril 'hccl|allreduce|all_reduce|allgather|reduce_scatter|alltoall' .
(0 results — dsl branch, including 3rdparty)
$ git grep -il hccl origin/master
(0 results)
```

No HCCL binding, no all-reduce/all-gather/reduce-scatter/all-to-all, no communication-computation
overlap primitive. For distributed MoE, tensor-parallel GEMM, or any multi-die/multi-card workload,
**the communication half does not exist here** and must come from a separate layer. Consistent with
CATLASS's scope as a single-kernel template library, but worth stating plainly.

### 5.2 What does exist — all on-chip, all within one kernel launch

| Primitive | API | Purpose |
|---|---|---|
| Pipe-to-pipe flag | `tla.flag(name, src_pipe, dst_pipe)` + `set_flag` / `wait_flag` | Order producer/consumer across MTE1/MTE2/MTE3/CUBE/VECTOR/FIX pipes within one core |
| **Cross-core flag** | `tla.cross_flag(name, mode=N)` + `cross_core_set_flag(f, pipe)` / `cross_core_wait_flag(f, pipe)` | **AIC ↔ AIV synchronization.** `mode` selects sync topology; the FA example uses `mode=4`. Roadmap lists modes 1/2/4 as targets, so coverage is still being filled in. |
| Named mutex | `tla.mutex(resource="ub_a", id=0)`, `.lock(pipe=)` / `.unlock(pipe=)` | Guard a named on-chip resource, per-pipe |
| Mutex guard | `with tla.mutex_guard(m1, m2, m3):` | RAII-style, multi-resource |
| Auto-sync | `@tla.kernel(auto_sync="v0")` | Compiler infers all flags |
| Pipeline barrier | `tla.pipe_barrier(tla.pipes.ALL)` | Drain pipes |
| Local memory barrier | `tla.local_mem_bar` | On-chip memory ordering |
| **GM atomic accumulate** | `tla.copy(gm_c, ub_b, tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD))` | Cross-block accumulation into global memory — the only cross-*block* communication mechanism |
| Cross-block coordination | none beyond atomics | No grid-wide barrier; the atomic example avoids races by restricting work to `if tla.arch.block_idx() == 0` |

`examples/end_to_end/cross_flag_two_way/` demonstrates bidirectional AIC↔AIV flagging. <sup>[[24]](#ref-24)</sup>

Synchronization is **explicit, fine-grained, and hardware-shaped**. The DSL gives the same
primitives an Ascend C programmer has, in Python, with an opt-in inference pass. It gives nothing
above the single-device kernel.

---

## 6. `dsl` vs `master`

**Shape of the divergence** (`git diff --numstat origin/master...dsl`, rolled up): <sup>[[11]](#ref-11)</sup>

| Path | Files | Added | Deleted |
|---|---|---|---|
| `python/tla_dsl/` | 376 | +93,267 | −0 |
| `include/` | 4 | +220 | −10 |
| `tests/` | 3 | +1,010 | −41 |
| `docs/`, `.gitmodules`, `.gitignore`, `.pre-commit-config.yaml` | 4 | +12 | −10 |

**Effectively 100% additive.** The DSL is a bolt-on subproject.

**Breaking changes to the C++ template API: none found.** The four touched `include/` files are
`epilogue/tile/copy_{gm_to_ub,ub_to_gm,ub_to_l1}_tla.hpp` and a new
`gemm/tile/ascend950/copy_l0c_to_ub.hpp`. The largest change (`copy_ub_to_l1_tla.hpp`, +96) is a new
**partial template specialization** for `Arch::Ascend950` with RowMajor→zN, added alongside existing
ones. Existing specializations are untouched. <sup>[[15]](#ref-15)</sup>

**Integration status: `dsl` is a long-lived side branch that has fallen behind.** <sup>[[11]](#ref-11)</sup>

- Fork point: `cc8edbd`, **2026-05-13**.
- `dsl` is **154 commits ahead**, **143 commits behind** `master`.
- `master` HEAD is `a75e2ef` (2026-08-06), titled 【v2.0.0】部分样例迁移至experimental — mainline is
  preparing a **v2.0.0** with examples moving to `experimental/`.

`dsl` has not been rebased or merged in ~3 months while mainline did 143 commits including a
major-version restructuring. A future merge will not be free. The existence of `dsl_dev`, `tla_v2`,
and `notla` branches alongside suggests branch topology is still in flux. <sup>[[1]](#ref-1)</sup>

---

## 7. Stability Assessment

| Signal | Observation | Interpretation |
|---|---|---|
| **Release inclusion** | 15 tags exist (v1.0.0 … v1.6.3, latest 2026-07-31). `git cat-file -e <tag>:python/tla_dsl` fails for every one. <sup>[[11]](#ref-11)</sup> | **The DSL has never been released.** Strongest single stability signal. |
| **Age** | First commit 2026-05-16, "ascend-catlass-DSL beta版本发布". 151 commits total. <sup>[[14]](#ref-14)</sup> | ~3 months old. Self-described beta. |
| **Hardware scope** | `SUPPORTED_ARCH_SCOPES = ("aiv.c310", "aic.c310")`; README says Ascend950PR/950DT only, CANN ≥ 9.1.0, Python 3.10–3.13. <sup>[[9]](#ref-9)</sup> <sup>[[3]](#ref-3)</sup> | Single hardware generation — the *newest* one. No A2/A3. Much narrower than the C++ library. |
| **API churn** | Env-var prefix renamed `TLA_DSL_` → `CATLASS_DSL_` and **reverted the same day**. "Upgrade building system" also reverted. <sup>[[25]](#ref-25)</sup> | 2 reverts / 154 commits. Low absolute rate, but both in user-facing surface within 48 hours of analysis. Naming is not settled. |
| **Feature completeness** | Roadmap lists as still-needed: 40+ SIMD ops for Matmul/FA, NZ format, MxFP8/MxFP4/FP8/int8/int32, tensor subscript access, L0C2UB and UB2L1 paths, cross-core sync modes 1/2/4, printf/DumpTensor, Python-class tiling structs. <sup>[[10]](#ref-10)</sup> | Core capability for the flagship use case (Matmul/FA) is **explicitly incomplete by the maintainers' own account**. |
| **SIMT maturity** | 3 ops, landed 2026-08-07. <sup>[[17]](#ref-17)</sup> | Prototype. |
| **Auto-sync maturity** | `auto_sync="v0"` — versioned string parameter. <sup>[[13]](#ref-13)</sup> | Deliberately marked v0. |
| **CI** | `.gitcode/` contains only issue and PR templates. No workflow definition. Roadmap lists "支持DSL的CI" as a Q3 goal. <sup>[[10]](#ref-10)</sup> | **No automated testing gate on the DSL today.** |
| **Test breadth** | 64 pytest modules + 113 lit/FileCheck tests, ~23,500 LOC — roughly equal to source LOC. <sup>[[4]](#ref-4)</sup> | Genuinely strong test *authoring* discipline, undermined by having no CI to run it. |
| **Hygiene** | `.pre-commit-config.yaml`, `.clang-format`, `OWNERS`, `CONTRIBUTING.md`, `SECURITYNOTE.md`, `OAT.xml`, third-party SBOM present. Dockerfiles for A2, A3, and DSL dev env. <sup>[[1]](#ref-1)</sup> | Mature process scaffolding, inherited from the parent project. |
| **License** | `LICENSE` (root and `python/tla_dsl/`) = **CANN Open Software License Agreement Version 2.0**, Huawei-authored. gitcode reports `NOASSERTION`. <sup>[[12]](#ref-12)</sup> | **Not OSI-approved.** Needs legal review before commercial dependency. |
| **Documentation** | English syntax guide (327 lines, with implementation file:line citations) added 2026-08-07. Rest is Chinese. mkdocs site configured but roadmap says "搭建文档网站" is still a goal. <sup>[[16]](#ref-16)</sup> | Improving fast; English coverage is new and partial. |

---

## 8. Activity Assessment

**Window: 2026-05-16 (first DSL commit) → 2026-08-07 (analysis date), ~12 weeks.** <sup>[[11]](#ref-11)</sup>

Commits touching `python/tla_dsl`, by month:

| Month | Commits |
|---|---|
| 2026-05 (partial) | 5 |
| 2026-06 | 21 |
| 2026-07 | **99** |
| 2026-08 (7 days) | 26 |

By ISO week — the trend is the story:

```
W21  ██ 2
W22  ██ 2
W23  █████ 5
W24  ██ 2
W25  ████ 4
W26  ████████ 8
W27  ████████████ 12
W28  ██████████████████ 18
W29  ████████████████████ 20
W30  ███████████████████████████████ 31
W31  ████████████████████ 20
W32  ██████████████████████████ 26   (partial week)
```

Contributors to `python/tla_dsl` — **13 distinct authors**:

| Author | Commits | Share |
|---|---|---|
| tianxinghui | 27 | 18% |
| CheaterAbec | 27 | 18% |
| cann-robot | 22 | 15% |
| arusso | 17 | 11% |
| zjw666888 | 16 | 11% |
| yuantao_ / yuantao | 11 | 7% |
| mdrumond | 10 | 7% |
| weixin_42818618 | 7 | 5% |
| init__zhb__ | 6 | 4% |
| yjp-hw | 4 | 3% |
| WinstonSmith | 3 | 2% |
| lijiaming_hw | 1 | <1% |

**Bus factor is healthy** — no author exceeds 18%, five contributors have ≥10 commits. Author names
suggest a mixed Huawei-internal and international team. Caveat: `cann-robot` (15%) is an automated
account, and commit share is a proxy for knowledge distribution, not a measurement of it.

**Mainline (`master`) cadence** for comparison: 23 (Jan), 9 (Feb), 38 (Mar), 32 (Apr), 40 (May), 72
(Jun), 32 (Jul), 11 (Aug-partial). Release cadence: v1.4.0 (2026-02-09), v1.5.0 (2026-04-01), v1.6.0
(2026-06-30), v1.6.3 (2026-07-31) — roughly every 6–8 weeks with patch releases. Mainline is healthy
and independently active. <sup>[[11]](#ref-11)</sup>

**Live-development datapoint.** The branch HEAD advanced from `6c1e7e2` to `c511c43` between the
initial API query and the completion of `git clone` during this analysis.

**Ecosystem.** 445 stars, 253 forks, 44 open issues. 11 of 44 open issues concern the DSL — mostly
feature requests for specific DSL operators (`streamk matmul`, `matmul_evg` variants,
`grouped_matmul_slice_m`), indicating the DSL has users making requests, not just maintainers
building. The Q3 roadmap commits to DSL implementations of SplitK, StreamK, fullLoad,
GroupMatmulSliceM, PFA, KDA, and BSA — all Ascend 950. <sup>[[2]](#ref-2)</sup> <sup>[[10]](#ref-10)</sup>

**Verdict:** activity is high, accelerating, well-distributed, and directed by a published roadmap.
Not an abandoned or single-maintainer effort.

---

## 9. Risks & Open Questions

1. **Hardware availability (blocking).** Ascend 950PR/950DT only. On A2/A3 not a single DSL example
   runs.
2. **License.** CANN Open Software License Agreement v2.0 is Huawei-authored, not OSI-approved.
   Legal review required for commercial use, redistribution, or derivative work.
3. **Unreleased and unversioned.** `__version__` falls back to `"0.0.0"` in a source tree. No
   release contains the DSL. No deprecation policy. Any API can change without notice — two
   user-facing changes were reverted the week of analysis.
4. **No CI.** ~23,500 LOC of tests with no automated gate running them. This is a stated Q3 goal.
5. **Branch divergence.** `dsl` is 143 commits behind a `master` restructuring for v2.0.0. Unclear
   whether `dsl`, `dsl_dev`, or `tla_v2` is the intended integration path.
6. **No abstraction win yet.** Currently more verbose than the C++ path it fronts.
7. **No distributed story.** Zero communication primitives.
8. **Toolchain lock-in.** Hard dependency on CANN ≥ 9.1.0, `bisheng`/`hivmc`, and the AscendNPU-IR
   submodule. Recommended path is a Docker image (`cann:9.1.0-950-ubuntu22.04-py3.12`).
9. **Documentation language.** Most docs are Chinese; the English syntax guide is one file, days old.

**Open questions not answerable statically:**

- Does DSL-generated code reach the C++ templates' claimed 0.98×–1.2× performance? No benchmark
  comparing the two paths exists in the repo.
- What is JIT compile latency, and how effective is the artifact cache in practice?
- Is `dsl` or `dsl_dev` the integration branch? (`dsl_dev` head `2ef01e70`, `tla_v2` head `2b467f88`
  — not analyzed in depth.)

---

## 10. How to Evaluate Further

**Prerequisites:** Linux host with CANN Toolkit ≥ 9.1.0, Python 3.10–3.13, an Ascend 950PR or 950DT
device, `torch` + `torch_npu`, Clang ≥ 10 (19 recommended) if building AscendNPU-IR manually. <sup>[[3]](#ref-3)</sup>

```bash
cd catlass/python/tla_dsl
bash build_docker_image.sh cann:9.1.0-950-ubuntu22.04-py3.12
# mount the full CATLASS repo into the container, then:
export TLA_DSL_PREBUILT_ASCENDNPU_IR="$PWD/3rdparty/AscendNPU-IR"
./build.sh
python -m pytest -q tests
```

**Concrete next experiments, in priority order:**

1. `basic_vadd.py` — smoke test the whole chain.
2. `basic_vadd_simt.py` — assess whether the SIMT model is usable enough to bet on. Highest-leverage
   question for productivity.
3. `basic_matmul.py --dtype f16` and the `auto_sync` / mutex variants — measure what `auto_sync="v0"`
   costs in performance versus hand-placed flags.
4. `flash_attention_infer.py`, then profile with `msprof op` against
   `examples/49_ascend950_flash_attention_infer` (C++) at identical shapes. **Decisive:** does the
   DSL cost performance?
5. Time cold vs. warm `tla.compile()` to quantify JIT latency and cache effectiveness.
6. Run the lit suite (113 tests) — testable **without an NPU device**, the cheapest way to gauge
   compiler health.

**Trigger conditions to re-evaluate:** the DSL appearing in a tagged release; Atlas A2/A3 support
landing; DSL CI going live; `dsl` merging to `master`.

---

## 11. Appendix

### A. Repository metadata (2026-08-07)

| | |
|---|---|
| Repo | `cann/catlass` <sup>[[1]](#ref-1)</sup> |
| Owner | CANN (Huawei) |
| Created | 2025-09-23 |
| Default branch | `master` (HEAD `a75e2ef`, 2026-08-06) |
| Analyzed branch | `dsl` (HEAD `c511c43`, 2026-08-07) |
| Stars / Forks / Open issues | 445 / 253 / 44 |
| Latest tag | v1.6.3 (2026-07-31) |
| License | CANN Open Software License Agreement Version 2.0 <sup>[[12]](#ref-12)</sup> |
| Other DSL branches | `dsl_dev`, `tla_v2`, `notla` |

### B. Glossary

| Term | Meaning |
|---|---|
| **CATLASS** | CANN Templates for Linear Algebra Subroutines — the C++ operator template library |
| **TLA** | CATLASS's tensor-layout algebra layer; structural analogue of CUTLASS's CuTe. The DSL is named after it |
| **AIC** | AI Cube core — matrix-multiply unit. `with tla.cube():` |
| **AIV** | AI Vector core — SIMD/vector unit. `with tla.vector():`. Two AIV sub-blocks pair with one AIC |
| **MIX** | A kernel containing both cube and vector regions; split by `TlaSplitMixedFuncPass` |
| **GM / L1 / L0A / L0B / L0C / UB** | Ascend memory hierarchy: global memory; L1 cache; matmul A/B operand buffers; matmul accumulator; Unified Buffer (vector working memory) |
| **Pipes** | MTE1 (L1→L0), MTE2 (GM→L1/UB), MTE3 (UB→GM), CUBE, VECTOR, FIX (fixpipe). Sync flags are declared as (src_pipe → dst_pipe) pairs |
| **FixPipe** | Post-matmul fixed-function stage: quantization, format conversion, L0C→UB/GM output |
| **zN / nZ / L0Clayout** | Ascend fractal tile layouts required by the cube unit |
| **SIMD mode** | `tla.vec.func(mode="simd")` — explicit tiles, registers, per-dtype vector length |
| **SIMT mode** | `tla.vec.func(mode="simt", thread_block_dim=N)` — CUDA-like per-thread model |
| **AscendNPU-IR / HIVM / HACC** | Huawei's MLIR-based NPU compiler infrastructure; the DSL's backend |
| **bisheng / hivmc** | Huawei's device compilers |
| **CATLASS_ARCH** | C++ target macro: `2201` = Atlas A2/A3, `3510` = Ascend 950PR/950DT |
| **arch_scope** | DSL target string: `aiv.c310` or `aic.c310` (c310 = Ascend950PR). Only these two |
| **cross_flag / mode** | AIC↔AIV synchronization flag; `mode` selects sync topology (1/2/4) |
| **auto_sync** | `@tla.kernel(auto_sync="v0")` — compiler-inferred pipe synchronization |
| **DLPack** | Cross-framework zero-copy tensor exchange protocol |

---

## 12. References

All source paths are relative to the CATLASS repository root on branch `dsl` at commit `c511c43`.
Statistics marked "git" are reproducible with the command shown, from a clone of the repo.

| # | Description | URL / Command |
|---|-------------|---------------|
| <a name="ref-1"></a>[1] | CATLASS repository, branch `dsl` @ `c511c43`; branch list including `dsl_dev`, `tla_v2`, `notla`; repo metadata (445 stars, 253 forks, 44 open issues, created 2025-09-23) | https://gitcode.com/cann/catlass/tree/dsl · `git ls-remote --heads https://gitcode.com/cann/catlass.git` |
| <a name="ref-2"></a>[2] | CATLASS root README — project description, CUTLASS positioning, 0.98×–1.2× performance claim, `CATLASS_ARCH` values (2201 = A2/A3, 3510 = Ascend 950), release history, March 2026 community-meeting decision on 950 support | https://gitcode.com/cann/catlass/blob/dsl/README.md |
| <a name="ref-3"></a>[3] | CATLASS DSL README — MLIR/AscendNPU-IR architecture statement, compatibility matrix (Ascend950PR/950DT, CANN ≥ 9.1.0, Python 3.10–3.13, Clang ≥ 10), directory overview, build instructions, statement that CATLASS-template-style wrapping is future work | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/README.md |
| <a name="ref-4"></a>[4] | Code size measurements: `catlass/` ~25,900 LOC Python; `csrc/` ~18,700 LOC C++; `tests/` ~23,500 LOC (64 pytest modules, 113 lit tests); `examples/` ~13,600 LOC in 44 files; `core_api.py` 6,336 LOC / 57 public functions; `ast_preprocessor.py` 3,967 LOC | git: `find python/tla_dsl -name '*.py' \| xargs wc -l` |
| <a name="ref-5"></a>[5] | `buildTlaPipeline` — the complete ordered TLA pass pipeline (23 passes, TlaLowerFuncPass through ConvertSCFToCF), including AIC/AIV/MIX classification and HIVM lowering | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/csrc/mlir/lib/Passes/PassRegistry.cpp#L43 |
| <a name="ref-6"></a>[6] | SIMT vector-add example — `mode="simt"`, `thread_idx()`, `thread_block_dim`, direct `gm_c[i] = gm_a[i] + gm_b[i]`; note that SIMT buffers must be statically shaped. Added by commit `6c1e7e2` (2026-08-07) | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/examples/end_to_end/simt/basic_vadd_simt.py |
| <a name="ref-7"></a>[7] | DSL Flash Attention inference example — 1,400 LOC kernel + 127 LOC tiling = 1,527 LOC; ~60 hand-declared sync flags including seven `tla.cross_flag(..., mode=4)` | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/examples/end_to_end/flash_attention_infer/flash_attention_infer.py |
| <a name="ref-8"></a>[8] | Absence of collective communication: zero matches for HCCL / all-reduce / all-gather / reduce-scatter / all-to-all across the whole repository, on both `dsl` (including `3rdparty/`) and `master` | git: `grep -ril 'hccl\|allreduce\|all_reduce\|allgather\|reduce_scatter\|alltoall' .` → 0 results; `git grep -il hccl origin/master` → 0 results |
| <a name="ref-9"></a>[9] | `SUPPORTED_ARCH_SCOPES = ("aiv.c310", "aic.c310")` and `DEFAULT_ARCH_SCOPE = "aiv.c310"` — the hardware constraint, Ascend 950 only | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/catlass/execution.py#L46-L47 |
| <a name="ref-10"></a>[10] | CATLASS 2026 Q3 RoadMap (issue #399, opened 2026-08-04) — lists as not-yet-done: 40+ SIMD ops for Matmul/FA, NZ format, MxFP8/MxFP4/FP8/int8/int32, tensor subscript access, L0C2UB and UB2L1 paths, cross-core sync modes 1/2/4, printf/DumpTensor, Python-class tiling structs, **DSL CI**, `dsl-gen` inductor backend, documentation website; planned DSL operators SplitK/StreamK/fullLoad/GroupMatmulSliceM/PFA/KDA/BSA | https://gitcode.com/cann/catlass/issues/399 |
| <a name="ref-11"></a>[11] | Git history statistics: 151 commits to `python/tla_dsl` from 13 authors; monthly and weekly cadence; 154 ahead / 143 behind `master`; merge-base `cc8edbd` (2026-05-13); `master` HEAD `a75e2ef` (2026-08-06); diff rollup +93,267/−0; tag list v1.0.0–v1.6.3 with none containing `python/tla_dsl` | git: `git log --format='%an' -- python/tla_dsl \| sort \| uniq -c`; `git rev-list --count origin/master..dsl`; `git diff --numstat origin/master...dsl`; `git cat-file -e <tag>:python/tla_dsl` |
| <a name="ref-12"></a>[12] | CANN Open Software License Agreement Version 2.0 — the license text at both repository root and `python/tla_dsl/`; Huawei-authored, not OSI-approved (gitcode API reports `NOASSERTION`) | https://gitcode.com/cann/catlass/blob/dsl/LICENSE |
| <a name="ref-13"></a>[13] | DSL basic matmul examples — `basic_matmul.py` 334 LOC with 15 hand-declared flags and manual double buffering; `basic_matmul_auto_sync.py` 259 LOC using `@tla.kernel(auto_sync="v0")` with zero flags; mutex and atomic-add variants | https://gitcode.com/cann/catlass/tree/dsl/python/tla_dsl/examples/end_to_end/basic_mmad |
| <a name="ref-14"></a>[14] | First commit introducing the DSL: `b22e826`, 2026-05-16, "ascend-catlass-DSL beta版本发布" | git: `git log --reverse --format='%h %ad %an \| %s' -- python/tla_dsl \| head -1` |
| <a name="ref-15"></a>[15] | C++ template changes on `dsl`: 4 files in `include/` (+220/−10); largest is an additive partial specialization of `CopyUb2L1Tla` for `Arch::Ascend950` RowMajor→zN; existing specializations untouched | git: `git diff origin/master...dsl -- include/` |
| <a name="ref-16"></a>[16] | CATLASS DSL Syntax Constraints (English) — `@tla.kernel` vs `@tla.jit`, static/dynamic value semantics, the three `for` forms, builtin whitelist, dynamic-region constraints. Added by commit `c511c43` (2026-08-07) | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/docs/dsl_python_syntax_guide_en.md |
| <a name="ref-17"></a>[17] | TLA MLIR dialect op inventory — 79 ops total, of which exactly 3 are SIMT (`tla.simt_add`, `tla.simt_load`, `tla.simt_store`) | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/catlass/_mlir_bindings/tla_ops_gen.py · git: `grep -c 'OPERATION_NAME = "tla\.' catlass/_mlir_bindings/tla_ops_gen.py` |
| <a name="ref-18"></a>[18] | `buildTlaCompilePassManagers` — TLA pipeline to LLVM handoff (ConvertFuncToLLVM, FinalizeMemRefToLLVM, ArithToLLVM, ReconcileUnrealizedCasts) | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/csrc/mlir/lib/Tools/CompilePipeline.cpp#L839 |
| <a name="ref-19"></a>[19] | Torch tensor integration via DLPack — `from_dlpack`, zero-copy semantics and lifetime requirement, `mark_layout_dynamic` / `mark_compact_shape_dynamic`, `tla.compile` / launch flow | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/docs/framework_integration.md |
| <a name="ref-20"></a>[20] | DSL vector-add example (SIMD mode) — explicit UB allocation, `tla.flag` pipe sync, `tla.tile_view` tiling, per-dtype `VL_ELE` (64 f32 / 128 f16 / 256 i8); mutex, `mutex_guard`, and atomic-add variants in the same file | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/examples/end_to_end/basic_vadd/basic_vadd.py |
| <a name="ref-21"></a>[21] | Mixed cube+vector example — `tla.cross_flag` / `cross_core_set_flag` / `cross_core_wait_flag` AIC→AIV handoff via shared UB pointer, `sub_block_idx()`, zN/nZ/L0Clayout, `CopyL0C2DstParams(l0c2ub_mode=SPLIT_M)` | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/examples/end_to_end/basic_mixed/basic_mixed.py |
| <a name="ref-22"></a>[22] | C++ comparison baselines on `master`: `00_basic_matmul/basic_matmul.cpp` 148 LOC; `40_flash_attention_infer_tla/` 1,401 LOC across 4 files (fai.cpp 314, fai_kernel.cpp 775, fai_tiling.cpp 155, kernel_common.hpp 157) | https://gitcode.com/cann/catlass/tree/master/examples/00_basic_matmul · https://gitcode.com/cann/catlass/tree/master/examples/40_flash_attention_infer_tla |
| <a name="ref-23"></a>[23] | CATLASS_DSL functional and performance debugging guide — msProf single-operator and Profiling whole-network analysis; states the DSL project is intended for fast functional/performance validation before migrating to a production project | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/docs/evaluation.md |
| <a name="ref-24"></a>[24] | Bidirectional AIC↔AIV cross-flag example | https://gitcode.com/cann/catlass/tree/dsl/python/tla_dsl/examples/end_to_end/cross_flag_two_way |
| <a name="ref-25"></a>[25] | API-churn evidence: commit `c247a7b` renamed env prefix `TLA_DSL_` → `CATLASS_DSL_`, reverted same day by `5bd56de`; `604d916` reverted "【DSL】Upgrade building system" | git: `git log --oneline origin/master..dsl \| grep -i revert` |
