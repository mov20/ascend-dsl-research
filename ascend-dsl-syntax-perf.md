# Table 2: Syntax Constructs & Performance Impact

*Last updated: 2026-03-30*

## Kernel Primitives

| Construct | Triton | TileLang | Pallas | cuTile | Helion | Gluon | Mojo | Perf impact |
|---|---|---|---|---|---|---|---|---|
| **Kernel definition** | `@triton.jit` decorator | `T.Kernel(...)` context | `@pl.kernel` + `BlockSpec` | `@ct.kernel` decorator | `@helion.kernel()` decorator | `@triton.jit` (extended) | `fn` with SIMD types | Minimal — boilerplate |
| **Thread/block indexing** | `tl.program_id(axis)` | `T.Kernel(bx, by, threads)` | `pl.program_id()` | `ct.bid(axis)` | Implicit (tile loops) | `tl.program_id` + warp_id | Manual thread mgmt | Low — compiler handles |
| **Tile/Block loop** | Manual (`for` + range) | `T.Pipelined(K, ...)` | `grid` in `BlockSpec` | None (single tile per block) | `hl.tile(dim)` loop | Manual | Manual loop + vectorize | **High** — determines data reuse |
| **Data load** | `tl.load(ptr, mask)` | `T.alloc_fragment()` + copy | `pl.load(ref, idx)` | `ct.load(tensor, idx, shape)` | `x[tile_m, tile_k]` (PyTorch indexing) | `tl.load` + warp-aware | Manual pointer + SIMD load | **High** — coalescing, bank conflicts |
| **Data store** | `tl.store(ptr, val, mask)` | Store via fragment | `pl.store(ref, idx, val)` | `ct.store(tensor, idx, tile)` | `out[tile_m, tile_n] = val` | `tl.store` | Manual SIMD store | **High** — write coalescing |
| **Matrix multiply** | `tl.dot(a, b)` | `T.gemm(a, b, c)` | `jax.lax.dot` | `a + b` (operator overload) | `torch.addmm(acc, a, b)` | `tl.dot` + explicit layout | `matmul` with tiling | **Critical** — Tensor/Cube core util |

## Memory Management

| Construct | Triton | TileLang | Pallas | cuTile | Helion | Gluon | Mojo | Perf impact |
|---|---|---|---|---|---|---|---|---|
| **Memory hierarchy** | Implicit (compiler) | Explicit: `alloc_fragment` (reg), `alloc_shared` (SRAM) | Explicit: `BlockSpec` + `prefetch` | Implicit (compiler auto) | Implicit (compiler) | Explicit: shared mem alloc | Manual: stack/heap/SIMD reg | **Critical** — bandwidth bottleneck |
| **Shared mem / SRAM** | Auto-managed | `T.alloc_shared()` | `pltpu.SMEM` / manual | Auto (Tile IR) | Auto (via Triton) | `tl.allocate_shared()` | Manual `Pointer` | **High** — data locality |
| **Prefetch / double buffer** | None (compiler) | `T.Pipelined(num_stages=N)` | `prefetch_scoped()` (TPU) | Auto | Auto | Manual | Manual | **Critical** — hide latency |
| **Memory coalescing** | Auto (compiler) | Auto | Auto + hints | Auto | Auto | Explicit (warp layout) | Manual (vectorize) | **High** — bandwidth util |

## Scheduling & Optimizations

| Construct | Triton | TileLang | Pallas | cuTile | Helion | Gluon | Mojo | Perf impact |
|---|---|---|---|---|---|---|---|---|
| **Tiling strategy** | Manual (BLOCK_M/N/K) | Declarative (`T.Kernel` params) | `BlockSpec(block_shape)` | Auto (compiler) | Auto (autotuner) | Manual | Manual | **Critical** — occupancy, reuse |
| **Autotuning** | `triton.autotune` decorator | Config search | None (manual) | None (auto-optimal) | Built-in autotuner | None | None | **High** — 2-5x difference |
| **Operator fusion** | Manual (write fused kernel) | None (kernel-level) | JAX fusion | None (kernel-level) | TorchInductor fusion | Manual | Manual | **High** — reduce mem traffic |
| **Pipeline / async** | `tl.async_copy` (limited) | `T.Pipelined(num_stages)` | `async_copy` (TPU) | Auto async | Auto | Manual | Manual | **Critical** — compute/mem overlap |
| **Warp/wave specialization** | None | None | None | None | None | ✅ `@warp_specialize` | Manual thread control | **High** — unlock peak for complex kernels |

## Ascend-Specific Constructs (TileLang-Ascend)

| Construct | Syntax | Perf impact | GPU DSL equivalent |
|---|---|---|---|
| **L1/UB buffers** | `alloc_L1()`, `alloc_ub()` | **Critical** — on-chip memory | `alloc_shared` (TileLang GPU) |
| **Cross-core comm** | `T.set_cross_flag()`, `T.wait_cross_flag()` | **High** — multi-core sync | No direct equivalent |
| **Vectorization** | `T.add()`, `T.mul()` (explicit) | **High** — Vector Unit util | Auto-vectorize in GPU DSLs |
| **CopyIn→Compute→CopyOut** | Explicit pipeline stages | **Critical** — Ascend execution model | `T.Pipelined` (TileLang GPU) |

## Summary: Syntax → Performance Impact

| Impact | Pattern | Best implementation | Notes for our Ascend DSL |
|--------|---------|---------------------|--------------------------|
| 🔴 **Critical** | Tiling + block sizes | Helion autotuner, cuTile auto | Need autotuner for Cube/Vector tile sizes |
| 🔴 **Critical** | Memory hierarchy control | TileLang explicit alloc, Pallas prefetch | Ascend: L0→L1→L2→HBM, need explicit control with auto defaults |
| 🔴 **Critical** | Double buffering / pipeline | TileLang `Pipelined(num_stages)` | Key for Ascend — Cube/Vector pipeline overlap |
| 🔴 **Critical** | Compute unit mapping | Not in any GPU DSL | **Unique to Ascend**: Cube vs Vector vs Scalar routing |
| 🟡 **High** | Autotuning | Helion, Triton autotune | Required for 90% peak |
| 🟡 **High** | Warp/core specialization | Gluon `@warp_specialize` | Ascend: multi-AI-core scheduling |
| 🟡 **High** | Data loading syntax | Helion (PyTorch indexing) | Simplicity: `x[tile_m, tile_k]` > `tl.load(ptr)` |
| 🟢 **Medium** | Kernel definition | All ~equivalent | Decorator-based (`@kernel`) |
| 🟢 **Medium** | Fusion | Helion (TorchInductor) | Graph-level — phase 2 |

---

## Key Challenges for Ascend NPU

Ascend 910B AI Core consists of two types of compute cores:
- **AIC** (AI Core / Cube) — matrix/tensor operations
- **AIV** (AI Vector) — vector operations; ratio is 1 AIC : 2 AIV on 910B

Each core (AIC and AIV) has its own: **MTE** (Memory Transfer Engine, DMA),
**Scalar Unit** (control flow, address calculation, instruction dispatch), and
**Compute Unit** (Cube or Vector). AIC and AIV have no direct interconnect —
data exchange goes through L2 / Global Memory.

Three challenges arise from this architecture that GPU DSLs don't face, or solve differently.

---

### Challenge 1 — Sync Insertion Between MTE and Compute Unit

Within each core, MTE (data load) and Compute (execution) run in parallel.
Explicit `set_flag`/`wait_flag` barriers must be inserted between them.
Wrong or missing barriers cause silent data hazards; extra barriers cause stalls.

| DSL | How sync is handled | User-visible? |
|-----|-------------------|---------------|
| **Triton** | `__syncthreads` auto-inserted at shared mem read/write boundaries | No |
| **TileLang GPU** | `T.Pipelined` handles barriers between stages | No |
| **TileLang-Ascend** | `T.set_cross_flag()` / `T.wait_cross_flag()` — explicit per stage | **Yes — manual** |
| **Pallas (TPU)** | XLA/Mosaic inserts barriers automatically | No |
| **Helion** | Inherits Triton backend — fully implicit | No |
| **cuTile** | Fully automatic (compiler-scheduled) | No |

**Key insight:** Every GPU DSL hides sync from the user via compiler analysis of
data flow. TileLang-Ascend is the only exception — it exposes hardware barriers
directly. The open question: can producer/consumer relationships on Ascend be
inferred from tile data flow alone, without user annotations?

---

### Challenge 2 — Double Buffering (Ping-Pong)

To hide HBM→UB load latency, MTE loads for tile N+1 must overlap with Compute
on tile N. This requires splitting the loop body into a preload phase and a
compute+async-load phase — and inserting correct sync barriers between them.

| DSL | Double buffer support | Syntax | Automation level |
|-----|-----------------------|--------|-----------------|
| **Triton** | Partial | `tl.async_copy` + manual barrier | Semi-manual |
| **TileLang GPU** | ✅ | `T.Pipelined(K, num_stages=2)` | Declarative |
| **TileLang-Ascend** | Partial | `T.Pipelined(K, num_stages=2)` (limited on Ascend) | Partial |
| **Pallas (TPU)** | ✅ | `pl.prefetch_scoped(ref, ...)` | Semi-manual |
| **Helion** | Compiler best-effort | None — fully implicit | Auto via Triton |
| **cuTile** | ✅ | None — fully auto | Fully automatic |

**Key insight:** TileLang's `T.Pipelined(num_stages=N)` is the most portable
declarative model. cuTile/Helion go further — fully automatic. On Ascend, ping-pong
is harder than on GPU because it must interact with sync insertion: the compiler
must simultaneously split the loop, hoist loads, and insert barriers correctly.

---

### Challenge 3 — On-Chip Memory (UB) Allocation and Reuse

Each AIC and AIV core has its own UB (Unified Buffer) — 256 KB on 910B.
After loop unrolling, many tile values can be live simultaneously, potentially
exhausting UB. Reuse requires liveness analysis across unrolled iterations.

| DSL | On-chip mem model | Liveness analysis | Region reuse |
|-----|------------------|------------------|--------------|
| **Triton** | Fully implicit | Yes (MLIR) | Yes |
| **TileLang GPU** | `T.alloc_shared()` explicit; compiler infers lifetime | Yes | Partial |
| **TileLang-Ascend** | `alloc_ub()` explicit; no compiler reuse | No — user manages | ❌ Manual |
| **Pallas (TPU)** | `BlockSpec` declares sizes; XLA assigns regions | Yes | Yes |
| **Helion** | Fully implicit via Triton | Yes | Yes |
| **cuTile** | Fully implicit (Tile IR) | Yes | Yes |

**Key insight:** Triton, Helion, cuTile rely on compiler liveness analysis to pack
tiles into shared mem — user only declares tile shapes. TileLang-Ascend is the
outlier: fully manual, no reuse. For a new Ascend DSL, the implicit model is
the right target: user declares tile shapes; compiler handles UB layout.
