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
