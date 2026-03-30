# Table 2: Syntax Constructs & Performance — Ascend Focus

*Last updated: 2026-03-30 | Source: PyAsc2 HLD (gitcode.com/compiler-team/pyasc/pull/99)*

This document maps the **four key compiler challenges** for Ascend NPU kernels (from the PyAsc2
design doc) to concrete syntax and compiler strategies across existing DSLs. The goal: understand
what solutions already exist in GPU DSLs and where PyAsc2 must do original work.

---

## Four Key Challenges

Ascend NPU has a fundamentally different execution model from GPUs. Where a GPU SM schedules
warps dynamically, Ascend's AI Core requires explicit management of:

1. **Programming model** — TPipe/TQue lifecycles must be hidden from the user
2. **Sync insertion** — `set_flag`/`wait_flag` between pipeline stages must be auto-inserted
3. **Ping-pong (double buffering)** — load for tile N+1 must overlap compute on tile N, automatically
4. **UB memory allocation** — SSA tile values after loop unrolling must fit in finite on-chip UB

These map directly to the performance gap: if any of the four is wrong, the kernel either
produces incorrect results or runs far below peak FLOPS.

---

## Challenge 1 — Programming Model: Hiding TPipe/TQue

### What Ascend requires (raw AscendC)

```cpp
// AscendC: developer manually creates pipe, queues, events
TPipe pipe;
TQue<QuePosition::VECIN, 2> inQueueA, inQueueB;
TQue<QuePosition::VECOUT, 1> outQueue;
pipe.InitBuffer(inQueueA, 2, TILE_LENGTH * sizeof(half));
// ... manual AllocTensor / FreeTensor / EnQue / DeQue on every op
```

Every kernel must manage buffer lifecycles explicitly. This is the main source of boilerplate
and bugs in raw AscendC code.

### How GPU DSLs handle the equivalent

| DSL | Approach | What user writes | What's hidden |
|-----|----------|-----------------|---------------|
| **Triton** | Fully implicit | `x = tl.load(ptr + offsets, mask=mask)` | All SRAM allocation, barriers |
| **TileLang GPU** | Declarative tiles | `A = T.alloc_fragment((BM, BK), dtype)` | TQue equiv., sync between stages |
| **TileLang-Ascend** | Semi-explicit | `alloc_L1(...)`, `alloc_ub(...)`, manual `set_cross_flag` | Nothing — user writes all of it |
| **Pallas (TPU)** | Declarative | `BlockSpec(block_shape=(BM, BK), memory_space=pltpu.SMEM)` | HBM↔SRAM DMA, barriers |
| **Helion** | Fully implicit | `a = x[tile_m, tile_k]` (PyTorch indexing) | All allocation, barriers, sync |

TileLang-Ascend is the only existing DSL with an Ascend backend — and it exposes all
hardware details to the user.

### PyAsc2 approach

Two types hide the hardware: `Tensor` (global memory / HBM descriptor) and `Tile` (on-chip value):

```python
@asc2.jit
def vadd(x_ptr, y_ptr, out_ptr, size: int, TILE: asc.ConstExpr[int]):
    x_gm   = asc2.tensor(x_ptr,   [size])   # Tensor: HBM descriptor
    y_gm   = asc2.tensor(y_ptr,   [size])
    out_gm = asc2.tensor(out_ptr, [size])

    base = asc2.block_idx() * asc2.num_tiles(x_gm, 0, [TILE]) * TILE
    for i in asc2.range(asc2.num_tiles(x_gm, 0, [TILE])):
        off = base + i * TILE
        x = asc2.load(x_gm, [TILE], offsets=[off])   # Tile: UB value
        y = asc2.load(y_gm, [TILE], offsets=[off])
        asc2.store(x + y, out_gm, offsets=[off])
```

Compiler generates: `TPipe` + `TQue` + `InitBuffer` + `AllocTensor`/`FreeTensor`/`EnQue`/`DeQue`
— entirely invisible to the user.

**Two memory management strategies** (`CompileOptions`):

| Strategy | `static_alloc` | How it works |
|----------|---------------|-------------|
| TPipe-managed (default) | `False` | `MaterializeTensor` pass emits `AllocTensor`/`FreeTensor`; flexible, small scalar overhead |
| Static allocation | `True` | `AllocateTensor` pass computes fixed UB layout at compile time; zero overhead |

**Key design decision:** `Tile` has **value semantics** in MLIR (`ValueSemantics` trait) — tiles are
SSA values, never mutated. The compiler is free to CSE, copy, and remap them to physical UB regions.

**Open questions** (noted in HLD §2.1): ergonomics of multi-dimensional tiling, mixed `UB`/`L0`
tile locations, interoperability with raw `asc` (PyAsc1) primitives.

---

## Challenge 2 — Sync Insertion: `set_flag` / `wait_flag`

### What Ascend requires

Ascend AI Core has separate Cube (matrix), Vector, and MTE (memory transfer) units running
in parallel. Between stages, explicit sync barriers are required:

```cpp
// AscendC: manual sync between MTE2 (load) and Vector (compute)
pipe.SetEventId(EVENT_ID0);   // MTE2 signals "load done"
pipe.WaitEventId(EVENT_ID0);  // Vector waits before consuming
```

Wrong or missing barriers → silent data hazards. Extra barriers → stalls.

### How GPU DSLs handle this

| DSL | Sync model | User-visible? |
|-----|-----------|---------------|
| **Triton** | `__syncthreads` auto-inserted at shared mem boundaries | No |
| **TileLang GPU** | `T.Pipelined` inserts async barriers | No |
| **TileLang-Ascend** | `T.set_cross_flag()` / `T.wait_cross_flag()` per stage | **Yes — manual** |
| **Pallas (TPU)** | XLA handles inter-unit sync | No |
| **Helion** | Inherits Triton backend | No |

TileLang-Ascend is the only DSL that directly exposes Ascend sync primitives — and puts the burden on the user.

### PyAsc2 approach

`insert_sync=True` is set automatically by `asc2.jit`. Sync passes run after lowering to `ascendc` dialect:

**On Ascend910B / 910_93:**
- `InsertSync` — classic queue-position–based algorithm

**On Ascend910_95** (preferred, `sync_v2=True`):
- `InsertBufIdSyncV2` — newer algorithm: tracks buffer IDs instead of queue positions, correctly infers producer/consumer across unrolled loop iterations; reduces redundant sync ops
- `FuseBufIdSync` — merges adjacent sync ops where possible
- `ParallelLoadStore` — exploits 910_95 ability to run loads and stores concurrently

```
asctile IR (tile ops, no barriers)
    ↓ [UnrollLoop, AscLower passes]
ascendc IR (buf ops, no barriers)
    ↓ [InsertBufIdSyncV2]
ascendc IR (buf ops + set_flag/wait_flag)
    ↓ [FuseBufIdSync]
ascendc IR (fused barriers)
    ↓ [CodeEmitter + Bisheng]
AscendC .o binary
```

**Key challenge** (HLD §2.2): After `UnrollLoop`, a single Python tile loop produces many interleaved
MTE2/Vector ops. The pass must track which buffer ID each op writes to and which reads from it,
across unroll boundaries — without inserting redundant or missing syncs.

**Status:** validated for 910_95; correctness across all variants and unroll patterns still being confirmed.

---

## Challenge 3 — Ping-Pong: Double Buffering for Latency Hiding

### Why it matters on Ascend

Without double buffering, the Vector unit stalls while MTE2 loads the current tile from HBM.
With ping-pong:

```
Iteration N:   [MTE2: Load tile N+1 → UB[pong]]  overlaps  [Vector: Compute tile N from UB[ping]]
Iteration N+1: [MTE2: Load tile N+2 → UB[ping]]  overlaps  [Vector: Compute tile N+1 from UB[pong]]
```

This requires splitting the loop body into a preload phase and a compute+async-load phase —
mechanical but error-prone manually.

### How GPU DSLs handle this

| DSL | Double buffer | Syntax | Automation |
|-----|--------------|--------|-----------|
| **Triton** | Partial | `tl.async_copy` | Semi-manual |
| **TileLang GPU** | ✅ | `T.Pipelined(K, num_stages=2)` | Declarative |
| **Pallas (TPU)** | ✅ | `pl.prefetch_scoped(ref, ...)` | Semi-manual |
| **TileLang-Ascend** | Partial | `T.Pipelined(K, num_stages=2)` (limited) | Partial |
| **Helion** | Compiler best-effort | None | Auto via Triton |
| **cuTile** | ✅ | None — compiler decides | Fully automatic |

### PyAsc2 approach

User writes a plain reduction loop; compiler detects the pattern and applies ping-pong automatically:

```python
# User writes:
for k in asc2.range(K // TILE_K, unroll_factor=2):
    a = asc2.load(A_gm, [TILE_M, TILE_K], offsets=[row_off, k * TILE_K])
    b = asc2.load(B_gm, [TILE_K, TILE_N], offsets=[k * TILE_K, col_off])
    acc = acc + asc2.matmul(a, b)

# Compiler generates (conceptual):
#   Preload: MTE2  A[k=0], B[k=0]  → UB[ping]
#   Loop:
#     Compute: Cube  A[UB[ping]] @ B[UB[ping]] → L0C → acc
#     Load:    MTE2  A[k+1], B[k+1]            → UB[pong]
#     set_flag(MTE2→Cube); wait_flag(MTE2→Cube)
#     swap(ping, pong)
```

**Key passes involved:**

| Pass | Role |
|------|------|
| `TagUnrollGroups` | Identifies `asc2.range` loops with `unroll_factor > 1` |
| `DensifyUnrollGroups` | Clusters load/store ops so pipeline scheduler sees them contiguously |
| `UnrollLoop` | Physically unrolls by `unroll_factor` |
| `InsertBufIdSyncV2` | Inserts correct barriers between unrolled MTE2 and Cube ops |
| `HoistUBAllocation` | Moves UB allocations above the loop (one allocation covers all iterations) |

**Open challenge** (HLD §2.3): Compiler must hoist loads outside dependency chain and split
loop body correctly — this is the most complex transformation, combining loop analysis,
sync insertion, and UB allocation.

---

## Challenge 4 — UB Memory Allocation and Reuse

### What Ascend requires

UB (Unified Buffer) is on-chip SRAM, shared between all tile ops in a kernel.
On Ascend910B: **256 KB per AI Core**. After loop unrolling, many tile SSA values
can be live simultaneously:

```
Unrolled iter 0: tile_A_0, tile_B_0, tile_C_0  → 3 live
Unrolled iter 1: tile_A_1, tile_B_1, tile_C_1  → 3 more
With ping-pong:  doubles to 12 live tiles simultaneously
```

Naive allocation fails if they don't all fit. Solution: **liveness analysis + region reuse**.

### How GPU DSLs handle this

| DSL | SRAM/UB allocation | Liveness | Region reuse |
|-----|-------------------|----------|--------------|
| **Triton** | Implicit; compiler | Yes (MLIR) | Yes |
| **TileLang GPU** | `T.alloc_shared()` explicit | Yes | Partial |
| **TileLang-Ascend** | `alloc_ub()` explicit | No — user manages | ❌ |
| **Pallas (TPU)** | `BlockSpec` declares; XLA assigns | Yes | Yes |
| **Helion** | Fully implicit via Triton | Yes | Yes |

TileLang-Ascend requires manual UB layout — a significant usability and correctness burden.

### PyAsc2 approach

Two dedicated compiler passes (HLD §6.3 and §7.2):

**`ReuseUBAllocation`** (enabled via `reuse_ub=True`):
Builds liveness intervals for all tile SSA values after unrolling; assigns UB offsets such that
non-overlapping live ranges share the same region:

```
tile_A_0: live [op0..op3]  →  UB offset 0
tile_A_1: live [op4..op7]  →  UB offset 0   ← reuses same region (non-overlapping)
tile_B_0: live [op0..op5]  →  UB offset TILE_SIZE
```

`reuse_ub_in_out=True` extends this to input/output tiles (experimental).

**`ComputeMemoryConsumption`** (asc2 mode only):
Sums all live tile sizes per `TPosition` (UB, L0A, L0B, L0C, L1); raises a **compile-time error**
if any limit is exceeded — rather than silent runtime corruption:

```python
# If TILE_M=256, TILE_K=128, TILE_N=256 with fp16:
# UB usage = 2×(256×128 + 128×256)×2 bytes = 2×131072 = 262144 bytes > 256 KB
# → CompileError: UB overflow at compile time
```

**`HoistUBAllocation`**: moves `AllocTensor` ops above loops so one allocation covers all
iterations (avoids per-iteration alloc overhead in TPipe-managed mode).

---

## Summary: Challenge × DSL Approach

| Challenge | TileLang-Ascend (best existing) | Triton / Helion (GPU ref) | **PyAsc2** |
|-----------|--------------------------------|--------------------------|-----------|
| **1. TPipe/TQue hiding** | ❌ Manual `alloc_ub`, `alloc_L1` | ✅ Fully implicit | ✅ Compiler-generated from tile data flow |
| **2. Sync insertion** | ❌ Manual `set_cross_flag` | ✅ Auto | ✅ `InsertBufIdSyncV2` (+ `FuseBufIdSync`) |
| **3. Ping-pong** | ⚠️ Partial via `T.Pipelined` | ⚠️ `tl.async_copy` (semi-manual) | ✅ Auto from `asc2.range(unroll_factor=N)` |
| **4. UB allocation** | ❌ Manual, no reuse | ✅ Fully implicit | ✅ `ReuseUBAllocation` + `ComputeMemoryConsumption` |

**Key takeaway:** TileLang-Ascend solves the backend (MLIR → AscendC) but exposes all four
hardware challenges to the user. PyAsc2 targets full automation of all four — which is why it
needs a dedicated `asctile` MLIR dialect and a richer lowering pipeline (AscTile → AscLower → ascendc passes).

---

## PyAsc2 API Reference

### Core operations

| Python API | `asctile` op | Perf impact | Notes |
|---|---|---|---|
| `asc2.tensor(ptr, shape)` | `TensorOp` | None | HBM descriptor only |
| `asc2.load(tensor, shape, offsets)` | `LoadOp` | **Critical** — DMA coalescing | Last dim must align to 32 bytes |
| `asc2.store(tile, tensor, offsets)` | `StoreOp` | **Critical** | Same alignment constraint |
| `a + b`, `a * b`, etc. | `BinaryOp` | Medium | Tile ⊕ tile, tile ⊕ scalar |
| `asc2.matmul(a, b)` | `MatmulOp` | **Critical** — Cube utilization | Result always fp32; lowered to L0A/L0B/L0C |
| `asc2.softmax(x)` | `SoftmaxOp` | High | Single op, not composed |
| `tile.sum()`, `tile.max()` | `ReductionOp` | High | Optional `keep_dims` |
| `asc2.exp(x)`, `asc2.sqrt(x)` | `UnaryOp` | Medium | Lowered by `TransformMathOps` + `ExpandMath` |
| `asc2.where(mask, a, b)` | `SelectOp` | Medium | Element-wise conditional |
| `asc2.atomic_add(val, tensor, offsets)` | `AtomicRMWOp` | High (serializes) | Write back to HBM |
| `asc2.block_idx()` | `scf` / scalar | None | Current NPU block |
| `asc2.range(n, unroll_factor=k)` | `scf.for` + tag | **Critical** — enables ping-pong | `unroll_factor` triggers `TagUnrollGroups` |

### Memory hierarchy (TileLocation)

| Location | Hardware | Use |
|----------|----------|-----|
| `UB` | Unified Buffer (default) | Vector operations |
| `L0A` / `L0B` | L0 matrix input buffers | Matmul inputs (auto-assigned) |
| `L0C` | L0 matrix output | Matmul accumulator (auto-assigned) |
| `L1` | L1 cache | Intermediate staging |

### CompileOptions for PyAsc2

| Option | Default | Effect |
|--------|---------|--------|
| `run_asc2_passes` | `True` | Enable AscTile + AscLower pipeline |
| `insert_sync` | `True` | Auto-insert `set_flag`/`wait_flag` |
| `static_alloc` | `False` | Static vs TPipe-managed UB allocation |
| `reuse_ub` | `False` | `ReuseUBAllocation` pass |
| `reuse_ub_in_out` | `False` | Extend reuse to I/O tiles (experimental) |
| `densify_load_store` | `False` | `DensifyUnrollGroups` pass (experimental) |
| `sync_v2` | `False` | Use `InsertBufIdSyncV2` on 910_95 |
| `opt_level` | `3` | Bisheng `-O` level (1–3) |
