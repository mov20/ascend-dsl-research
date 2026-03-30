# Table 2: Syntax Constructs & Performance — Ascend Focus

*Last updated: 2026-03-30*

This document maps the **four key compiler challenges** for Ascend NPU kernels (from the PyAsc2
design doc) to concrete syntax and compiler strategies across existing DSLs. The goal is to
understand what solutions already exist in GPU DSLs and where PyAsc2 needs to do original work.

---

## Four Key Challenges

Ascend NPU has a fundamentally different execution model from GPUs. Where a GPU SM schedules
warps dynamically, Ascend's AI Core requires explicit management of:

1. **Programming model** — TPipe/TQue lifecycles and sync barriers must be hidden from the user
2. **Sync insertion** — `set_flag`/`wait_flag` between pipeline stages must be auto-inserted
3. **Ping-pong (double buffering)** — load for tile N+1 must overlap compute on tile N, automatically
4. **UB memory allocation** — SSA tile values after loop unrolling must fit in finite on-chip UB

These map directly to the performance gap: if any of the four is wrong, the kernel either
produces incorrect results or runs far below peak FLOPS.

---

## Challenge 1 — Programming Model: Hiding TPipe/TQue

### What Ascend requires (AscendC)

```cpp
// AscendC: developer manually creates pipe, queues, events
TPipe pipe;
TQue<QuePosition::VECIN, 2> inQueueA, inQueueB;
TQue<QuePosition::VECOUT, 1> outQueue;
pipe.InitBuffer(inQueueA, 2, TILE_LENGTH * sizeof(half));
// ... manual AllocTensor / FreeTensor / EnQue / DeQue on every operation
```

Every kernel must manage buffer lifecycles explicitly. This is the main source of boilerplate
and bugs in raw AscendC code.

### How GPU DSLs handle the equivalent

| DSL | Approach | What user writes | What's hidden |
|-----|----------|-----------------|---------------|
| **Triton** | Fully implicit | `x = tl.load(ptr + offsets, mask=mask)` | All SRAM allocation, barriers |
| **TileLang GPU** | Declarative tiles | `A = T.alloc_fragment((BM, BK), dtype)` | TQue equiv., sync between stages |
| **TileLang-Ascend** | Semi-explicit | `A_l1 = alloc_L1(...)`, `A_ub = alloc_ub(...)`, manual `T.set_cross_flag` | None — user writes all of it |
| **Pallas (TPU)** | Declarative | `BlockSpec(block_shape=(BM, BK), memory_space=pltpu.SMEM)` | HBM↔SRAM DMA, barriers |
| **Helion** | Fully implicit | `a = x[tile_m, tile_k]` (PyTorch indexing) | All allocation, barriers, sync |

### PyAsc2 approach

User writes tile-based Python; compiler generates all TPipe/TQue lifecycle code:

```python
@asc2.jit
def add_kernel(x: Tensor, y: Tensor, out: Tensor, TILE: tl.constexpr):
    pid = asc2.program_id(0)
    tile = asc2.tile(pid, TILE)
    a = x[tile]          # compiler: AllocTensor → inQueueA.EnQue
    b = y[tile]          # compiler: AllocTensor → inQueueB.EnQue
    out[tile] = a + b    # compiler: DeQue → compute → outQueue.EnQue → FreeTensor
```

**Key design decisions:**
- NumPy-like indexing (`x[tile]`) as primary API — closest to Helion
- `asc2.tile()` replaces `tl.program_id` + offset arithmetic
- Compiler infers buffer count, TQue depth, and pipe stages from data flow

**Gap vs GPU DSLs:** GPU DSLs don't need TPipe/TQue — they're GPU-specific constructs.
PyAsc2 must generate this correctly from data-flow analysis alone.

---

## Challenge 2 — Sync Insertion: `set_flag` / `wait_flag`

### What Ascend requires

Ascend AI Core has separate Cube (matrix), Vector, and MTE (memory transfer) units that run
in parallel. Between pipeline stages, the compiler must insert explicit sync barriers:

```cpp
// AscendC sync pattern between MTE2 (load) → Vector (compute)
pipe.SetEventId(EVENT_ID0);
pipe.WaitEventId(EVENT_ID0);
```

Wrong or missing barriers → data hazards (silent correctness bugs).
Extra barriers → stalls (performance loss).

### How GPU DSLs handle this

| DSL | Sync model | Explicit barriers? |
|-----|-----------|-------------------|
| **Triton** | GPU: `__syncthreads` auto-inserted at shared mem boundaries | No — fully hidden |
| **TileLang GPU** | `T.Pipelined` inserts async barriers | No — declarative |
| **TileLang-Ascend** | `T.set_cross_flag()` / `T.wait_cross_flag()` per stage | **Yes — manual** |
| **Pallas (TPU)** | XLA handles inter-unit sync | No — fully hidden |
| **Helion** | Inherits from Triton backend | No — fully hidden |

TileLang-Ascend is the only existing DSL that directly exposes Ascend sync primitives —
and it puts the burden on the user.

### PyAsc2 approach

**InsertBufIdSyncV2** compiler pass: runs after tile-level IR lowering, analyzes
producer/consumer relationships across loop iterations, and inserts `set_flag`/`wait_flag`
at the `ascendc` dialect level — invisible to the user.

```
asctile IR (tile ops)
    ↓ [UnrollTileLoops]
ascendc IR (buf ops, no barriers)
    ↓ [InsertBufIdSyncV2]
ascendc IR (buf ops + set_flag/wait_flag)
    ↓ [Bisheng]
AscendC C++
```

**Key challenge:** After loop unrolling, a single Python tile loop may produce many interleaved
MTE2/Vector ops. The pass must correctly track which buffer ID each op writes to and which
reads from it, across unroll boundaries.

**Status per hardware:** Algorithm validated for Ascend910_95; still being confirmed for
910B and 910_93 variants (noted as open in design doc).

---

## Challenge 3 — Ping-Pong: Double Buffering for Memory Latency Hiding

### Why it matters on Ascend

Ascend HBM bandwidth is high, but latency to UB is significant. Without double buffering,
the Vector unit stalls waiting for MTE2 to finish loading the current tile. With ping-pong:

```
Iteration N:   [Load tile N+1] overlaps with [Compute tile N]
Iteration N+1: [Load tile N+2] overlaps with [Compute tile N+1]
```

This requires splitting the loop body into a "preload" phase and a "compute + async load" phase,
which is mechanical but error-prone to do manually.

### How GPU DSLs handle this

| DSL | Double buffer support | Syntax | Automation level |
|-----|-----------------------|--------|-----------------|
| **Triton** | Partial — `tl.async_copy` | Manual async copy + barrier | Semi-manual |
| **TileLang GPU** | ✅ Declarative | `T.Pipelined(K, num_stages=2)` | **Fully automatic** |
| **Pallas (TPU)** | ✅ `prefetch_scoped()` | `pl.prefetch_scoped(ref, ...)` | Semi-manual |
| **TileLang-Ascend** | Partial — via T.Pipelined | `T.Pipelined(K, num_stages=2)` (limited) | Partial |
| **Helion** | Inherits Triton backend | None explicit | Compiler best-effort |
| **cuTile** | ✅ Auto | None — compiler decides | **Fully automatic** |

TileLang GPU's `T.Pipelined` is the closest reference implementation.

### PyAsc2 approach

Automatic ping-pong from tile-level loop structure — user writes a simple tile loop:

```python
# User writes:
for k in asc2.range(K, TILE_K):
    acc += A[tile_m, k] @ B[k, tile_n]   # compiler infers: double buffer A, B

# Compiler generates (conceptual):
# Preload: MTE2 loads A[tile_m, 0..TILE_K], B[0..TILE_K, tile_n] → UB[ping]
# Loop:
#   Compute: Vector  A[UB[ping]] @ B[UB[ping]] → acc
#   Load:    MTE2    A[..k+1..], B[..k+1..] → UB[pong]
#   swap(ping, pong)
```

**Compiler work required:**
1. Detect reduction loop over `k` with load-then-compute pattern
2. Split loop body: hoist loads into async DMA phase
3. Allocate two UB slots per tile (ping + pong buffers)
4. Insert `set_flag`/`wait_flag` at the DMA/compute boundary

This is the most complex of the four challenges — it combines sync insertion, UB allocation,
and loop transformation.

---

## Challenge 4 — UB Memory Allocation and Reuse

### What Ascend requires

UB (Unified Buffer) is on-chip SRAM shared between all tile operations in a kernel.
On Ascend910B: 256 KB per AI Core. After loop unrolling, a kernel may create many tile
SSA values simultaneously — naive allocation fails if they don't all fit.

```
Unrolled iter 0: tile_A_0, tile_B_0, tile_C_0   → 3 live at once
Unrolled iter 1: tile_A_1, tile_B_1, tile_C_1   → 3 more
...
Total live: potentially 6+ tiles × tile_size → exceeds 256KB
```

Solution: **liveness analysis** + **region reuse** — once `tile_A_0` is consumed, its
UB region can be reused for `tile_A_2`.

### How GPU DSLs handle this

| DSL | Shared mem allocation | Liveness analysis | Region reuse |
|-----|-----------------------|------------------|--------------|
| **Triton** | Implicit; compiler places in registers/L1 | Yes (MLIR) | Yes (compiler) |
| **TileLang GPU** | `T.alloc_shared()` explicit; compiler infers lifetime | Yes | Partial |
| **TileLang-Ascend** | `alloc_ub()` explicit; **no reuse** | No — user manages | ❌ Manual |
| **Pallas (TPU)** | `BlockSpec` declares sizes; compiler assigns | Yes | Yes (XLA) |
| **Helion** | Fully implicit via Triton | Yes | Yes |

TileLang-Ascend currently requires manual UB layout — a significant usability and
correctness burden.

### PyAsc2 approach

Two compiler passes:

**`ReuseUBAllocation`** — runs after loop unrolling; builds liveness intervals for all tile
SSA values; assigns UB offsets such that non-overlapping live ranges share the same region:

```
tile_A_0: live [op0..op3]  →  UB[0..TILE_SIZE]
tile_A_1: live [op4..op7]  →  UB[0..TILE_SIZE]   ← reuses same region
tile_B_0: live [op0..op5]  →  UB[TILE_SIZE..2*TILE_SIZE]
```

**`ComputeMemoryConsumption`** — validates that peak live UB footprint does not exceed
hardware limit; raises a compile-time error if tile sizes are too large (rather than silent
runtime corruption).

**UB allocation model:**
```python
# User just writes:
TILE_M, TILE_K, TILE_N = 128, 64, 128  # constexpr at JIT time
# Compiler auto-computes: 2 ping-pong × (TILE_M×TILE_K + TILE_K×TILE_N) × dtype_bytes
# Validates vs 256KB UB limit before generating any code
```

---

## Summary: Challenge × DSL Approach

| Challenge | TileLang-Ascend | Triton (GPU ref) | Pallas (TPU ref) | **PyAsc2 approach** |
|-----------|----------------|-----------------|-----------------|---------------------|
| **1. TPipe/TQue hiding** | ❌ Manual `alloc_ub` + `alloc_L1` | ✅ Fully implicit | ✅ `BlockSpec` | ✅ Compiler-generated from tile data flow |
| **2. Sync insertion** | ❌ Manual `set_cross_flag` | ✅ Auto (`__syncthreads`) | ✅ XLA handles | ✅ `InsertBufIdSyncV2` pass |
| **3. Ping-pong / double buffer** | ⚠️ Partial via `T.Pipelined` | ⚠️ `tl.async_copy` (manual) | ⚠️ `prefetch_scoped` (manual) | ✅ Automatic from loop structure |
| **4. UB allocation & reuse** | ❌ Manual, no reuse | ✅ Fully implicit | ✅ XLA manages | ✅ `ReuseUBAllocation` + `ComputeMemoryConsumption` |

**Key takeaway:** TileLang-Ascend solves the backend problem (MLIR → AscendC) but exposes
all four hardware challenges to the user. PyAsc2 targets full automation of all four —
which is also why it needs a dedicated `asctile` MLIR dialect and a richer lowering pipeline.

---

## Kernel Primitives Reference

| Construct | Triton | TileLang | Pallas | Helion | PyAsc2 (target) | Perf impact |
|---|---|---|---|---|---|---|
| **Kernel entry** | `@triton.jit` | `T.Kernel(bm, bn, k)` | `@pl.kernel` + `BlockSpec` | `@helion.kernel()` | `@asc2.jit` | Minimal |
| **Tile indexing** | `tl.program_id(axis)` | Kernel params | `pl.program_id()` | Implicit | `asc2.program_id()` + `asc2.tile()` | Low |
| **Load from HBM** | `tl.load(ptr, mask)` | `T.alloc_fragment` + copy | `pl.load(ref, idx)` | `x[tile_m, tile_k]` | `x[tile]` | **High** — coalescing |
| **Store to HBM** | `tl.store(ptr, val)` | Fragment store | `pl.store(ref, idx, val)` | `out[tile] = val` | `out[tile] = val` | **High** |
| **Matrix multiply** | `tl.dot(a, b)` | `T.gemm(a, b, c)` | `jax.lax.dot` | `torch.addmm(acc, a, b)` | `asc2.dot(a, b)` | **Critical** — Cube utilization |
| **Reduction loop** | Manual `for` | `T.Pipelined(K, ...)` | `grid` dimension | `for k in hl.tile(K)` | `for k in asc2.range(K, TILE_K)` | **Critical** — ping-pong |
| **On-chip buffer** | Implicit | `T.alloc_shared()` | `pltpu.SMEM` | Implicit | Implicit (compiler) | **Critical** — UB limit |
| **Sync barrier** | Implicit | `T.set_cross_flag()` (Ascend) | Implicit | Implicit | Implicit (`InsertBufIdSyncV2`) | **Critical** — correctness |
| **Autotuning** | `triton.autotune` | Config search | Manual | Built-in | `asc2.autotune` (planned) | **High** — tile sizes |
