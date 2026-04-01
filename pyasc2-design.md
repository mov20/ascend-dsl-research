# PyAsc2 API Design

## 1. Requirements

- Express kernels in terms of **tensors** (ND-arrays in global memory) and **tiles** (fixed-shape chunks in on-chip memory). Buffer addresses, TPipe/TQue lifecycles, and synchronization barriers are not exposed to the user.
- Provide NumPy-like syntax for arithmetic, reductions, shape manipulation, masking, and atomics.
- **[Main engineering challenge]** Automate synchronization insertion, UB memory allocation, and ping-pong optimization through compiler passes and compiler hints.
- Target hardware: Ascend 910B/C, 950.

## 2. Key Challenges for Ascend NPU

Ascend 910B AI Core has two types of compute cores:
- **AIC** (Cube) — matrix/tensor operations
- **AIV** (Vector) — vector operations; 1 AIC : 2 AIV ratio on 910B

Each core has its own MTE (DMA), Scalar Unit, and Compute Unit.
AIC and AIV have no direct interconnect — data exchange goes through L2/Global Memory.

### 2.1 Synchronization Insertion

MTE and Compute Unit within each core run in parallel. Explicit `set_flag`/`wait_flag`
barriers must be inserted between them. Missing barriers cause silent data hazards;
redundant barriers cause stalls. Must be fully automated by the compiler.

### 2.2 Ping-Pong (Double Buffering)

MTE loads for tile N+1 must overlap with Compute on tile N.
Requires the compiler to split the loop body, hoist loads, and insert correct barriers —
without user annotation.

### 2.3 UB Memory Allocation and Reuse

On-chip UB (Unified Buffer) is ~256 KB per Da Vinci core (910B).
After loop unrolling, many tile SSA values can be live simultaneously.
Compiler must compute liveness and reuse freed UB regions.
Tile shapes are statically known at JIT time; total live footprint must be
validated at compile time.

### 2.4 Hardware Differences: 910B/C vs 950

> *Partial information — to be updated as more documentation becomes available.*

**910C** — two 910B dies in one package. Same Da Vinci architecture, AIC/AIV model unchanged.

**950 (Ascend A5)** — new architecture with changes relevant to DSL design:

- **AIC↔AIV data transfer via TPUSH/TPOP**: hardware instructions implementing
  a tag-based dual-channel FIFO between Cube and Vector cores. On 910B/C
  (PLATFORM_A2A3), ring buffer lives in Global Memory — DMA in and out.
  On 950 (PLATFORM_A5), ring buffer lives in consumer's on-chip SRAM —
  **zero-copy**. This fundamentally changes the cost of Cube↔Vector pipelines.
- **Memory access granularity**: 512 bytes → 128 bytes.
- **New data formats**: MXFP4, MXFP8, HiF8 (in addition to FP16/BF16/INT8).
- **Pipeline tuning**: different buffer allocation strategy and pipeline depth
  vs. A2/A3 (confirmed in PTO FA kernel for A5).
- **Sync model**: `set_flag`/`wait_flag` between PIPE_MTE2 and PIPE_V are still
  explicit even on A5 (confirmed in PTO engram kernel source).

## 3. Programming Model Analysis

### 3.1 AscendC

AscendC is the official C++ kernel language for Ascend NPU and the compilation
target for PyAsc2. Understanding how it handles the three key challenges defines
the baseline that PyAsc2 must improve upon.

#### Sync Insertion

AscendC exposes synchronization directly to the user via `TPipe` and `TQue`.
Every data transfer requires explicit `EnQue`/`DeQue` calls; every pipeline
stage boundary requires explicit `SetEventId`/`WaitEventId`:

```cpp
TPipe pipe;
TQue<QuePosition::VECIN, 2> inQueue;
TQue<QuePosition::VECOUT, 1> outQueue;
pipe.InitBuffer(inQueue, 2, TILE_SIZE);    // TPipe allocates UB memory for TQue buffers
pipe.InitBuffer(outQueue, 1, TILE_SIZE);

// MTE2 stage: load
LocalTensor<half> tile = inQueue.AllocTensor<half>();
DataCopy(tile, gm_src[offset], TILE_SIZE);
inQueue.EnQue(tile);              // signal: load done

// Vector stage: compute
LocalTensor<half> tile = inQueue.DeQue<half>();   // wait: load done
Add(out, tile, tile2, TILE_SIZE);
outQueue.EnQue(out);

// MTE3 stage: store
LocalTensor<half> out = outQueue.DeQue<half>();
DataCopy(gm_dst[offset], out, TILE_SIZE);
outQueue.FreeTensor(out);

// Cross-unit sync (e.g. between independent pipelines)
pipe.SetEventId(EVENT_ID0);       // producer signals
pipe.WaitEventId(EVENT_ID0);      // consumer waits
```
Source: Ascend C Operator Development Guide, CANN 8.0 https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html

Missing or misplaced EnQue/DeQue / SetEventId/WaitEventId causes silent data hazards. AscendC does not insert any barriers automatically.

**Conclusion**: AscendC doesn't solve the challenge — it exposes it. The user is responsible for every barrier manually. This is a source of bugs and boilerplate.

#### Ping-pong

```cpp
// Queue depth=2: two UB slots (ping + pong)
// Depth is set in TWO places — they must match:
constexpr int QUEUE_DEPTH = 2;

TQue<QuePosition::VECIN, QUEUE_DEPTH> inQueue;   // (1) template param
pipe.InitBuffer(inQueue, QUEUE_DEPTH, TILE_SIZE); // (2) runtime init

for (int i = 0; i < num_tiles; i++) {
    // MTE2: load tile i+1 while Vector computes tile i
    LocalTensor<half> tile = inQueue.AllocTensor<half>();
    DataCopy(tile, gm_src[(i+1) * TILE_SIZE], TILE_SIZE);
    inQueue.EnQue(tile);

    // Vector: compute tile i
    LocalTensor<half> cur = inQueue.DeQue<half>();
    Add(out, cur, cur2, TILE_SIZE);
    inQueue.FreeTensor(cur);
}
```
Depth 2 = two slots in UB. While Vector works on ping, MTE2 loads into pong. But:

User chooses queue depth manually
User structures the loop manually to achieve overlap
Compiler doesn't help

**Conclusion**: AscendC supports ping-pong, but requires manual orchestration. No automatic loop body partitioning.

#### UB Memory Allocation and Reuse

```cpp
TPipe pipe;
TQue<QuePosition::VECIN,  2> inQueueA;   // reserves 2 × TILE_SIZE bytes in UB
TQue<QuePosition::VECIN,  2> inQueueB;   // reserves 2 × TILE_SIZE bytes in UB
TQue<QuePosition::VECOUT, 1> outQueue;   // reserves 1 × TILE_SIZE bytes in UB
// Total: 5 × TILE_SIZE — must fit in 256 KB; no compiler check

pipe.InitBuffer(inQueueA,  2, TILE_SIZE);
pipe.InitBuffer(inQueueB,  2, TILE_SIZE);
pipe.InitBuffer(outQueue,  1, TILE_SIZE);

for (int i = 0; i < num_tiles; i++) {
    LocalTensor<half> a = inQueueA.AllocTensor<half>();  // acquire free UB slot
    // ... use a ...
    inQueueA.FreeTensor(a);  // return slot — reused next iteration
}
```

`TPipe` partitions UB at kernel initialization via `InitBuffer` — one call per queue, each reserving `depth × tile_size` bytes statically. Within the loop, `AllocTensor` acquires a free slot from the queue's pool and `FreeTensor` returns it for reuse on the next iteration. Two queues never share a UB region, even if they are never live simultaneously — there is no liveness analysis.

**Conclusion**: AscendC requires the user to manually plan UB layout — sizing each buffer, choosing queue depths, and validating the total fit within 256 KB. Exceeding the limit causes silent memory corruption at runtime. The compiler provides no assistance.


### 3.2 Triton

#### Sync Insertion

No sync primitives are exposed to the user. The TritonGPU IR lowering pass inserts async copy
wait groups and shared memory barriers automatically, based on data flow analysis. <sup>[[10]](#ref-10)</sup>

```python
@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)   # no barrier needed — compiler inserts it
    y = tl.load(Y + offs)
    tl.store(Z + offs, x + y)
```

**Conclusion**: Sync is fully automated. The user cannot and does not insert barriers manually.

#### Ping-Pong

The user writes a plain loop. The compiler's software pipelining pass automatically reorders
instructions across iteration boundaries — issuing async loads for iteration K+N-1 while
computing iteration K. Pipeline depth is controlled by a single `num_stages` hint. <sup>[[10]](#ref-10)</sup>

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, num_stages: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=num_stages):  # compiler pipelines this loop
        a = tl.load(...)
        b = tl.load(...)
        acc += tl.dot(a, b)
    tl.store(...)
```

**Conclusion**: Loop restructuring and barrier insertion are fully automated. `num_stages` is a
user-facing tuning parameter — in practice set via autotuning, not manually.

#### UB Memory Allocation and Reuse

Shared memory (GPU equivalent of Ascend UB) is managed entirely by the compiler. No user-visible
allocation calls exist. If the total shared memory footprint exceeds hardware limits, Triton raises
a compile-time error — unlike AscendC which silently corrupts memory at runtime.

Triton handles shared memory allocation through two separate mechanisms:

**1. Encoding assignment** — decides *what* goes to shared memory:
- **Pure elementwise ops** (add, mul, relu) — encoding passes through unchanged
  (`SameOperandsAndResultEncoding` trait); tensors stay in registers. No shared memory allocated.
- **Reductions** (softmax, sum, max) — `ReduceDataDuplication` pass assigns `SwizzledSharedEncodingAttr`
  to reduction intermediates, inserting `LocalAllocOp` for shared memory staging. <sup>[[12]](#ref-12)</sup>
- **Matrix ops** (`tl.dot`) — `BlockedToMMA` pass converts operand encodings from `Blocked` to MMA
  shared memory encoding, inserting layout conversion ops automatically. <sup>[[13]](#ref-13)</sup>

**2. Liveness analysis** — decides *where* in shared memory each buffer lives:
`AllocationAnalysis` class in `Allocation.cpp` runs three phases: <sup>[[11]](#ref-11)</sup>
- `getValuesAndSizes()` — collects all shared memory values and their sizes
- `resolveLiveness()` — computes live ranges via MLIR standard liveness analysis
- `computeOffsets()` — assigns offsets using interference graph + first-fit graph coloring,
  reusing freed regions for non-overlapping buffers

**Conclusion**: The user writes the same code regardless of whether shared memory is needed.
Encoding assignment and liveness-based reuse are fully automated by the compiler.
Unlike AscendC, fit within hardware limits is validated at compile time.

### 3.3 cuTile

#### Sync Insertion

No sync primitives are exposed to the user. The TileIR compiler automatically injects all
necessary barriers during code generation. Explicit synchronization or communication within
a block is not permitted by design — this is a fundamental constraint of the cuTile model. <sup>[[14]](#ref-14)</sup>

This includes TMA (Tensor Memory Accelerator) async operations: the `convert-tileaa-to-tileas`
pass converts `tileaa.tiled_load` into async loads, and the `convert-pipeline-to-nvvm` pass
generates the corresponding `nvvm.mbarrier.*` intrinsics for memory-vs-compute
synchronization. <sup>[[15]](#ref-15) [[16]](#ref-16)</sup>

```python
@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)
```

An 86-line Python cuTile kernel expands to ~1,900 lines of PTX with 20 barrier objects —
none written by the developer. <sup>[[15]](#ref-15)</sup>

**Conclusion**: Sync is fully automated and not even expressible by the user.
cuTile is the most restrictive model — no escape hatch for manual barriers.

#### Ping-Pong

The user writes a simple loop. The TileIR compiler transforms it into a pipelined loop
through three passes: <sup>[[15]](#ref-15) [[16]](#ref-16)</sup>
1. `convert-tileaa-to-tileas` — converts tiled loads into async loads with pipeline ops
2. `tileas-materialize-async` — creates the async pipeline structure with multi-buffering
3. `convert-pipeline-to-nvvm` — lowers to NVVM barrier intrinsics (`nvvm.mbarrier.*`)

The result is a three-phase loop — prologue (pre-load N iterations), steady-state
(overlap load K+N with compute K), and epilogue (drain remaining computes).

Unlike Triton, there is no user-facing `num_stages` parameter — the compiler determines
pipeline depth automatically.

Performance evidence: GEMM achieves ~90% of cuBLAS on Blackwell with zero user-written
pipelining. <sup>[[17]](#ref-17)</sup> Attention kernels are still a work in progress —
TiledAttention research implementation reaches 0.63x vs fused PyTorch SDPA. <sup>[[18]](#ref-18)</sup>

**Conclusion**: Ping-pong is fully automated. No loop restructuring, no buffer management,
no pipeline depth hint from the user. Strong results for GEMM; attention still maturing.

#### UB Memory Allocation and Reuse

**SIMT kernels (shared memory):** The compiler decides what goes to shared memory and
allocates it automatically. The user has no shared memory API — no `__shared__` declarations,
no size hints. In practice, the compiler allocated 180 KB of shared memory for an 86-line
MOE kernel without any user input. <sup>[[15]](#ref-15) [[19]](#ref-19)</sup>

**Tensor core kernels on Blackwell (TMEM):** SM100 introduces TMEM — 256 KB per SM,
dedicated to tensor cores, separate from shared memory. Operand A lives in TMEM or SMEM,
operand B in SMEM, accumulator in TMEM exclusively. Allocation is dynamic via
`tcgen05.alloc` (32-column minimum, power-of-2 granularity). <sup>[[20]](#ref-20)</sup>

cuTile handles TMEM automatically including contention handling — retry with 100ns backoff
when allocation fails. On Hopper, matrix operands competed for register file space;
on Blackwell, TMEM decouples tensor cores from CUDA cores entirely. <sup>[[19]](#ref-19)</sup>

**Conclusion**: All on-chip memory allocation — shared memory for SIMT, TMEM for tensor
cores — is fully compiler-managed. No user-visible allocation API exists.

### 3.4 TileLang-Ascend

#### Sync Insertion

TileLang-Ascend offers two modes — manual and automatic.

**Manual:** user writes explicit primitives — `T.set_flag()`, `T.wait_flag()`,
`T.set_cross_flag()` / `T.wait_cross_flag()` (for Cube↔Vector sync),
`T.pipe_barrier()`, `T.barrier_all()`. <sup>[[21]](#ref-21)</sup>

**Automatic:** enabled via pass config: <sup>[[22]](#ref-22)</sup>
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```
The `AscendSyncInsert` pass analyzes producer-consumer data dependencies,
separates Cube/Vector code regions (`CombineCV` pass), and inserts fine-grained
flag operations at synchronization points. <sup>[[23]](#ref-23)</sup>

**Known limitation:** with double buffering + auto sync, the compiler currently
generates conservative `PipeBarrier<PIPE_ALL>` instead of fine-grained flags,
which may hurt performance. <sup>[[24]](#ref-24)</sup>

**Conclusion**: Hybrid model — auto sync available but not yet mature. Manual
sync remains the practical default for performance-critical kernels.

#### Ping-Pong

`T.Pipelined(range, num_stages=N)` is the primary API. The user sets pipeline
depth and marks which loop to pipeline — necessary in nested loops where
different levels produce different overlap behavior. The `CrossCorePipeline` pass
extends buffer dimensions and restructures the loop. <sup>[[21]](#ref-21)</sup>

```python
for m in T.range(M // block_M):
    for k in T.Pipelined(K // block_K, num_stages=2):  # only this loop is pipelined
        T.copy(A_global, A_L1)
        T.gemm_v0(A_L1, B_L1, C_L0C)
```

Similar to Triton's `num_stages` — user-facing tuning parameter, not fully automated.
Combined with the auto sync limitation (Issue #110), pipelined kernels currently
get conservative `PipeBarrier<PIPE_ALL>` instead of fine-grained flags. <sup>[[24]](#ref-24)</sup>

**Conclusion**: Loop pipelining is compiler-supported via `T.Pipelined`, but
pipeline depth and loop selection are user-chosen. Barrier insertion is still conservative.

#### UB Memory Allocation and Reuse

User explicitly allocates buffers at each memory hierarchy level: <sup>[[21]](#ref-21)</sup>

```python
q_l1     = T.alloc_L1([block_M, dim], dtype)        # L1 buffer
acc_l0c  = T.alloc_L0C([block_M, block_N], dtype)   # L0C (register-like SRAM)
softmax  = T.alloc_ub([block_M // 2], dtype)         # UB (256 KB limit)
```

**Default mode:** no liveness analysis. User must manually assign UB offsets
via `T.annotate_address(buffer, offset)` — similar to AscendC's manual planning.
No compile-time validation; exceeding 256 KB causes silent corruption.

**Opt-in mode:** `TL_ASCEND_MEMORY_PLANNING=True` enables the `AscendMemoryPlanning`
pass — liveness analysis, interference graph, first-fit coloring for buffer reuse.
Eliminates manual `T.annotate_address()`. Validates UB fit at compile time. <sup>[[25]](#ref-25)</sup>

L1 and L0A/B/C allocation has no compile-time limit checking in either mode.

**Conclusion**: Memory hierarchy placement is always user-chosen (unlike Triton).
Buffer reuse and validation are available but opt-in — not yet the default.

### 3.5 Triton-Ascend

### 3.6 PyAsc v3

## 4. Key Design Decisions

## 5. API Specification

## 6. References

| # | Description | URL |
|---|-------------|-----|
| 1 | PyAsc2 HLD (internal) | https://gitcode.com/compiler-team/pyasc/pull/99 |
| 2 | PyPTO — tile-based framework for Ascend (official) | https://github.com/hw-native-sys/pypto |
| 3 | PyPTO official repo (GitCode/CANN) | https://gitcode.com/cann/pypto/ |
| 4 | PTO ISA — TPUSH/TPOP protocol | https://github.com/hw-native-sys/pypto/blob/main/docs/en/reference/pto-isa/01-tpush_tpop.md |
| 5 | PTO Tile Library (pto-isa) | https://github.com/hengliao1972/pto-isa |
| 6 | PTO A5 kernel examples | https://github.com/hengliao1972/pto-isa/tree/main/kernels/manual/a5 |
| 7 | PTO A5 Flash Attention kernel | https://github.com/hengliao1972/pto-isa/blob/main/kernels/manual/a5/flash_atten/README.md |
| 8 | PTO engram SIMT kernel (A5) | https://github.com/hengliao1972/pto-isa/blob/main/kernels/manual/a5/engram_simt/engram-simt_kernel.cpp |
| 9 | Ascend 910B architecture overview | https://arxiv.org/html/2505.15112v1 |
| 10 | Triton matmul tutorial — pipelined kernel with no user-written barriers | https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html |
| 11 | Triton shared memory allocation — `AllocationAnalysis` | https://github.com/triton-lang/triton/blob/main/lib/Analysis/Allocation.cpp |
| 12 | Triton encoding assignment for reductions — `ReduceDataDuplication` | https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/ReduceDataDuplication.cpp |
| 13 | Triton encoding assignment for dot/matmul — `BlockedToMMA` | https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp |
| 14 | cuTile execution model — no intra-block sync | https://docs.nvidia.com/cuda/cutile-python/execution.html |
| 15 | TileIR internals — from cuTile to MLIR/LLVM to SASS | https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/ |
| 16 | TileIR source code (NVIDIA, open-source) | https://github.com/NVIDIA/cuda-tile |
| 17 | NVIDIA blog — high-performance matrix multiply in cuTile | https://developer.nvidia.com/blog/how-to-write-high-performance-matrix-multiply-in-nvidia-cuda-tile |
| 18 | TiledAttention — SDPA in cuTile (arXiv:2603.01960) | https://arxiv.org/abs/2603.01960 |
| 19 | CuTile on Blackwell — compiler moat analysis (TMEM, mbarrier details) | https://patricktoulme.substack.com/p/cutile-on-blackwell-nvidias-compiler |
| 20 | tcgen05 for dummies — TMEM allocation, operand placement | https://gau-nernst.github.io/tcgen05/ |
| 21 | TileLang-Ascend repo — examples and sync primitives | https://github.com/tile-ai/tilelang-ascend |
| 22 | TileLang-Ascend auto sync config (Issue #98) | https://github.com/tile-ai/tilelang-ascend/issues/98 |
| 23 | TileLang-Ascend CombineCV + AscendSyncInsert passes | https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/src/transform/ascend_combinecv.cc |
| 24 | TileLang-Ascend double buffer sync issue (#110) | https://github.com/tile-ai/tilelang-ascend/issues/110 |
| 25 | TileLang-Ascend roadmap — memory planning, autotuner (Issue #3) | https://github.com/tile-ai/tilelang-ascend/issues/3 |
