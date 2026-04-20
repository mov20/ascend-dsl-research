# PyAsc2 API Design

## Contents

- [1. Requirements](#1-requirements)
- [2. Key Challenges for Ascend NPU](#2-key-challenges-for-ascend-npu)
  - [2.1 Synchronization Insertion](#21-synchronization-insertion)
  - [2.2 Ping-Pong (Double Buffering)](#22-ping-pong-double-buffering)
  - [2.3 UB Memory Allocation and Reuse](#23-ub-memory-allocation-and-reuse)
  - [2.4 Hardware Differences: 910B/C vs 950](#24-hardware-differences-910bc-vs-950)
    - [2.4.1 Hardware Capability Deltas](#241-hardware-capability-deltas)
    - [2.4.2 Implications for the Three DSL Challenges](#242-implications-for-the-three-dsl-challenges)
- [3. Programming Model Analysis](#3-programming-model-analysis)
  - [3.1 AscendC](#31-ascendc)
    - [3.1.1 AscendC on 910B/C](#311-ascendc-on-910bc)
    - [3.1.2 AscendC on 950](#312-ascendc-on-950)
  - [3.2 Triton](#32-triton)
  - [3.3 cuTile](#33-cutile)
  - [3.4 TileLang-Ascend](#34-tilelang-ascend)
  - [3.5 Triton-Ascend](#35-triton-ascend)
  - [3.6 PyPTO](#36-pypto)
    - [3.6.1 PyPTO-main](#361-pypto-main)
    - [3.6.2 PyPTOv3 (redesign)](#362-pyptov3-redesign)
  - [3.7 Pallas (JAX)](#37-pallas-jax)
  - [3.8 Mojo (Modular)](#38-mojo-modular)
  - [3.9 Comparison](#39-comparison)
- [4. Key Design Decisions](#4-key-design-decisions)
- [5. API Specification](#5-api-specification)
- [6. References](#6-references)

## 1. Requirements

(sorted by priority)
- Express kernels in terms of **tensors** (ND-arrays in global memory) and **tiles** (fixed-shape chunks in on-chip memory). Buffer addresses, TPipe/TQue lifecycles, and synchronization barriers are not exposed to the user.
- Performance pyAsc2 code is 90% of optimized AscendC operators
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

**910C** is two 910B dies in one package — same Da Vinci architecture, same AIC/AIV
model, same challenges as A2/A3. This section therefore focuses on 950 (A5).

950 is not an incremental refresh of 910b. It introduces new hardware capabilities
(GM atomics, register-addressable SIMD, kernel-side printf) and two new AscendC
programming surfaces (covered in §3.1.2). Findings below are derived from inspection
of the CANN 9 SDK preview. <sup>[[58]](#ref-58)</sup>

#### 2.4.1 Hardware Capability Deltas

950 adds the following hardware capabilities relative to 910b: <sup>[[58]](#ref-58)</sup>

- **Register file is first-class.** The vector register file is
  now exposed
- **Global-memory atomics (new hardware).** 950 adds hardware atomics —
  CAS, Add, Max, Min, Or, And, Xor — operating on both global memory and UB.
  - **Low-precision matmul first-class.** Bit-mode matmul is now exposed as
  a dedicated hardware path, consistent with the new MX-family data formats
  (below).
- **New data formats.** MXFP4, MXFP8, HiF8 (in addition to FP16 / BF16 / INT8);
  the type-conversion matrix grew roughly 3× to cover the new dtypes,
  including saturating casts.
- **AIC↔AIV ring buffer moves on-chip.** The tag-based dual-channel FIFO
  between Cube and Vector cores (TPUSH/TPOP) changed backing location: on
  910B/C the ring buffer lives in Global Memory (DMA in and out), on 950 it
  lives in the consumer's on-chip SRAM — **zero-copy**: the consumer
  dereferences slot data directly. <sup>[[4]](#ref-4)</sup>
- **Cross-core address resolution.** On-chip ring-buffer placement means
  the producer must know the consumer's SRAM base. Resolved via per-function
  constants plus an allocator-reserved region in the consumer's SRAM.
- **Memory access granularity**: 512 bytes → 128 bytes.
- **Pipeline sync unchanged at the hardware level.** Handshake between
  memory engines and vector cores remains explicit — no new hardware
  barriers.

#### 2.4.2 Implications for the Three DSL Challenges

- **Sync insertion.** Three surfaces now coexist with different sync models:
  basic_api keeps explicit `set_flag` / `wait_flag` and TPipe events;
  MicroAPI adds `MaskReg` predication at the register-tile granularity
  (masks are not barriers — barriers remain basic_api's responsibility);
  SIMT-API adds warp-level primitives and GM atomics for fine-grained sync.
  A DSL must decide which model to expose (or hide) and stay coherent across
  surfaces.
- **Ping-pong.** The on-chip TPUSH/TPOP ring buffer on A5 removes the GM
  round-trip for Cube↔Vector handoff — the cost model of a pipelined stage
  shifts substantially. On MicroAPI, pipelining is a concern at register-tile
  granularity, not UB-tile granularity, because loops iterate over `RegTensor`
  chunks.
- **UB memory allocation.** Two new address-spaces to plan: the register file
  (first-class in MicroAPI) and the reserved consumer SRAM segment for the
  TPUSH/TPOP ring buffer (on A5, a fixed exclusion zone inside UB or L1).
  The DSL allocator must model both.
- **Portability.** Targeting only basic_api is the conservative choice but
  forfeits 950's register-file throughput and GM atomics. Targeting MicroAPI
  reaches the register file but requires c310 (950-only builds). A DSL that
  claims to target 910b + 950 must lower to basic_api only, or implement
  per-target lowering.

## 3. Programming Model Analysis

### 3.1 AscendC

AscendC is the official C++ kernel language for Ascend NPU and the compilation
target for PyAsc2. Understanding how it handles the three key challenges defines
the baseline that PyAsc2 must improve upon.

AscendC is not a single programming model — it has evolved with the hardware.
On 910B/C the only surface is `basic_api` (TPipe/TQue, memory-centric). On 950
two additional surfaces appeared — MicroAPI (register-tensor SIMD with
predication) and SIMT-API (CUDA-like per-thread scalar) — neither of which
compiles for 910b. We therefore split this section along the 910B/C vs 950
axis: §3.1.1 covers `basic_api` behavior that applies to both targets (content
written against 910b c220, unchanged on c310); §3.1.2 covers what is new and
950-exclusive.

#### 3.1.1 AscendC on 910B/C

On 910B/C there is one surface: `basic_api` via build-mode `c220`. The three
key challenges are analyzed below in this surface.

##### Sync Insertion

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

**Conclusion**: AscendC doesn't solve the challenge — it exposes it. The user is responsible for every barrier manually. This is a source of bugs.

##### Ping-pong

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
Depth 2 = two slots in UB. While Vector works on ping, MTE2 loads into pong. But: User chooses queue depth manually; User structures the loop manually to achieve overlap; Compiler doesn't help

**Conclusion**: AscendC supports ping-pong, but requires manual orchestration. No automatic loop body partitioning.

##### UB Memory Allocation and Reuse

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

#### 3.1.2 AscendC on 950

On 950, `basic_api` still exists — the same `TPipe` / `TQue` / `LocalTensor`
programming style analyzed in §3.1.1 is available via build-mode `c310`, and
the three-challenge analysis above carries over unchanged. What is new on 950
are (a) c310 deltas inside `basic_api`, and (b) two brand-new API surfaces
(MicroAPI, SIMT-API) that are unavailable on 910b. <sup>[[58]](#ref-58)</sup>

The three surfaces stack as follows:

| Surface | 910b (c220) | 950 (c310) | Style |
|---|---|---|---|
| **basic_api** | ✓ 35 impl files | ✓ 39 impl files | Hardware intrinsics, memory-centric (`__ubuf__` ptrs + explicit `repeatTime` / `BlkStride` / `RepStride`) |
| **MicroAPI** | ✗ absent | ✓ 20 interface files + `dav_c310/` backend | Register-tensor SIMD functional. `RegTensor<T>` values, `MaskReg` predication, `LoadAlign`/`StoreAlign`, arch-dispatched via `__NPU_ARCH__` |
| **SIMT-API** | ✗ absent | ✓ 21 files + `dav_c310/` backend | CUDA-like per-thread scalar. Per-thread values, atomics on `__gm__`/`__ubuf__`, warp-level primitives |

Canonical example — same op (`Relu`) in each style:

```cpp
// 910b basic_api (dav_c220)
template <typename T>
void ReluIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src,
                        uint8_t repeatTime, const UnaryRepeatParams& p) {
    vrelu(dst, src, repeatTime,
          p.dstBlkStride, p.srcBlkStride, p.dstRepStride, p.srcRepStride);
}

// 950 MicroAPI (micro_api/dav_c310)
namespace MicroAPI {
template <typename T, MaskMergeMode mode, typename U>
void Relu(U& dstReg, U& srcReg, MaskReg& mask) {      // U = RegTensor<T>
    ReluImpl<T, mode, U>(dstReg, srcReg, mask);
}
}

// 950 SIMT-API (simt_api/dav_c310)
template <typename T>
T AbsImpl(T x) { return (x < 0) ? -x : x; }           // scalar on thread
```

A kernel written against MicroAPI or SIMT-API **will not compile** for 910b —
both surfaces are 950-exclusive and both require the c310 build-mode. Choosing
a surface is therefore a portability decision a DSL must make explicit.

##### basic_api on c310 — what changed

The programming model is the same; the surface  gained capabilities:
**TODO-Claude** dont like naming abstraction level by "surfaces". Change across doc to "level" 

The three-challenge baseline (manual sync, manual ping-pong, manual UB layout)
is unchanged at the basic_api surface on 950. 

##### MicroAPI — register-tensor SIMD with predication (950-exclusive)

MicroAPI operates on **register tensors**, not memory. The primitives take
`RegTensor<T>` values, a first-class `MaskReg` for predication, and
`LoadAlign` / `StoreAlign` to move data between UB and registers. The
MicroAPI interface dispatches to per-arch backends at compile time via
`__NPU_ARCH__`:
**TODO-Claude** Add more complex code example than just relu. at least softmax
```cpp
// micro_api/kernel_micro_vec_unary_intf_impl.h
#if   __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_vec_unary_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_vec_unary_impl.h"
#elif __NPU_ARCH__ == 5102
#include "micro_api/dav_m510/kernel_micro_vec_unary_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_vec_unary_impl.h"   // 950
#endif

namespace MicroAPI {
template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Relu(U& dstReg, U& srcReg, MaskReg& mask) {
    ReluImpl<T, mode, U>(dstReg, srcReg, mask);
}
}
```

The **caller pattern** shipped in c310's `basic_api/dav_c310/kernel_operator_vec_unary_impl.h`
shows how MicroAPI ops compose into a kernel: a loop iterates over register-tile
chunks, each chunk loaded via `LoadAlign`, processed under a predication mask,
and stored via `StoreAlign`.

```cpp
for (uint16_t i = 0; i < repeatTime; ++i) {
    mask = MicroAPI::UpdateMask<T, RegType::trait>(sreg);
    MicroAPI::LoadAlign(srcReg, src + i * repeatStride);
    func(dstReg, srcReg, mask);                        // e.g. MicroAPI::Relu
    MicroAPI::StoreAlign(dst + i * repeatStride, dstReg, mask);
}
```

**Sync Insertion.** MicroAPI does not introduce automatic barriers. The
`MaskReg` predication controls which lanes of a register tile participate in
an op — it is not a barrier. Cross-unit synchronization between MTE/Vector/Cube
pipelines still relies on `set_flag` / `wait_flag` and TPipe events from
`basic_api`. **Conclusion**: sync is still the user's responsibility, and
MicroAPI kernels typically still layer over basic_api for data movement.

**Ping-pong.** Pipelining shifts from UB-tile granularity to register-tile
granularity. The caller loop above runs over `repeatTime` register chunks per
UB tile; overlap between load (MTE2) and compute (Vector) is still achieved
via ping-pong at the UB level (basic_api), but within a UB tile, MicroAPI adds
a second axis of latency hiding via chunked register operations.
**Conclusion**: MicroAPI does not replace UB-level ping-pong; it adds a finer
pipelining axis below it, both still manual.

**UB Memory Allocation.** MicroAPI introduces the register file as a planned
address space. `RegTensor<T>` values occupy vector registers; `LoadAlign` /
`StoreAlign` are the explicit move operations between UB and registers. UB
layout itself is still managed by the basic_api `TPipe` / `TQue` machinery.
**Conclusion**: UB allocation remains a basic_api concern; MicroAPI adds a
second-tier register-allocation problem that the programmer (or a DSL compiler)
must solve — evidenced by softmax's end-to-end rewrite from `membase/` to
`regbase/` on 950.

**Overall**: MicroAPI is a new surface *below* basic_api, not a replacement.
It exposes the register file and predication; it does not automate sync or
memory planning.

##### SIMT-API — brief note (not a pyasc2 target)

SIMT-API is a CUDA-like per-thread scalar surface (`AbsImpl`, `AtomicCasImpl`,
warp-level primitives, a CPU-debug shim). It is 950-exclusive and represents
a different programming model entirely: instead of tile-level intrinsics it
exposes per-thread operations, with the compiler and hardware responsible for
SIMT-style lane grouping and latency hiding.

**pyasc2 does not target SIMT-API.** SIMT-style per-thread code on Ascend is
not expected to reach the performance ceiling needed by a pyasc2 kernel; the
DSL's goal is ≥90% of peak hardware potential and SIMT lowers that ceiling by
giving up tile-level orchestration. SIMT-API is noted here for completeness
and because its atomics are mirrored into basic_api.

##### pyasc2 implication

pyasc2 targets `basic_api` (baseline, portable across 910b and 950) and
MicroAPI (950-only, for register-file throughput and the new dtype matrix).
SIMT-API is out of scope. A pyasc2 program targeting 950 must lower to a
c310 build that mixes basic_api for data movement and TPipe orchestration with
MicroAPI for compute inside tiles; targeting 910b lowers to basic_api only
(c220). This surface split is the core portability decision §4 must make
explicit.

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


> TODO: Exmplanation above is about MMA. Better here to insert GEMM example (not vecadd)
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

> TODO: if possible to get references - insert perf numbers B200 GEMM (matmul kernels)
**Conclusion**: Sync is fully automated and not even expressible by the user.
cuTile is the most restrictive model — no escape hatch for manual barriers.

#### Ping-Pong

The user writes a simple loop. The TileIR compiler transforms it into a pipelined loop
through three passes: <sup>[[15]](#ref-15) [[16]](#ref-16)</sup>


1. `convert-tileaa-to-tileas` — converts tiled loads into async loads with pipeline ops
> TODO: clarify if this "tiled loads are about MMA or vector ops
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
> TODO: about not clear. what is 100us? what is matrix operands "competed"?

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
> TODO: Clarify why Buffer reuse and validation is not yet default option. 


### 3.5 Triton-Ascend

Triton-Ascend is Huawei's fork of Triton for Ascend NPU. From the user's perspective —
standard Triton Python API. Internally, a completely different compilation path. <sup>[[28]](#ref-28)</sup>

Compilation pipeline — does not use TTGIR (GPU-specific), stays at TTIR: <sup>[[28]](#ref-28) [[29]](#ref-29)</sup>
```
TTIR → triton-to-linalg → Linalg IR → HFusion → HIVM
  → HIVMToStandard (HIVM ops → CCE device library calls, e.g. vadd_2d_f16)
  → LLVM IR → kernel.o (linked with CCE libs, loaded via ascendcl)
```

HIVM (Hybrid ISA Virtual Machine) is the core dialect of AscendNPU-IR
(open-source: `Ascend/AscendNPU-IR` on gitcode.com) — abstracts Ascend computation,
data movement, and synchronization at tile level. Built on standard MLIR dialects:
`Linalg`, `MemRef`, `SCF`, `Bufferization`, `Tensor`. <sup>[[29]](#ref-29)</sup>

Confirmed working on Ascend (with `torch_npu`): matmul (`tl.dot`), fused softmax,
layer norm, fused attention (Flash Attention v2), vector add. <sup>[[27]](#ref-27)</sup>

#### Sync Insertion

Automated — same as standard Triton, no explicit barriers from the user. <sup>[[26]](#ref-26)</sup>

Internally handled by HIVM passes (`bishengir/lib/Dialect/HIVM/Transforms/`): <sup>[[30]](#ref-30)</sup>
- `InjectSync/` — intra-core sync: inserts `set_flag`/`wait_flag` between MTE and
  compute pipelines based on data dependency analysis
- `GraphSyncSolver/` — graph-based solver for optimal barrier placement,
  minimizing redundant barriers while preserving correctness
- `InjectBlockSync.cpp` — inter-block sync for cross-core data dependencies
- `SplitMixKernel.cpp` — splits CV-fused kernels into separate AIC (cube) and
  AIV (vector) functions, inserting cross-core sync at data exchange points

Confirmed by lit tests: `inject-sync.mlir`, `sync-solver.mlir`,
`sync-solver-cross-core.mlir`, `inject-block-sync.mlir`. <sup>[[30]](#ref-30)</sup>

Key Ascend-specific adaptation at user level: grid is fixed to the number of
physical cores (not thousands of blocks like GPU). For large data, two-level
tiling with `BLOCK_SIZE_SUB` — an inner loop to fit 192 KB UB: <sup>[[26]](#ref-26)</sup>

```python
@triton.jit
def kernel(inp, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    for block_idx in range(pid, num_blocks, NUM_CORE):  # round-robin across cores
        base = block_idx * BLOCK_SIZE
        for sub_idx in range(BLOCK_SIZE // BLOCK_SIZE_SUB):  # fit UB
            offs = base + sub_idx * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
            x = tl.load(inp + offs, mask=offs < N)
            tl.store(out + offs, x, mask=offs < N)
```

**What works as-is:** small-scale kernels where one block fits in UB.
**What requires adaptation:** large data (add BLOCK_SIZE_SUB loop),
grid sizing (fix to physical core count), i32/i64 comparisons (cast to float32),
tail axis alignment (32-byte for vector, 512-byte for CV ops). <sup>[[26]](#ref-26)</sup>

**Conclusion**: Sync is automated. Standard Triton kernels can run on Ascend
for small-scale ops; large-scale kernels need two-level tiling and grid adaptation.

#### Ping-Pong

multiBuffer is enabled by default — no user annotation required. <sup>[[26]](#ref-26)</sup>

Implemented by HIVM passes (`bishengir/lib/Dialect/HIVM/Transforms/`): <sup>[[30]](#ref-30)</sup>
- `EnableMultiBuffer.cpp` — enables double buffering for eligible allocations
- `MarkMultiBuffer.cpp` — marks which buffers should be multi-buffered based
  on access patterns and loop structure
- `CVPipelining.cpp` — reorders cube and vector code to enable CV core pipeline
  parallelism (load on one core while other computes)
- `OptMemPlanForPipeline.cpp` — adjusts memory plan to accommodate pipelined execution

The `BLOCK_SIZE_SUB` inner loop creates the pipelining opportunity: each sub-block
iteration is a natural stage for overlap. After doublebuffer is enabled, UB capacity
is halved (192 KB → 96 KB effective). <sup>[[26]](#ref-26)</sup>

Confirmed by lit tests: `enable-multi-buffer.mlir`, `mark-multi-buffer.mlir`,
`cv-pipelining.mlir`. <sup>[[30]](#ref-30)</sup>

**Conclusion**: Ping-pong is fully automated via multiBuffer (on by default).
No `num_stages` hint like Triton GPU, no `T.Pipelined` like TileLang-Ascend.

#### UB Memory Allocation and Reuse

UB allocation is handled by the `hivm-plan-memory` pass. Two-phase approach: <sup>[[30]](#ref-30)</sup>
1. **`MemLivenessAnalysis`** — uses standard MLIR `Liveness` analysis, extended with
   Ascend-specific buffer aliasing, multi-buffer tracking, and gen/kill maps
2. **`MemPlan`** — assigns UB offsets based on liveness, reuses non-overlapping buffers,
   handles multi-buffer (ping-pong) reuse scenarios

Hardware limits hardcoded in `PlanMemory.cpp`: UB = 192 KB, L1 = 512 KB, L0C = 128 KB.
Overflow caught at compile time. <sup>[[30]](#ref-30)</sup>

Additional HIVM passes (`bishengir/lib/Dialect/HIVM/Transforms/`): <sup>[[30]](#ref-30)</sup>
- `AutoInferBufferSize.cpp` — automatic buffer size inference
- `SetBufferSize.cpp` — buffer size assignment after inference

Confirmed by lit tests: `plan-memory.mlir`, `hivm-auto-infer-buffer-size.mlir`,
`hivm-set-buffer-size.mlir`. <sup>[[30]](#ref-30)</sup>

**Conclusion**: UB allocation is fully automated — standard MLIR liveness analysis
extended with Ascend-specific buffer reuse. GPU Triton's `AllocationAnalysis` is not used;
the entire path is Ascend-specific via HIVM dialect.

### 3.6 PyPTO

#### 3.6.1 PyPTO-main

PyPTO (Parallel Tensor/Tile Operation) is a tile-based programming framework for
Ascend NPU. The user writes tensor-level code; the compiler handles tiling, memory,
sync, and pipelining automatically. <sup>[[31]](#ref-31)</sup>

Pipeline (C++ framework, not MLIR): <sup>[[31]](#ref-31)</sup>
```
Python (@pypto.frontend.jit) → Tensor Graph → Tile Graph → Block Graph
  → Execution Graph → PTO Virtual Instructions → CANN Backend → NPU binary
```

User-facing code — no sync, no memory management, no barriers:
```python
@pypto.frontend.jit
def matmul_kernel(a: pypto.Tensor([], pypto.DT_FP32),
                  b: pypto.Tensor([], pypto.DT_FP32),
                  out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out[:] = pypto.matmul(a, b, pypto.DT_FP32)
```

#### Sync Insertion

Fully automated. The `InsertSync` pass at Block Graph level performs: <sup>[[32]](#ref-32)</sup>
- RAW/WAW/WAR data dependency analysis using interval trees (`DataDependencySearcher`)
- Pipe-aware `set_flag`/`wait_flag` injection across all pipeline pairs
  (AIC_MTE2, AIC_M, AIV_MTE2, AIV_V, AIV_MTE3, etc.)
- Event ID allocation with deadlock detection and recovery
- Cross-core sync for CV-fused kernels

Post-optimization: `tune_sync_for_vf` relaxes unnecessary barriers
for vector-fusion subgraphs. <sup>[[33]](#ref-33)</sup>

**Conclusion**: User writes zero sync primitives. The compiler performs full
pipeline-aware dependency analysis and barrier injection.

#### Ping-Pong

Automated at Tile Graph level. The `n_buffer_merge` pass doubles buffers
for overlapping data transfer and computation. <sup>[[34]](#ref-34)</sup>
Combined with `schedule_ooo` (out-of-order scheduling) and `add_alloc` at
Block Graph level for pipeline parallelism. <sup>[[35]](#ref-35)</sup>

The user controls tile shapes via `pypto.set_vec_tile_shapes()` or
`pypto.set_cube_tile_shapes()` — this indirectly determines buffer count
and pipeline depth. No explicit `num_stages` or `T.Pipelined`.

**Conclusion**: Ping-pong is fully automated. User only sets tile shapes.

#### UB Memory Allocation and Reuse

Multi-stage memory management across the compilation pipeline:
- **Tile Graph**: `assign_memory_type` — decides which memory level each tile
  uses (DDR, L1, L0A/B/C, UB) based on operation type <sup>[[36]](#ref-36)</sup>
- **Block Graph**: `add_alloc` / `remove_alloc` — inserts buffer allocations <sup>[[35]](#ref-35)</sup>;
  `memory_reuse/` contains liveness-based reuse passes including
  `global_memory_reuse.cpp`, `merge_src_dst_buffer.cpp`, and connection matrix
  analysis for non-overlapping buffer sharing <sup>[[37]](#ref-37)</sup>

**Conclusion**: UB allocation is fully automated — memory space assignment,
buffer allocation, and liveness-based reuse are all compiler-managed.
No user-visible memory APIs.

#### 3.6.2 PyPTOv3 (redesign)

PyPTOv3 is a ground-up rewrite of PyPTO with a new IR system, new DSL, and
multi-level abstraction. Active development at github.com/hw-native-sys/pypto.
Targets 910B and 950. <sup>[[38]](#ref-38)</sup>

Pipeline — custom C++ AST-based IR for passes, MLIR as output format: <sup>[[38]](#ref-38)</sup>
```
Python (@pl.program / @pl.function) → Custom C++ IR → SSA conversion
  → passes (tensor→tile, memory, sync)
  → PTO codegen → MLIR text (.pto file: func.func, arith.*, memref)
  → ptoas (PTO assembler) → NPU binary
```

Key design difference from PyPTO-main: **explicit memory spaces** in the user API.
The user specifies data movement targets — compiler handles the rest:
```python
@pl.program
class MatmulExample:
    @pl.function
    def main(self, a: pl.Tensor[[M, K], pl.BF16],
                   b: pl.Tensor[[K, N], pl.BF16]) -> pl.Tensor[[M, N], pl.FP32]:
        a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)   # DDR → L1
        b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        a_l0a = pl.move(a_l1, target_memory=pl.Mem.Left)                 # L1 → L0A
        b_l0b = pl.move(b_l1, target_memory=pl.Mem.Right)                # L1 → L0B
        c_acc = pl.matmul(a_l0a, b_l0b)                                  # → L0C (Acc)
        return pl.store(c_acc, [0, 0], out)
```

#### Sync Insertion

Fully automated via a 4-phase `SyncInserter` algorithm on the SSA IR: <sup>[[39]](#ref-39)</sup>

1. **CollectSyncPairs** — walks the AST tracking last writers/readers per MemRef.
   Detects RAW, WAW, WAR hazards across statements. Handles `if`/`else` branches
   via state merging, `for` loops via fixed-point iteration. Removes transitive
   and linear redundant pairs.
2. **AdjustScopeCrossings** — moves sync points when producer and consumer are in
   different control flow scopes (e.g. one inside `if`, other outside).
3. **AssignEventIds** — allocates from 8 hardware event IDs per pipe pair
   (`EventIdManager`), with position-based free tracking.
4. **ApplyInsertions** — mutates AST to insert `set_flag`/`wait_flag` calls
   at computed positions.

Pipeline assignment per op is backend-specific (`backend->InferPipe(call)`) —
supports 910B and 950 with different pipe configurations. <sup>[[39]](#ref-39)</sup>

Cross-core sync for mixed AIC/AIV kernels via `expand_mixed_kernel_pass`. <sup>[[40]](#ref-40)</sup>

**Conclusion**: User writes no sync. The compiler performs full MemRef-level
dependency analysis with scope-aware insertion and hardware event ID management.

#### Ping-Pong

Handled by the optimization pipeline (Default strategy): <sup>[[38]](#ref-38)</sup>
1. `InitMemRef` — assigns memory spaces and inserts buffer allocations <sup>[[41]](#ref-41)</sup>
2. `MemoryReuse` — shares buffers with non-overlapping lifetimes <sup>[[42]](#ref-42)</sup>
3. `LegalizePTOBufferReuse` — legalizes buffer reuse for PTO backend <sup>[[43]](#ref-43)</sup>

No explicit `num_stages` or `T.Pipelined`. The Qwen3 decode example (pypto-lib PR #25)
shows manual tile sizing for TILELET (2 KB vector) and TILE (16 KB cube) budgets —
the user controls chunk sizes, the compiler handles pipelining. <sup>[[44]](#ref-44)</sup>

**Conclusion**: Pipelining is compiler-managed. User controls tile shapes
to fit hardware budgets.

#### UB Memory Allocation and Reuse

Three-pass approach: <sup>[[45]](#ref-45) [[41]](#ref-41) [[46]](#ref-46)</sup>
1. `InferTileMemorySpace` — infers memory space (Vec/Mat/Left/Right/Acc) for each
   tile based on operation semantics <sup>[[45]](#ref-45)</sup>
2. `InitMemRef` — creates MemRef descriptors with sizes, assigns concrete memory
   spaces, inserts alloc/free points <sup>[[41]](#ref-41)</sup>
3. `AllocateMemoryAddr` — assigns concrete byte addresses within each memory
   space <sup>[[46]](#ref-46)</sup>

Between steps 2 and 3, `MemoryReuse` performs liveness analysis and merges
non-overlapping buffers into shared memory regions. <sup>[[42]](#ref-42)</sup>

**Conclusion**: UB allocation is fully automated — infer space, create buffers,
reuse by liveness, assign addresses. User specifies `target_memory` on loads
but never manages addresses or buffer sizes.

### 3.7 Pallas (JAX)

Pallas is Google's kernel DSL for TPU (and GPU via Triton backend).
Part of JAX. No decorator, no custom parser — kernels are regular Python functions
passed to `pl.pallas_call()`, which triggers JAX tracing. <sup>[[47]](#ref-47)</sup>

Pipeline: <sup>[[48]](#ref-48)</sup>
```
pl.pallas_call(kernel_fn, grid, BlockSpecs)
  → JAX tracing (symbolic execution) → Jaxpr (JAX IR)
  → Pallas IR (Jaxpr + BlockSpec + grid)
  → Mosaic compiler (MLIR: vector + arith + tpu dialects)
  → TPU IR → LLO → TPU machine code
```

User-facing code — kernel is plain Python, `pallas_call` defines tiling:
```python
def matmul_kernel(a_ref, b_ref, o_ref):
    o_ref[:, :] = jnp.dot(a_ref[:, :], b_ref[:, :])

result = pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((64, 64), jnp.float32),
    grid=(1,),
    in_specs=[pl.BlockSpec((64, 128), lambda i: (0, 0)),
              pl.BlockSpec((128, 64), lambda i: (0, 0))],
    out_specs=pl.BlockSpec((64, 64), lambda i: (0, 0)),
)(a, b)
```

#### Sync Insertion

TPU has two compute units (MXU for matmul, VPU for vector ops) but they share
a **single instruction stream** — no parallel pipeline hazards between them. <sup>[[49]](#ref-49)</sup>

Pipelining on TPU is **DMA vs compute** overlap: the DMA engine loads the next tile
while compute processes the current one. This is the only source of concurrency
within a core. <sup>[[49]](#ref-49)</sup>

**Automatic (default):** user writes `pallas_call` with `BlockSpec`. Mosaic compiler
generates all DMA operations, semaphores, and double-buffer swaps from the BlockSpec
— user never writes sync code: <sup>[[47]](#ref-47)</sup>
```python
def kernel(a_ref, b_ref, o_ref):
    o_ref[:, :] = a_ref[:, :] + b_ref[:, :]  # no DMA, no sync

pl.pallas_call(kernel, grid=(N,),
    in_specs=[pl.BlockSpec((128,), lambda i: (i,))],
    out_specs=pl.BlockSpec((128,), lambda i: (i,)),
    ...)(a, b)
# Mosaic: BlockSpec → DMA schedule → semaphore insertion → double buffering
```

**Manual (expert):** for advanced patterns (paged attention, custom prefetch),
user writes `pltpu.async_copy` + semaphores directly: <sup>[[50]](#ref-50)</sup>
```python
sem = pltpu.SemaphoreType.DMA((2,))
pltpu.async_copy(hbm_ref.at[slice], vmem_buffer, sem).wait()
```

Compare: Ascend has 3+ parallel units (MTE2, Cube, Vector, MTE3) requiring
fine-grained `set_flag`/`wait_flag` between each pair. TPU has 2 (DMA + compute).

**Conclusion**: Sync is compiler-generated from BlockSpec by default.
Manual semaphores available as escape hatch. Simpler than Ascend — only
DMA↔compute overlap, no Cube↔Vector hazards.

#### Ping-Pong

TPU on-chip memory: **VMEM** (Vector Memory) — 32 MB SRAM per core,
equivalent to Ascend's UB but ~170× larger (32 MB vs 192 KB). <sup>[[49]](#ref-49)</sup>

Default: 2-level double buffering for all inputs and outputs, generated
automatically by Mosaic from BlockSpec. Sufficient for most kernels because
VMEM is large enough that double buffering rarely causes pressure. <sup>[[49]](#ref-49)</sup>

Advanced control via `pltpu.emit_pipeline` — supports nested pipelines and
lookahead prefetch (fetch next block as soon as a buffer slot is free, not
just one iteration ahead). <sup>[[51]](#ref-51)</sup>

**Conclusion**: Ping-pong is automatic (2-stage default from BlockSpec).
Advanced pipelining API available but rarely needed due to large VMEM.

#### UB Memory Allocation and Reuse

TPU memory hierarchy: <sup>[[49]](#ref-49)</sup>
- **HBM** — main memory (GBs), slow
- **VMEM** — 32 MB vector SRAM per core, holds tiles during compute
- **VREG** — vector registers (8×128 tiles for fp32), fastest
- **SMEM** — ~4-8 KB scalar SRAM, for control/metadata only

User controls tiling via `BlockSpec(shape, index_map)` — this determines how much
VMEM each tile consumes. The compiler handles VMEM allocation internally. <sup>[[47]](#ref-47)</sup>

Tiling constraints: last 2 dimensions must be divisible by 8 and 128 respectively
(matching 8×128 vector register shape). <sup>[[49]](#ref-49)</sup>

No user-visible VMEM address management — unlike Ascend's manual UB offset planning.
If total VMEM usage exceeds 32 MB (including double-buffer overhead), compilation fails.

**Conclusion**: VMEM allocation is fully compiler-managed. User only controls tile
shapes via BlockSpec. The 32 MB budget is ~170× Ascend's UB — memory pressure is
rarely an issue on TPU.

### 3.8 Mojo (Modular)

Mojo is a Python superset by Modular (Chris Lattner). Systems-level language
with explicit GPU kernel programming. Uses MLIR internally via KGEN compiler. <sup>[[52]](#ref-52)</sup>

Pipeline: <sup>[[52]](#ref-52)</sup>
```
Mojo source → KGEN compiler (MLIR-based) → platform-specific backend
  → PTX (NVIDIA) / HIP (AMD) / Metal (Apple) → GPU binary
```

Abstraction level: **between CUDA and Triton**. Thread-centric like CUDA
(explicit threads, blocks, sync), but with tile-level abstractions
(`TileTensor`, `TileIO`, `TilePipeline`) for structured kernels. <sup>[[53]](#ref-53)</sup>

Supports NVIDIA, AMD, and Apple GPUs. No TPU or Ascend support.

Simple kernel — CUDA-like thread model, Python-like syntax: <sup>[[55]](#ref-55)</sup>
```mojo
def scalar_add(vector: UnsafePointer[Float32], size: Int, scalar: Float32):
    idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < size:
        vector[idx] += scalar

ctx.enqueue_function[scalar_add](device_buffer, 20, Float32(5.0),
                                  grid_dim=1, block_dim=20)
```

#### Sync Insertion

**Manual.** User writes explicit barriers — closer to CUDA than to Triton: <sup>[[54]](#ref-54)</sup>
- `barrier()` — block-level sync (like CUDA `__syncthreads()`)
- `syncwarp()` — warp-level sync
- `named_barrier(id, count)` — multiple independent barriers

Platform-specific differences managed by user: <sup>[[53]](#ref-53)</sup>
- **NVIDIA Blackwell**: hardware `mbarrier` with automatic byte counting for TMA
- **AMD MI300X**: no `mbarrier` — explicit atomic counters with `s_sleep 0` yield

Structured kernel pattern ("three pillars") separates concerns: <sup>[[53]](#ref-53)</sup>
- `TileIO` — data movement between memory levels
- `TilePipeline` — producer-consumer with `acquire`/`release` semantics
- `TileOp` — MMA/compute instructions

```mojo
# Three warp roles execute in parallel:
# Load warp:    acquires producer stage → TMA load → signals completion
# MMA warp:     waits on input pipeline → computes → signals output pipeline
# Epilogue warp: reads from TMEM → writes to global memory
with producer.acquire() as tiles:
    tile_io.load(tiles, barrier, k_coord)  # explicit DMA + barrier
```

**Conclusion**: Sync is fully manual. The "three pillars" pattern provides structure
but the user orchestrates all barriers. No compiler-automated sync insertion.

#### Ping-Pong

**Manual.** Mojo intentionally rejects Triton-style automation — their position:
automatic compilation improves accessibility but hits a performance ceiling for
production inference, forcing developers back to low-level code. <sup>[[56]](#ref-56)</sup>

Instead, the `TilePipeline` abstraction provides structured manual control
via `acquire`/`release` context managers: <sup>[[53]](#ref-53)</sup>

```mojo
with producer.acquire() as tiles:
    tile_io.load(tiles, barrier, k_coord)   # load next tile
# release: signals consumer that data is ready

with consumer.acquire() as tiles:
    tile_op.mma(tiles)                       # compute on current tile
# release: signals producer that buffer is free
```

Three warp roles execute in parallel: load warp (DMA), MMA warp (compute),
epilogue warp (store). User assigns warps to roles explicitly. <sup>[[53]](#ref-53)</sup>

Result: 48% less code than CUTLASS with identical performance, but still
fully user-orchestrated. <sup>[[56]](#ref-56)</sup>

**Conclusion**: Pipelining is user-structured via `TilePipeline`, not compiler-generated.
Design choice: structured manual control over automation with escape hatches.

#### UB Memory Allocation and Reuse

**Manual.** User declares memory explicitly: <sup>[[55]](#ref-55)</sup>
- **Shared memory**: `Shared` type for block-visible scratchpad
- **Registers**: small compile-time-sized arrays auto-allocated by compiler
- **Global memory**: `DeviceBuffer` types

`TileTensor` carries compile-time layout information (shape, stride, swizzle)
and memory address space placement — enabling the compiler to verify layouts
at compile time without managing allocation itself. <sup>[[57]](#ref-57)</sup>

Platform-specific memory handled by user: <sup>[[53]](#ref-53)</sup>
- **NVIDIA Blackwell**: TMEM (256 KB) managed via explicit context managers —
  entering signals readiness, exiting deallocates
- **AMD MI300X**: accumulators live in VGPRs, compiler manages register allocation

No liveness-based buffer reuse by the compiler. User controls all allocation
and lifetime.

**Conclusion**: Memory is user-managed. `TileTensor` provides type-safe layout
verification but not automatic allocation or reuse. Consistent with Mojo's
philosophy: structured control over automation.

### 3.9 Comparison

| | **AscendC** | **Triton** | **cuTile** | **TileLang-Ascend** | **Triton-Ascend** | **PyPTO-main** | **PyPTOv3** | **Pallas** | **Mojo** |
|---|---|---|---|---|---|---|---|---|---|
| **Abstraction level** | Instruction | Tile/block | Tile | Tile (hybrid) | Tile/block | Tensor | Tensor + Tile (expert) | Tile (BlockSpec) | Thread + Tile (structured) |
| **Sync insertion** | Manual | Auto | Auto (no escape) | Hybrid (auto opt-in) | Auto (HIVM) | Auto | Auto (4-phase) | Auto (Mosaic from BlockSpec) | Manual (three pillars) |
| **Ping-pong** | Manual | Auto (`num_stages`) | Auto (no hint) | `T.Pipelined(num_stages)` | Auto (multiBuffer) | Auto | Auto | Auto (2-stage default) | Manual (`TilePipeline`) |
| **On-chip memory** | UB 192 KB | Shared ~228 KB | Shared + TMEM 256 KB | UB 192 KB | UB 192 KB | UB 192 KB | UB 192 KB | VMEM 32 MB | Shared (GPU) |
| **Memory mgmt** | Manual | Auto (AllocationAnalysis) | Auto | Hybrid (opt-in planning) | Auto (PlanMemory/HIVM) | Auto | Auto (3-pass) | Auto (BlockSpec) | Manual (TileTensor) |
| **Overflow detection** | Silent corruption | Compile-time error | Compile-time error | Partial (opt-in) | Compile-time error | ? | ? | Compile-time error | Compile-time (layout) |
| **IR type** | C++ (AscendC) | MLIR (TTIR/TTGIR) | MLIR (TileIR) | TVM TIR | MLIR (TTIR→HIVM) | Custom C++ graphs | Custom C++ AST | Jaxpr → Mosaic (MLIR) | MLIR (KGEN) |
| **User memory control** | Full manual | None | None | Explicit hierarchy | Indirect (BLOCK_SIZE) | None | `target_memory` hint | BlockSpec (tile shape) | Full manual (Shared, TMEM) |
| **Hardware** | Ascend 910B/C | NVIDIA/AMD/Intel | NVIDIA Blackwell | Ascend A2/A3 | Ascend A2/A3 | Ascend 910B | 910B, 950 | Google TPU | NVIDIA/AMD/Apple |

#### Key Observations for PyAsc2 Design

**1. Full automation is achievable on Ascend — performance cost varies.**
Among Ascend-targeting frameworks, Triton-Ascend, TileLang-Ascend, PyPTO-main, and
PyPTOv3 all automate sync, ping-pong, and UB allocation. But automation quality differs:
TileLang-Ascend's auto sync generates conservative `PipeBarrier<PIPE_ALL>` instead of
fine-grained flags. No published benchmarks compare automated vs hand-optimized AscendC
for the same kernels. The gap between "correct" and "optimal" automatic code is the main
engineering challenge for PyAsc2.

**2. The right abstraction level is tensor — with tile escape hatch.**
PyPTOv3's two-level design covers both use cases. Triton's tile-only model forces all
users into low-level thinking. cuTile's no-escape model limits advanced optimization.
PyAsc2 should follow PyPTOv3's approach: tensor-level default, tile-level available.

**3. Memory hierarchy hints > full manual control.**
cuTile (fully implicit) and AscendC (fully explicit) are the two extremes. PyPTOv3's
`target_memory=pl.Mem.Mat` is the sweet spot — the user declares intent, the compiler
handles mechanics.

**4. Liveness-based UB reuse is critical — more so than on GPU.**
On GPU, shared memory reuse pressure is low — up to 228 KB available, rarely exhausted.
On Ascend, UB is 192 KB (96 KB with double buffering) and almost always the bottleneck.
Suboptimal reuse forces smaller tiles or spills to global memory. PyAsc2 should provide
automatic reuse by default with an option for expert override of buffer placement.

**5. Compile-time UB overflow detection is non-negotiable.**
AscendC's silent corruption is the #1 developer pain point. Every modern framework
catches overflow at compile time. PyAsc2 must validate UB/L1/L0 fit at compile time
with clear error messages.

**6. `num_stages` vs fully automatic pipelining.**
Triton and TileLang-Ascend expose pipeline depth to users. cuTile, PyPTO-main, and
PyPTOv3 hide it entirely. For PyAsc2: default to automatic, provide `num_stages` as
optional expert hint.

**7. Ascend-native IR outperforms adapted GPU IR.**
Triton-Ascend's TTIR→HIVM path works but requires user-level rewrites (BLOCK_SIZE_SUB,
fixed grid, alignment). Native Ascend frameworks handle these in the compiler. PyAsc2
should be Ascend-native from the start.

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
| 26 | Triton-Ascend programming guide (gitcode.com primary repo) | https://gitcode.com/Ascend/triton-ascend/blob/main/docs/en/programming_guide.md |
| 27 | Triton-Ascend examples — confirmed on Ascend with torch_npu | https://gitcode.com/Ascend/triton-ascend/tree/main/docs/en/examples |
| 28 | Triton-Ascend architecture design and core features | https://gitcode.com/Ascend/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md |
| 29 | AscendNPU-IR architecture — HIVM/HFusion/HACC dialect definitions | https://gitcode.com/Ascend/AscendNPU-IR/blob/main/docs/source/en/introduction/architecture.md |
| 30 | AscendNPU-IR PlanMemory pass — liveness-based UB allocation | https://gitcode.com/Ascend/AscendNPU-IR/blob/main/bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp |
| 31 | PyPTO-main matmul example | https://gitcode.com/cann/pypto/blob/master/examples/01_beginner/compute/matmul_ops.py |
| 32 | PyPTO InsertSync pass — RAW/WAW/WAR dependency analysis | https://gitcode.com/cann/pypto/blob/master/framework/src/passes/block_graph_pass/insert_sync.cpp |
| 33 | PyPTO tune_sync_for_vf — barrier relaxation for vector fusion | https://gitcode.com/cann/pypto/blob/master/framework/src/passes/block_graph_pass/tune_sync_for_vf.cpp |
| 34 | PyPTO n_buffer_merge — double buffering at Tile Graph level | https://gitcode.com/cann/pypto/blob/master/framework/src/passes/tile_graph_pass/graph_partition/n_buffer_merge.cpp |
| 35 | PyPTO add_alloc / schedule_ooo — Block Graph allocation and scheduling | https://gitcode.com/cann/pypto/blob/master/framework/src/passes/block_graph_pass/schedule_ooo/add_alloc.cpp |
| 36 | PyPTO assign_memory_type — memory space assignment at Tile Graph | https://gitcode.com/cann/pypto/blob/master/framework/src/passes/tile_graph_pass/data_path/assign_memory_type.cpp |
| 37 | PyPTO memory_reuse — liveness-based buffer reuse | https://gitcode.com/cann/pypto/blob/master/framework/src/passes/block_graph_pass/memory_reuse/global_memory_reuse.cpp |
| 38 | PyPTOv3 language guide — DSL, memory hierarchy, optimization pipeline | https://github.com/hw-native-sys/pypto/blob/main/docs/en/user/01-language_guide.md |
| 39 | PyPTOv3 insert_sync_pass — 4-phase sync insertion algorithm | https://github.com/hw-native-sys/pypto/blob/main/src/ir/transforms/insert_sync_pass.cpp |
| 40 | PyPTOv3 expand_mixed_kernel_pass — AIC/AIV split + cross-core sync | https://github.com/hw-native-sys/pypto/blob/main/src/ir/transforms/expand_mixed_kernel_pass.cpp |
| 41 | PyPTOv3 init_memref — buffer allocation and memory space assignment | https://github.com/hw-native-sys/pypto/blob/main/src/ir/transforms/init_memref.cpp |
| 42 | PyPTOv3 memory_reuse_pass — liveness-based buffer sharing | https://github.com/hw-native-sys/pypto/blob/main/src/ir/transforms/memory_reuse_pass.cpp |
| 43 | PyPTOv3 legalize_pto_buffer_reuse — PTO backend buffer legalization | https://github.com/hw-native-sys/pypto/blob/main/src/ir/transforms/legalize_pto_buffer_reuse_pass.cpp |
| 44 | PyPTOv3 Qwen3 decode example — tilelet-aware tiling | https://github.com/hw-native-sys/pypto-lib/pull/25 |
| 45 | PyPTOv3 infer_tile_memory_space — memory space inference | https://github.com/hw-native-sys/pypto/blob/main/src/ir/transforms/infer_tile_memory_space_pass.cpp |
| 46 | PyPTOv3 allocate_memory_addr — concrete address assignment | https://github.com/hw-native-sys/pypto/blob/main/src/ir/transforms/allocate_memory_addr_pass.cpp |
| 47 | Pallas — JAX kernel language overview | https://jax.readthedocs.io/en/latest/pallas/index.html |
| 48 | Pallas design notes — compilation pipeline | https://jax.readthedocs.io/en/latest/pallas/design/design.html |
| 49 | Pallas TPU details — memory, sync, tiling constraints | https://docs.jax.dev/en/latest/pallas/tpu/details.html |
| 50 | Pallas paged attention kernel — async_copy + semaphore usage | https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py |
| 51 | Pallas TPU pipelining — emit_pipeline, lookahead prefetch | https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html |
| 52 | Mojo MLIR-based compilation — KGEN compiler (arXiv:2509.21039) | https://arxiv.org/abs/2509.21039 |
| 53 | Structured Mojo Kernels Part 2 — three pillars (TileIO, TilePipeline, TileOp) | https://www.modular.com/blog/structured-mojo-kernels-part-2-the-three-pillars |
| 54 | Mojo GPU sync primitives — barrier, syncwarp, named_barrier | https://docs.modular.com/mojo/manual/gpu/block-and-warp/ |
| 55 | Mojo GPU fundamentals — kernel model, thread indexing | https://docs.modular.com/mojo/manual/gpu/fundamentals/ |
| 56 | Structured Mojo Kernels Part 1 — design philosophy, performance | https://www.modular.com/blog/structured-mojo-kernels-part-1-peak-performance-half-the-code |
| 57 | TileTensor — parametric tile-level tensors in Mojo | https://www.modular.com/blog/tiletensor-part-1-safer-more-efficient-gpu-kernels |
| 58 | CANN 9 SDK preview — 950/A5 programming model findings (internal notes from SDK source inspection) | — |
