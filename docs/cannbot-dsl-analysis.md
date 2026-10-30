# CANNBot-DSL: Project and Programming Model Analysis

## One-Page Summary

**Main finding.** CANNBot-DSL and PyAsc v2/AscTile share an AscendC compilation target but make different decisions visible to the programmer. The strongest transfer opportunity is an optional tile-expression layer with compiler-managed realization, supported by diagnostics and functional debugging. Replacing the CANNBot backend is not the recommended starting point.

**Project and status.** CANNBot-DSL targets high-performance Ascend operators, including attention, quantization, recurrent kernels, and distributed-training components. This repository contains samples, tests, documentation, and packaging; its compiler/runtime comes from a separate wheel. The October 30, 2026 snapshot shows active DSL 0.8 migration and model-level fusion work. Open proposals indicate direction, not delivery commitments. Some reported defects concern 0.7/0.7.1 and must not be presented as verified 0.8 failures. See sections 2–4. [[3]](#ref-3) [[6]](#ref-6) [[16]](#ref-16)

**Abstraction boundary.** The documented AI Core path is Python → CANNIR/MLIR → AscendC → Bisheng/CCEC → device code. AscendC is the generated language/API layer, not the complete backend. SIMD/SIMT execution domains, basic register operations, memory spaces, and low-level matrix operations have direct semantic counterparts. Tensor views, generic transfers, ring buffers, and dependency handling require selection or expansion; compile-time configuration can disappear. Exact emitted mappings remain unverified without compiler artifacts. See sections 5.5–5.7. [[4]](#ref-4) [[50]](#ref-50)

**Comparison.** Conventional Triton expresses blocks of values and delegates much intra-program mapping to the compiler. TileLang exposes scoped tiles and transfers with compiler-assisted layouts and pipelines. CANNBot-DSL's inspected examples expose Ascend storage paths, register computation, and pipeline stages directly. PyAsc v2 contains both the older AscendC-like `asc` API and the higher-level AscTile: logical tile expressions let its compiler manage more intermediate storage and generate supported VF code. These are overlapping abstraction dimensions, not a universal ranking of languages. See sections 5.1–5.4 and 5.8. [[25]](#ref-25) [[26]](#ref-26) [[57]](#ref-57)

**Transfer priorities.** Start with compiler-contract regression tests and consistent source/IR/AscendC bundles. Next, consider a NumPy-based functional interpreter and a bounded tile-value layer over existing CANNBot primitives. Subsequently evaluate inferred VF construction, lifetime-based temporary-memory reuse, and automatic completion of supported transfer paths. Preserve explicit expert control. VF fusion and allocation reuse are opt-in in the inspected AscTile compiler; neither establishes a general performance advantage. [[57]](#ref-57)

**Meaning and limits.** The interpreter executes supported kernel logic on CPU for algorithm debugging; it does not validate generated code, races, or device timing. Transfer-path completion inserts intermediate copies, such as UB → L1 → L0A, for supported requests; it does not guarantee optimal placement or performance. Porting passes requires compatible IR, effects, alias, and asynchronous-lifetime contracts. Both systems already automate some synchronization. See section 5.8. [[59]](#ref-59) [[64]](#ref-64)

**Research focus.** Prioritize semantic preservation, concurrency contracts, reproducible numerical validation, and end-to-end fusion benefits. Findings are source-based at the commits recorded below; no compiler or NPU benchmark results were independently reproduced.

---

**Evidence date:** October 30, 2026  
**Inspected commit:** `2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c` [[1]](#ref-1)  
**Method:** local source, documentation, and history inspection; GitCode API queries; official Triton and TileLang documentation. No NPU experiments or independent benchmark reproduction were performed.

This is a dated research baseline. Open issue and merge-request states describe the evidence snapshot and may change. Repository source links are pinned to the inspected commits; external documentation links may change.

## 1. Assessment

CANNBot-DSL is a hardware-explicit Python programming system being exercised on demanding Ascend model workloads. Its main research challenge is preserving correctness and predictable performance as fusion, concurrency, and distributed execution become more complex.

The repository demonstrates substantial implementation depth and active development, with compatibility and integration stabilization underway. This is an engineering assessment, not an official maturity designation.

The most promising research directions are compiler semantic preservation, enforceable concurrency contracts, reproducible numerical validation, and the relationship between kernel optimization and model latency. Several serious reports concern DSL 0.7/0.7.1; they do not establish that the same failures persist in 0.8.

## 2. Purpose, Architecture, and Workloads

The project lets developers express NPU operators in Python while controlling tiling, storage placement, transfers, and execution across matrix and vector units. Its documentation identifies predictable performance, accessible optimization decisions, and suitability for agent-assisted programming as design goals. Programming-model overview [[2]](#ref-2)

This checkout supplies operator implementations, tests, documentation, and Native packaging. The compiler/runtime is supplied through the separately installed `cannbot-dsl` wheel, imported as `cannbotdsl`. CANNBot, the broader infrastructure agent, is a separate project. Compiler-source availability was not established by this investigation. Repository README [[3]](#ref-3)

The documented compilation path is:

```text
Python host/kernel descriptions
    → CANNIR / MLIR
    → compiler passes
    → generated AscendC
    → device binary
```

MLIR and AscendC dumps provide observation points for studying lowering and generated schedules. Some diagnostic options are version-sensitive. Compiler observation guide [[4]](#ref-4)

The operator collection covers attention, sparse and quantized MLA, matrix multiplication, normalization, recurrent KDA, and communication-related workloads. Three directions stand out:

- **Model-specific fusion:** projections, normalization, RoPE, quantization, cache updates, and attention.
- **Recurrent execution:** persistent kernels, KDA state, and speculative-state handling.
- **Distributed training:** MoonEP planning, expert-weight prefetch, and gradient reduction.

The checked-in MoonEP implementation combines DSL-written components with existing `ElasticBuffer` dispatch/combine communication. That integration boundary is relevant to both correctness and performance. MoonEP design [[5]](#ref-5)

## 3. Current Status

| Area | Evidence and assessment |
| --- | --- |
| Development activity | The inspected history contains 61 commits since October 1, 2026, covering operators, optimization, CI, and documentation. Activity alone does not establish maturity. |
| DSL 0.8 transition | Several operator families have migrated, while attention, MQSMLA, and matrix-multiplication migration work remains open. Attention [[6]](#ref-6), MQSMLA [[7]](#ref-7), matmul [[8]](#ref-8) |
| Testing | Accuracy tests and hardware CI exist. The CI runner performs precision checks before profiling eligible cases and isolates runs using locks and separate caches. A current overall pass rate was not verified. CI runner [[9]](#ref-9), workflow [[10]](#ref-10) |
| Deployment | Native packaging exports selected configurations. Binary coverage, JIT fallback, and wheel compatibility require explicit management. Packaging guide [[11]](#ref-11) |
| Documentation | The baseline README describes 0.7. An open update describes a published 0.8 package and CANN 9.2.1 guidance. Package installation was not independently verified. Homepage update [[12]](#ref-12) |
| Hardware | The documented sample support targets Ascend 950PR/950DT, ARCH 3510. An alternative packaging target does not itself establish validated sample portability. Hardware model [[13]](#ref-13), 910B/C discussion [[14]](#ref-14) |

The GitCode releases and tags queries returned empty lists. This does not establish that wheel releases are unavailable through other channels.

An important version qualification: issue #79 reports SIMT integer truncation on 0.7.0. A response states that an internal fix was prepared for 0.8.0. This is a useful versioned regression candidate, not an independently established current 0.8 defect. Issue #79 [[15]](#ref-15)

## 4. Near-Term Direction

The investigation found concrete work in progress, but no verified dated roadmap. The following are open proposals at the evidence date, not delivery commitments.

| Direction | Work in progress |
| --- | --- |
| Consolidate DSL 0.8 support | Operator migration and updated installation documentation. Matmul migration [[8]](#ref-8), homepage [[12]](#ref-12) |
| Expand model-level fusion | A DeepSeek V4.1 attention megakernel proposes combining projection, normalization, RoPE, cache updates, and sparse attention in one launch. PR #258 [[16]](#ref-16) |
| Extend quantization | MXFP8 stochastic rounding and full-quantized MLA prolog with small-token optimization. PR #286 [[17]](#ref-17), PR #281 [[18]](#ref-18) |
| Reduce distributed overhead | MoonEP planning, host argument preparation, gradient-transfer memory use, and communication performance. PR #274 [[19]](#ref-19), PR #261 [[20]](#ref-20) |
| Support variable resources | Remove fixed core-count assumptions from attention epilogue scheduling. PR #277 [[21]](#ref-21) |
| Harden contracts | MLA scaling/boundary validation and duplicate Native catalog rejection. PR #291 [[22]](#ref-22), PR #290 [[23]](#ref-23) |

**Inference:** the immediate direction is 0.8 consolidation alongside larger fused regions and closer model integration. No delivery dates can be inferred reliably.

A documentation proposal reports whole-model latency changing from 13.05 to 12.23 ms, about 6.28%, while explicitly stating that it did not rerun the performance experiments. This is an attributed result, not independently reproduced evidence. PR #270 [[24]](#ref-24)

## 5. Programming Model and Abstraction Layers

### 5.1 Approach and Compiler Boundary

The central distinction is which implementation decisions remain in the source program:

- **Triton:** express computation over blocks of tensor values; the compiler largely chooses their distribution across threads, registers, and shared memory.
- **TileLang:** express tiles in selected memory scopes, transfers, and computation; the compiler fills in thread mappings and lowers tile operations.
- **CANNBot-DSL:** express Ascend storage levels, computation engines, and producer–consumer structure directly; the compiler lowers those descriptions and handles supported dependencies.

The comparison uses conventional `triton.language`, upstream TileLang, and this CANNBot-DSL checkout. It compares programming approaches, not identical hardware capabilities or benchmark results. Backend extensions can change the abstraction boundary. Triton's Gluon is discussed separately because it exposes more control than conventional Triton. Triton model [[25]](#ref-25), TileLang overview [[26]](#ref-26), Gluon overview [[27]](#ref-27)

### 5.2 CANNBot-DSL Layers Explained

The layers below are an analytical decomposition, not an official numbered hierarchy. Synchronization and specialization cut across them.

**1. Operator interface and host orchestration.** Ordinary Python/PyTorch code validates inputs and allocates outputs. A DSL `@host` function launches `@kernel` code. Host tiling selects implementation, tile sizes, buffer depths, and participating cores. The matmul sample makes this division explicit through a tiling class, kernel implementation, and public operator interface. Introduction [[28]](#ref-28), matmul source [[29]](#ref-29)

**2. Tensor views, layouts, and tiles.** A DSL `Tensor` describes storage location plus layout. Logical/physical shape, stride, format, and tile views describe access to that storage. `tile_slice` creates an alias rather than transferring data. Logical indexing can hide physical fractal-address calculations while retaining alignment constraints. This storage-layout abstraction differs from distributing GPU tensor elements across lanes and warps. Layouts [[30]](#ref-30)

**3. Memory placement and movement.** Buffers explicitly occupy GM, UB, L1, L0A, L0B, L0C, or other Ascend spaces. `mem_copy` connects them, with selected transfers supporting physical-format conversion. For a matrix operation, the source can describe the following hierarchy directly:

```text
GM → L1 → L0A/L0B → matrix computation → L0C → GM or UB
```

The programmer retains the hardware storage hierarchy while the compiler handles lower-level realization. Memory guide [[31]](#ref-31), compute guide [[32]](#ref-32)

**4. Buffer lifetimes and pipeline structure.** Ring-buffer abstractions organize storage reuse between stages. The checkout contains both Channel examples and newer RingBuffer code; exact API claims must be versioned. Dependency tracking does not imply automatic schedule construction. In the Add sample, the programmer writes load(k), compute(k−1), and store(k−2), including startup and draining. Add implementation [[33]](#ref-33)

**5. Computation engines and execution modes.** Matrix `matmul` operates on operands in designated memories. Register SIMD exposes vector loads, arithmetic, masks, reductions, and stores inside `vf(mode="simd")`. SIMT exposes thread-indexed computation through `vf(mode="simt")`. Scalar logic supplies indexing and control. The language therefore spans collective matrix operations and much more explicit vector-register programming. Compute guide [[32]](#ref-32)

**Synchronization boundary.** In the documented Channel model, the compiler generates supported producer–consumer dependencies. Other vector-memory overlaps, cross-core protocols, barriers, and cache-maintenance cases remain explicit. Automatic synchronization has a defined scope; it is not a blanket guarantee. Synchronization guide [[34]](#ref-34)

**Specialization boundary.** Python construction and compile-time choices generate specialized device programs. `TensorSpec` and `Dim` can describe reusable dynamic dimensions. Choosing what is static remains an important performance and compilation-cost decision. JIT and cache guide [[35]](#ref-35)

### 5.3 Comparison Table by Abstraction Layer

| Layer | CANNBot-DSL | Triton: conventional triton.language | TileLang |
| --- | --- | --- | --- |
| Host orchestration | Explicit host entry and host tiling select kernels and resource configurations. | Python wrapper launches a JIT kernel over a program grid. | Python/JIT construction and T.Kernel describe launch structure. |
| Computation unit | Memory-resident tiles, matrix operations, SIMD registers, or SIMT threads. | A program instance operates on blocks of tensor values. | Tiles in buffers, collective operations, and parallel loops. |
| Work partitioning | Programmer maps tiles to cores/blocks and coordinates AIC/AIV work. | Programmer assigns work to program IDs; compiler distributes work within each instance. | Programmer supplies grid, tile sizes, and thread count; compiler maps tile work to threads. |
| Tensor/layout abstraction | Storage views with logical/physical layouts and explicit memory locations. | Value/pointer tensors; internal thread/register layouts largely compiler-selected. | Buffer layouts and thread-distributed fragments; inference or explicit annotations. |
| Memory hierarchy | Explicit Ascend spaces, including UB, L1, and L0A/B/C. | On-chip allocation largely compiler-managed in ordinary tensor code. | Explicit scopes, including shared memory and register fragments. |
| Data movement | Explicit mem_copy between selected levels, with format-conversion paths. | Logical loads/stores; compiler manages much of their on-chip staging. | Explicit T.copy between scopes; compiler lowers transfer mechanisms. |
| Matrix computation | Prepare required operand memories, then call matmul. | tl.dot expresses a block product. | T.gemm expresses a tile product over declared buffers/fragments. |
| Elementwise computation | Vector-register operations and masks, or SIMT indexing. | Tensor expressions, broadcasting, and reductions. | T.Parallel loops and buffer/fragment operations. |
| Software pipeline | Programmer structures stages, reuse, and overlap. | Compiler constructs pipelines under controls such as num_stages. | T.Pipelined requests inferred pipelines; stage/order annotations provide explicit control. |
| Synchronization | Generated for supported dependency abstractions; other dependencies remain explicit. | Compiler handles ordinary intra-program communication; inter-program coordination remains algorithm-dependent. | Compiler inserts supported synchronization; manual async protocols require corresponding controls. |

Sources for the Triton columns: programming model [[25]](#ref-25), language reference [[36]](#ref-36), GEMM tutorial [[37]](#ref-37), pipeline controls [[38]](#ref-38).

Sources for TileLang: language basics [[39]](#ref-39), tile declarations and inference [[26]](#ref-26), instruction reference [[40]](#ref-40), software pipelines [[41]](#ref-41).

### 5.4 Interpretation and Research Implications

CANNBot-DSL is closest to TileLang in making memory placement and movement part of source code. In the inspected examples it also exposes Ascend storage paths and register-vector execution directly. TileLang's common model delegates more thread-distribution and pipeline construction work to inference.

Conventional Triton starts from a more abstract blocked computation. Gluon narrows the difference by exposing layouts, shared memory, warp specialization, and target-specific operations. A complete study should distinguish Triton language from Gluon rather than treating all Triton programming as equally abstract. Gluon [[27]](#ref-27), Gluon tensor layouts [[42]](#ref-42)

For experiments, record which decisions the programmer supplies: tile shape, memory placement, thread distribution, buffer depth, pipeline order, and synchronization. Compare correctness, development effort, generated code, and measured performance. A single high-level/low-level label hides these independent dimensions.

### 5.5 AscendC as a Code-Generation Target: What the Backend Actually Is

**Terminology.** For the documented AI Core compilation path, CANNBot-DSL generates AscendC source. AscendC is the target language and programming API layer; Bisheng/CCEC is the downstream compiler toolchain that produces device code. CANNBot-DSL also has its own CANNIR lowering and translation stages. Calling AscendC "the backend" is reasonable shorthand for the code-generation target, but does not identify the complete compiler backend or runtime.

```text
CANNBot-DSL Python
    → CANNIR / MLIR
    → DSL lowering and translation
    → generated AscendC source
    → Bisheng / CCEC compilation
    → NPU machine code
```

This chain is documented in the repository; no generated artifact or compiler invocation was independently reproduced in this investigation. It describes AI Core code and is not a complete account of AICPU or host packaging. Local compilation guide [[4]](#ref-4), Bisheng overview [[43]](#ref-43)

**AscendC itself spans abstraction levels.** The asc-devkit development documentation distinguishes pointer-oriented SIMD C, thread-oriented SIMT, Basic C++ Tensor APIs with TPipe/TQue, and Tensor APIs carrying Layout semantics. Higher-level algorithm components and templates sit above these interfaces. Consequently, AscendC should not be represented as a single instruction-level layer. The development documentation does not establish availability in every released CANN toolchain. API-level guide [[44]](#ref-44)

#### Comparison with AscendC Interfaces

| Dimension | CANNBot-DSL in this checkout | AscendC lower-level interfaces | AscendC framework / higher-level components |
| --- | --- | --- | --- |
| Program construction | Python construction, compile-time choices, DSL tracing, specialization. | C/C++ source, constants, templates, and explicit kernel entry points. | C++ objects and configurable library components. |
| Tensor description | Memory location, logical/physical layouts, typed tile views. | Raw pointers or GlobalTensor/LocalTensor/RegTensor; extended Tensor API includes Shape/Stride and Layout. | Components accept tensors and algorithm-specific configuration. |
| Memory movement | mem_copy between explicitly selected memories; some transfer parameters derived from tensor metadata. | Explicit transfers and matrix-load/output operations; interface-dependent parameter inference. | Matmul can encapsulate common internal tiling and transfer logic. |
| Buffer reuse | Buffer/RingBuffer; Channel appears in older examples. | Explicit storage and lifetime management. | TPipe/TQue manage resources and queue operations. |
| Synchronization | Supported dependencies lowered by the DSL; other protocols remain explicit. | Explicit primitives, with partial compiler auto-sync support where applicable. | Queue APIs encapsulate supported producer/consumer synchronization. |
| Vector computation | vf SIMD scopes, vector-register operations, masks; SIMT alternative. | RegTensor/MaskReg or C register operations in SIMD VF; SIMT interfaces. | Memory-oriented vector APIs can hide register-level implementation. |
| Matrix computation | matmul on prepared L0A/L0B/L0C operands; surrounding transfers remain visible. | Mmad / corresponding tensor-level matrix primitives. | Matmul objects encapsulate common tiling, transfers, and computation. |
| Pipeline schedule | Explicit load/compute/store staging in inspected examples. | Programmer controls low-level staging and overlap. | Queues support stage coordination; algorithm components encapsulate selected schedules. |

Sources: C++ Tensor model [[45]](#ref-45), Reg vector APIs [[46]](#ref-46), TPipe/TQue model [[47]](#ref-47), compiler auto-sync option [[48]](#ref-48), high-level Matmul [[49]](#ref-49). API correspondence is conceptual, not proof of the exact emitted call sequence.

#### What the DSL Adds, and What It Retains

The strongest abstraction gains are program construction and specialization in Python, tensor-view-based parameter derivation, reusable storage containers, and compiler processing of supported dependencies. Python's shorter syntax is only one part of that contribution: CANNIR also supplies an intermediate representation for transformations and analysis.

The retained responsibilities are substantial: algorithm decomposition, memory hierarchy, tile sizes, engine selection, pipeline staging, and explicit synchronization outside supported automation. A vector expression built from vload/vadd/vstore remains close to AscendC register programming. Similarly, the DSL's low-level matmul primitive does not provide the same algorithm abstraction as an AscendC high-level Matmul object. Depending on the task, that component may hide more implementation detail than the inspected DSL sample. DSL-to-AscendC concept map [[50]](#ref-50), DSL matrix model [[32]](#ref-32)

This also corrects a tempting oversimplification: AscendC already provides abstraction through queues and partial compiler synchronization. The comparison is not "manual AscendC versus automatic DSL"; it is the extent and contract of automation at each interface and compiler version.

**Research implication.** A useful comparison has three implementations: hand-written AscendC at matching granularity, an AscendC high-level component where applicable, and CANNBot-DSL. Compare source-level decisions, generated AscendC, resulting instructions, compilation time, and warm runtime under identical conditions. Sharing a downstream compiler does not guarantee identical machine code, performance, or feature coverage: the DSL's representation and translation can expose or constrain what reaches it.

### 5.6 Which DSL Abstractions Lower Directly to AscendC?

**Evidence boundary.** The public checkout contains samples and API documentation, not the translator implementation. No `.asc`, `.cce`, or `.mlir` artifacts were found by the tracked/unignored file search, and `cannbotdsl` is not installed in the Python environment checked here. Consequently, the mappings below distinguish documented semantic correspondence from verified emitted symbols. None is presented as an independently confirmed literal one-call-to-one-call translation.

"Direct" here means that the programmer has already selected an operation or execution construct with an AscendC counterpart. It does not mean one machine instruction: dtype, format, alignment, target, and optimizations can change the emitted sequence.

#### A. Direct Semantic Counterparts

| DSL construct | AscendC counterpart | Assessment |
| --- | --- | --- |
| @kernel | AI Core kernel function / kernel qualifiers | Direct execution-role correspondence; a Python class is not thereby preserved as a device C++ class. |
| vf(mode="simd") | SIMD VF function/domain | Direct execution-domain correspondence; extracting a scope into a function can require generated parameters and calls. |
| vf(mode="simt", thread=N) | SIMT VF and launch-bound/thread configuration | Direct execution-domain correspondence; wrapper generation is additional. |
| vload, vstore | Register loads/stores | Direct primitive family; exact overload and sequence depend on access mode. |
| vadd and other basic v* arithmetic | Register arithmetic, such as Reg::Add or a C API equivalent | Direct primitive family, not proof of the emitted namespace or symbol. |
| Mask and mask operations | Mask register type and operations | Direct semantic counterpart. |
| matmul on L0 operands | Mmad / corresponding low-level matrix primitive | Direct matrix-compute abstraction, not the high-level AscendC Matmul algorithm object. |
| get_block_idx / get_subblock_id | Device block/subblock identification | Direct indexing role; exact generated access remains unverified. |
| Explicit pipe synchronization and cache operations | Corresponding hardware synchronization/cache primitives | Direct operation families; exact mapping is operation-specific. |
| MemLoc.UB / L1 / L0A / L0B / L0C | Ascend storage spaces | Direct memory-space correspondence; allocation representation is a separate question. |

The repository's concept map [[50]](#ref-50) documents these correspondences. AscendC exposes both C and C++ register interfaces for the same hardware operations, so a semantic match does not identify which interface the translator emits. AscendC Reg API [[46]](#ref-46)

#### B. Abstractions Requiring Selection, Expansion, or Dependency Analysis

| DSL construct | What has to be realized below it | What must not be assumed |
| --- | --- | --- |
| mem_copy | Select transfer family from source/destination memories and options; derive extents, strides, conversion and output parameters. | There is no demonstrated universal mem_copy → DataCopy call. L1→L0 loading and L0C output have different counterparts. |
| Tensor / Layout / tile_slice / views | Represent storage, calculate coordinates and offsets, propagate shape/stride/format; retain dynamic calculations where necessary. | A DSL Tensor is not proven to become an AscendC LocalTensor or asc::te::Tensor object. Similar APIs do not establish reuse. |
| Buffer | Reserve storage, determine capacity/alignment, and expose typed views. | A Buffer → TBuf correspondence is conceptual, not a verified emitted allocation API. |
| RingBuffer / next_write / next_read | Provide depth slots, independent read/write cursor progression, wraparound, and selected views. | RingBuffer is not proven to lower to TQue. Cursor operations do not themselves express a blocking runtime queue. |
| Channel / produce / consume | Realize slots and supported producer–consumer dependencies, including generated synchronization. | The appendix's TPipe/TQue analogy does not prove that the translator instantiates those C++ classes. |
| CrossCore Channel | Realize the declared storage handoff and synchronization across execution units. | Not a single generic queue instruction or a blanket cross-core correctness guarantee. |
| @jit helper | Trace/compose helper logic and lower it with the surrounding program. | No universal @jit → C++ function mapping; VF helpers and ordinary helpers can differ. |
| Runtime if / range | Lower control flow and typed expressions; optimize or unroll when applicable. | A source loop need not survive unchanged, and a hardware loop is not guaranteed. |

Sources: layout semantics [[30]](#ref-30), RingBuffer API [[51]](#ref-51), synchronization [[34]](#ref-34), control flow [[52]](#ref-52). Some describe behavior rather than translator internals; expansion details above are the required semantics, not a reconstructed implementation.

#### C. Compile-Time Constructs That Need No Device-Side Counterpart

- Python configuration objects, constructor calculations, and host tiling select constants and implementations. Their results may survive; the original Python objects need not.
- const_expr and target_version choose a path at compilation time. A discarded path is absent from the resulting program.
- range_constexpr and fixed-sequence iteration expand computation at compilation time; no corresponding runtime iterator is required.
- Static shape/stride expressions can fold to constants. Dynamic expressions can remain in generated parameters or arithmetic; layout metadata is not always entirely erased.

Staging model [[53]](#ref-53), type system [[54]](#ref-54), control-flow rules [[52]](#ref-52)

#### D. Host, Compilation, and Packaging Abstractions

TensorSpec, Dim, compile, JIT cache selection, @export, and Native wheel construction describe compilation contracts or deployment. Their effects can influence specialization, generated arguments, checks, launch wrappers, or packaged artifacts. They are not AI Core arithmetic operations. Similarly, @host orchestration and framework output allocation must be analyzed separately from device-kernel translation. AscendC has its own host facilities, but that does not prove the DSL lowers these constructs one-to-one to them. Compilation contracts [[35]](#ref-35), Native packaging [[11]](#ref-11)

**Consequence for abstraction analysis:** the closest-to-AscendC part is primitive computation and execution domains. The DSL contributes more than an API rename around storage views, dependency handling, staging, specialization, and integration. How much of that contribution is custom compiler work versus reuse of AscendC libraries remains an implementation question.

**Verification needed for a literal lowering map:** pin the DSL wheel and CANN version; compile small isolated kernels with MLIR and AscendC dumps enabled; correlate each source construct with IR before/after lowering and emitted declarations/calls. A second operation using the same buffer is needed to expose dependency insertion. Test different transfer directions separately. Label observed mappings by version; one sample cannot establish the universal lowering rule. Dump controls [[4]](#ref-4)

### 5.7 Obtaining Paired DSL, MLIR, and AscendC Evidence

Use a compatible Linux environment with the DSL wheel and its CANN dependencies. Start with compilation to AscendC, without invoking the returned program or allocating NPU tensors. This does not require kernel execution; whether translation avoids all device queries depends on the wheel, imports, and example. An explicit target helps cross-compilation but does not remove package dependencies. No such run has been performed on this Mac.

**Pin the environment.** Retain the exact wheel and its SHA-256, not just the public version string. Record Python version, architecture, installed distributions, actual CANN version, target, and compiler flags. Record `bisheng --version` when available; if this is only a translation environment without Bisheng, state that explicitly. Save the probe source and repository commit with the outputs. Use one output directory per probe and invocation; preserve stderr and the exit status, including failures.

Example collection commands, run from the CANNBot-DSL checkout root in the prepared Linux environment:

```bash
mkdir -p project-analysis/lowering-evidence/copy-buffer-001
python --version > project-analysis/lowering-evidence/copy-buffer-001/python-version.txt
python -m pip freeze --all > project-analysis/lowering-evidence/copy-buffer-001/python-packages.txt
git rev-parse HEAD > project-analysis/lowering-evidence/copy-buffer-001/repository-commit.txt
python -c 'import cannbotdsl as cbd; print(cbd.__version__); print(cbd.__file__)' > project-analysis/lowering-evidence/copy-buffer-001/dsl-identity.txt
```

Separately record the actual wheel checksum and CANN/toolchain identity from that installation. `pip freeze` alone does not establish binary identity or the installed CANN SDK version.

**First probe: a static GM→UB→GM copy.** Save the following as `probe_copy.py` inside the run directory. It exercises a storage allocation and two transfer directions, without vector computation, dynamic dimensions, or resource queries in user code. The Buffer constructor follows the checked-in 0.8-style API. This is a proposed probe, not a verified compatible script for every DSL build.

```python
import cannbotdsl as cbd


@cbd.kernel
def copy_kernel(x, y):
    scratch = cbd.Buffer(
        cbd.MemLoc.UB,
        cbd.dtypes.float32,
        cbd.make_layout((64,)),
    )
    cbd.mem_copy(scratch, x)
    cbd.mem_copy(y, scratch)


@cbd.host
def run_copy(x, y):
    copy_kernel[1](x, y)


spec = cbd.TensorSpec((64,), cbd.dtypes.float32)
cbd.compile(run_copy, spec, spec)
# Deliberately do not invoke the compiled program.
```

**Enable dumps before importing the package.** Source the matching CANN environment first if the installation requires it. In a fresh shell, use:

```bash
export CANNBOTDSL_NPU_ARCH=dav-3510
export CANNBOTDSL_DUMP_MLIR=1
export CANNBOTDSL_DUMP_ASCENDC=1
export CANNBOTDSL_PRINT_MLIR=1
export CANNBOTDSL_PIPE_STAGE=translate
export CANNBOTDSL_LOG_LEVEL=10
export CANNBOTDSL_LOG_TO_CONSOLE=1
export CANNBOTDSL_DUMP_DIR="$PWD/project-analysis/lowering-evidence/copy-buffer-001/dump"

python project-analysis/lowering-evidence/copy-buffer-001/probe_copy.py > project-analysis/lowering-evidence/copy-buffer-001/compile.log 2>&1
```

These diagnostic switches are described by the repository compiler guide [[4]](#ref-4), including target override, translation-stage stopping, and cache bypass. Verify them against the pinned wheel. Leave the production pass pipeline and synchronization settings unchanged for the baseline; do not enable AscendC reuse. Do not assume that one `.mlir` file preserves every pass snapshot: inspect the actual output and retain the printed IR log as well. Generated filenames can vary.

**Acceptance criteria:** the saved source, MLIR, and `.asc` belong to the same run; the log identifies successful translation or its exact failure; IR and generated code contain a consistent kernel symbol/dataflow. A translation-stage stop or an existing file alone is not proof of full compilation or runtime correctness. If only one IR stage is captured, describe it as such rather than claiming before/after pass evidence.

**Expand one construct at a time:** add SIMD load/add/store; change Buffer to RingBuffer; add a second pipeline iteration that reuses storage; test layout slicing; test L1→L0 and matrix computation separately. Include a compile-time branch as an erasure probe. Compare declarations, pointer arithmetic, calls, and synchronization with surrounding code. Inspect relevant generated headers and helpers too: an `.asc` call to a helper does not reveal the helper's entire implementation.

For each finding, record: DSL source range, observed IR operation, emitted AscendC symbol or sequence, supporting artifact path, and environment identity. Use "observed in this build" until multiple cases establish a broader rule. Runtime tests on supported NPUs are a subsequent, separate validation step.

### 5.8 Comparison with Local PyAsc v2 / AscTile and Transfer Opportunities

#### Scope and Evidence

This comparison inspects a local PyAsc checkout on branch `v2`, commit `936279f2b71ecb8cc781f438e8e51222e2e7f08f` (October 29, 2026), against the CANNBot-DSL baseline above. It is a source-based comparison, not an upstream refresh or performance experiment. The PyAsc working tree contains untracked analysis/design documents; the conclusions below use tracked implementation, tests, and AscTile documentation. No PyAsc files were changed.

There are two relevant interfaces in that branch:

- **`asc`:** the existing Python interface closely corresponding to AscendC APIs.
- **`asc.experimental.asctile`:** the newer tile-oriented layer in `experimental/asctile`, with its own IR, transformation pipeline, and interpreter.

The main comparator for the new DSL is **AscTile**, not the root README's description of the older `asc` interface. The README's 90%-performance statement is a project claim/goal; this analysis does not establish a general performance ratio. AscTile overview [[55]](#ref-55)

#### Central Difference: Storage-Oriented Versus Value-Oriented Programming

CANNBot-DSL exposes storage views, explicit buffers/ring buffers, data transfers, and SIMD/SIMT computation. In inspected examples the author determines buffer organization and pipeline stages. The compiler still performs memory planning and supported dependency handling; this is not fully manual allocation or synchronization.

AscTile usually exposes **logical tile values**. Expressions produce results; the compiler can select physical storage and reuse allocations subject to effects and lifetime constraints. Its IR marks operations such as Softmax, RMSNorm, and Matmul as pure, while stores, atomics, and accumulator updates have side effects. Thus value semantics is not a claim that every AscTile operation is immutable or effect-free. Operation definitions [[56]](#ref-56)

For a vector expression, the conceptual difference is:

```text
CANNBot-DSL explicit SIMD path:
    select UB storage → copy → load registers → arithmetic/masks → store registers → copy

AscTile common path:
    copy_in → tile expression → copy_out
    compiler chooses intermediate storage and, when enabled/supported, VF register realization
```

Both retain explicit workload partitioning and tile-size choices. AscTile is not an automatic whole-model compiler or a guarantee of optimal tiling.

#### Abstraction-Layer Comparison

| Layer | CANNBot-DSL baseline | PyAsc v2 / AscTile implementation | Significance |
| --- | --- | --- | --- |
| Python entry | Public Python wrapper, @host orchestration, @kernel, internal @jit helpers. | @asctile.jit kernel using the existing PyAsc specialization/runtime infrastructure. | Similar Python integration, different staging and entry contracts. |
| Main local object | Tensor as storage plus logical/physical views; Buffer/RingBuffer explicit. | LocalTensor as a logical tile result; supported arithmetic produces SSA values. | More storage/lifetime decisions can move into the compiler. |
| Shape/layout | Logical/physical formats, strides, and tile views are visible. | Local tile shapes are static at JIT time; global dimensions can be dynamic; shape/broadcast/reduction APIs. | Similar multidimensional reasoning, not interchangeable layout types. |
| Placement and transfers | Programmer selects memory levels and writes transfers. | Explicit locations and Auto placement; passes resolve locations, complete transfer paths, split loads, and verify legal locations. | A compiler-assisted placement/path layer exists; not a universal cost-based optimizer. |
| Elementwise operations | Explicit register SIMD operations or thread-indexed SIMT in inspected kernels. | Tile expressions and operations such as RMSNorm/Softmax, lowered by passes. | AscTile exposes more algorithmic semantics before register lowering. |
| Vector fusion | Author can express register-resident VF computation; compiler/VF optimizations also exist. | Optional pipeline groups supported vector operations and lowers to register VF code. | The transferable addition is inferring VF code from higher-level operations, not introducing VF support itself. |
| Allocation | Declared buffers, capacity, and ring depth; underlying planning is automatic. | Static or TPipe-backed allocation; optional lifetime-based reuse across UB/L1/L0 memories. | Different ownership of reuse decisions; explicit aliases complicate adaptation. |
| Synchronization | Supported automatic buffer dependencies plus explicit engine/core/cache protocols. | Architecture-specific queue/BufID insertion, cross-core passes, and optional verification. | Both automate synchronization; contracts and diagnostics are the meaningful comparison. |
| Software overlap | Explicit stage order and ring-buffer reuse in the Add example. | Unrolling, reuse groups, dependency insertion, and a gm_barrier control. | More implicit scheduling support, but not proof of arbitrary automatic pipeline synthesis. |
| Debug execution | Golden tests and compiler dumps are visible; a comparable functional interpreter was not established from this checkout. | ASCTILE_INTERPRET selects a NumPy-based functional path. | Useful for rapid algorithm debugging; cannot validate device races or timing. |
| Backend | CANNIR/MLIR → AscendC → Bisheng; backend not present here. | AscTile → Asc/AscendC/AscVF/EmitAsc transformations → AscendC source → Bisheng; implementation visible. | Common target does not imply compatible IR or reusable pass binaries. |
| Escape to lower levels | Explicit storage, register operations, synchronization primitives. | to_asc/from_asc and inline/inline_vf provide lower-level interoperability. | A high-level layer can coexist with an explicit expert path. |

The implementation's compiler schedule [[57]](#ref-57) establishes the actual pass order and options. It includes Auto-location resolution, transfer completion, location verification, matrix legalization, optional VF fusion, optional reuse, allocation dispatch, cross-core synchronization, architecture-specific intra-core synchronization, and memory-consumption accounting.

#### Implemented Features Versus Defaults and Claims

- `insert_sync=True` is the AscTile default. C310 uses BufID synchronization; the other inspected branch uses queue synchronization. This does not establish correctness for every program or support for cross-NPU communication.
- `vf_fusion=False` by default. Enabling it is rejected outside C310. Its pass sequence and FileCheck tests are present; no general speedup was measured here.
- `reuse_alloc=0` by default; values 1 and 2 select different reuse implementations. The option documentation explicitly notes potential performance regressions.
- Static allocation is selected conditionally. The dispatch pass can fall back to TPipe-backed allocation, including a documented Broadcast interaction.
- The interpreter iterates blocks/subblocks sequentially and evaluates operations through NumPy. It still uses the project's frontend/build infrastructure; it is not a cycle-accurate simulator or proof that the entire package works without SDK dependencies.
- The source `range` exposes `unroll_factor` and `gm_barrier`. Unrolling enables overlap opportunities but should not be presented as a general equivalent of TileLang's pipeline scheduler.

Evidence: compile options and pipeline [[57]](#ref-57), allocation dispatch [[58]](#ref-58), interpreter execution [[59]](#ref-59), range implementation [[60]](#ref-60), software-pipelining description [[61]](#ref-61).

#### Functional Interpreter: Running the Algorithm on CPU

A functional interpreter implements supported DSL operations using CPU operations, rather than executing the compiled kernel on the NPU. In AscTile, the JIT decorator can select `InterpretedFunction` through `ASCTILE_INTERPRET`. The interpreter substitutes an operations module, wraps arguments, and executes blocks/subblocks sequentially. The implementation uses NumPy for array computations while retaining frontend validation; the installed package can still require built frontend components and SDK dependencies.

For example, a kernel with the logical sequence below can be checked on small CPU arrays:

```text
Kernel logic                  Functional interpretation
load a tile                   Read an array region
y = (x + 1) * 2               Evaluate NumPy array operations
store y                       Write the destination array region
```

The benefit is checking the same kernel's supported indexing, formulas, shapes, and boundary handling without device execution. Intermediate values can be inspected using ordinary debugging techniques. This is complementary to an independently written mathematical reference: shared implementation assumptions can otherwise hide mistakes.

| Question | What the interpreter establishes |
| --- | --- |
| Are indexing and the formula correct? | Functional evidence within the implemented operation semantics. |
| Is the final partial tile handled correctly? | Functional evidence if the relevant copy/boundary semantics are modeled. |
| Did the compiler generate correct AscendC? | Not established: the generated-code execution path is bypassed. |
| Are barriers and parallel accesses correct? | Not established: sequential CPU execution does not model NPU races. |
| Will rounding exactly match the NPU? | Not guaranteed; requires explicit hardware-accurate numeric modeling. |
| Will the kernel be fast? | Not established; this is not a timing simulator. |

For CANNBot-DSL, an initial interpreter should cover a clearly defined logical subset. Modeling raw registers, masks, physical layouts, and mutable aliases is substantially harder than evaluating pure tile expressions. Interpreter dispatch [[62]](#ref-62), execution [[59]](#ref-59), operation implementations [[63]](#ref-63)

#### Transfer-Path Completion: Inserting Intermediate Copies

Ascend memory spaces do not support arbitrary direct transfers. A program may express that data is needed in a destination space while the hardware requires an intermediate step. The inspected `FulfillDataTransfer` pass can expand supported local-memory copies through an intermediate L1 or UB tile.

One concrete route represented by that implementation is:

```text
Requested transfer:       UB ──────────────────→ L0A

Completed transfer path:  UB ──copy──→ L1 ──copy──→ L0A
                                      ↑
                              intermediate tile
```

The pass checks a table of permitted edges, searches for a supported intermediate location, and rewrites a copy into two copies with an intermediate result. This is a bounded transformation over supported paths, not an arbitrary routing algorithm or a universal performance optimizer.

Two separate decisions must be distinguished:

- **Placement:** choose where a tile should reside. AscTile's Auto-location resolution addresses this.
- **Path completion:** given source and destination locations, insert required intermediate transfers. `FulfillDataTransfer` addresses this.

In an explicit CANNBot kernel, the author can describe the intermediate buffers and transfers. A higher-level layer could accept the supported transfer intent and generate that boilerplate. It should preserve explicit placement when requested and expose the resulting storage/copy costs: a legal path can consume more memory and increase latency. Format, alignment, synchronization, and target constraints still apply. Transfer completion implementation [[64]](#ref-64), location resolution [[65]](#ref-65)

#### Technologies Worth Transferring from PyAsc to CANNBot-DSL

The proposals below are adaptation opportunities, not claims that the corresponding capability is absent from the undisclosed CANNBot compiler. Direct reuse of source or pass binaries requires a separate dependency, license, and compatibility review; no code was imported in this investigation.

| Priority | Candidate | Concrete evidence in PyAsc | Adaptation to CANNBot-DSL | Main constraint |
| --- | --- | --- | --- | --- |
| P0 | Compiler-contract and negative regression tests | Location verifier, expected-error MLIR tests, VF transformation FileCheck tests. | Bring the test cases and verification method to CANNIR: dtype/location/shape contracts, integer widths, aliases, and dependency edges. | Translate tests to the actual CANNIR operations; do not assume its verifier implementation is missing. |
| P1 | Functional interpreter / reference execution | JIT dispatch to InterpretedFunction and NumPy operations. | Add an execution mode for a supported logical subset, initially copies, elementwise expressions, reductions, and masks. Compare with independently specified references. | Raw register semantics, rounding, layouts, and mutable aliases need explicit modeling. It cannot detect NPU synchronization failures. |
| P1 | Optional tile-value layer | Pure high-level IR operations and low-level escape APIs. | Add a composable layer over existing explicit kernels: logical tiles, arithmetic/broadcast/reduction, compiler-managed temporaries. Keep expert storage and synchronization control. | Requires an effect/alias contract at the boundary with mutable Tensor views and RingBuffer slots. |
| P2 | Automatic VF-region construction | FindVFGroup, LowerToReg, DispatchVFFusion, MaterializeLoadStore; regression tests. | Lower eligible high-level expression regions to existing register/VF operations; keep explicit VF code as an escape path. | Needs visible operation semantics, register-pressure limits, reductions, and numerical constraints; much less benefit for already hand-fused VF code. |
| P2 | Lifetime-based memory reuse | ReuseTensorAllocation handles storage positions, views, dominance, and persistent loop values. | Reuse compiler-owned temporaries first; expose a memory report and opt-in policy. | Must preserve async lifetimes, aliases, explicit physical placement, and ring depth. Source-program order alone is not sufficient. |
| P2 | Legal transfer-path completion | ResolveAutoLocation, FulfillDataTransfer, VerifyTensorLocation. | Offer Auto placement/path selection only where the user opts in; preserve explicit placements. | A legal extra copy can increase latency or memory use. Target-specific paths and layout constraints require a cost model. |
| P1 | Structured lowering artifacts | Runtime saves codegen.mlir, ascir.mlir, and ascendc.cpp. | Produce a per-compilation bundle with source, IR stages, AscendC, options, and build identities. | CANNBot already has dumps; the addition is a stable paired artifact contract, not a new dumping capability. |

Evidence for contract testing: location verifier [[66]](#ref-66), negative tests [[67]](#ref-67). The atomic IR additionally requires matching element types, illustrating how contracts can be encoded at the operation level. Atomic operation definition [[56]](#ref-56)

Evidence for fusion: VF pipeline regression [[68]](#ref-68). The expected transformation contains register loads, add/multiply operations, masks, and stores. This is repository test evidence of intended transformations, not a newly generated paired dump or a test result from this session.

Evidence for reuse: ReuseTensorAllocation [[69]](#ref-69). Its analysis follows views, uses dominance for lifetime relationships, restricts cross-position reuse, and excludes selected loop-carried values. Its assumptions must be reassessed for CANNBot's asynchronous execution model.

Evidence for placement and bridging: transfer completion [[64]](#ref-64), low-level interoperability [[70]](#ref-70). to_asc preserves shape/dtype without copying or layout conversion; an analogous bridge would need CANNBot-specific alias and side-effect semantics.

Evidence for reproducibility: runtime compilation [[71]](#ref-71) explicitly dumps pre-pass IR, post-pass IR, and generated AscendC. Memory usage is passed to runtime metadata, and launcher checks [[72]](#ref-72) compare reported consumption with device capacities before first launch. That is not a demonstrated universal compile-time capacity proof.

#### Recommended Integration Direction

Preserve CANNBot-DSL's explicit Ascend model and add an **optional higher-level tile-expression layer**. Do not begin by replacing its backend or importing all AscTile passes.

1. Adapt a small contract/regression corpus and make source-to-IR-to-AscendC artifacts reproducible.
2. Define a bounded logical tile subset with a reference interpreter and explicit dtype/shape/effect rules.
3. Lower pure regions to current CANNBot operations; keep buffer mutation, communication, and explicit synchronization as boundaries initially.
4. Add inferred VF construction and reuse only after those boundaries are modeled and tested.

The shared AscendC target makes algorithm comparison and generated-code comparison practical. It does not make AscTile's dialects, pass ordering, runtime, or synchronization analysis drop-in compatible with CANNIR. CANNBot backend access is needed for a real pass port; without it, a separate frontend/library prototype and external test harness are more feasible.

**Conclusion:** the main technology to borrow is the architecture of a semantic tile layer with compiler-owned realization and observable transformations. Renaming AscTile operators or replacing one memory-copy wrapper would capture only a small part of its value.

## 6. Top Technical Challenges

The ranking weighs technical impact and research value. Issue reports establish reported observations, not independently verified causes or present-version failures.

### 6.1 Enforceable Concurrency Contracts

Reports on 0.7/0.7.1 describe multicore communication losing data and shared AICPU scratch storage racing across invocations. The transfer report identifies a same-peer submission pattern and a block-per-peer workaround. Current MoonEP code likewise documents communication-domain and storage-reuse constraints.

The research question is which restrictions can be represented and checked automatically. Persistence of the reported failures in 0.8 remains unverified. Issue #77 [[73]](#ref-73), issue #50 [[74]](#ref-74), MoonEP constraints [[5]](#ref-5)

### 6.2 Semantic Preservation Through Lowering

Integer-width changes, compiler crashes, and unsupported atomic-copy combinations illustrate distinct failure classes: incorrect lowering, failed compilation, and missing diagnostics. They require separate tests and version attribution. A crash from a feature combination does not itself prove that the combination is supported.

Inspecting intermediate representations and generated AscendC can help distinguish frontend, optimization, translation, and execution failures. Issue #79 [[15]](#ref-15), issue #84 [[75]](#ref-75), issue #85 [[76]](#ref-76)

### 6.3 Reproducible Numerical and Interface Validation

Correctness includes shape, dtype, layout, mutation, and numerical error. At the baseline, the matmul comparator flattens arrays before checking size, discarding original shape within that helper. This is a directly inspected test limitation, not evidence that the operator returns incorrect shapes. Comparator source [[77]](#ref-77), issue #48 [[78]](#ref-78)

PR #291 reports 166 expanded tests retained locally while proposing four checked-in functional cases. This exposes a gap between reported validation and the evidence future contributors can reproduce from the repository. PR #291 [[22]](#ref-22)

### 6.4 End-to-End Benefits from Fusion

Fusion interacts with launch overhead, host preparation, communication, synchronization, and cross-layer state. MoonEP optimization work measures host costs and differences in rank launch timing alongside device time. Kernel duration alone can miss the limiting factor. Published results require matched hardware, precision, warmup, and timing conditions before comparison. PR #274 [[19]](#ref-19), model-level report [[24]](#ref-24)

### 6.5 Specialization and External Runtime Integration

Native packaging exports selected configurations; device-side composition requires additional ABI, scratch-memory, initialization, and completion guarantees. A feature request proposes device-callable exports for PyPTO and describes a limited prototype. This is a requested capability, not established general support. Issue #75 [[79]](#ref-79), packaging guide [[11]](#ref-11)

## 7. Recommended Research Starting Points

| Priority | Question | Experiment | Success criterion |
| --- | --- | --- | --- |
| 1 | Does lowering preserve declared semantics? | Reproduce integer-width and atomic-copy cases in pinned 0.7/0.8 environments; inspect MLIR and AscendC. | Valid cases preserve semantics; unsupported cases produce useful diagnostics. |
| 2 | Which concurrency patterns are safe? | Sweep blocks, peers, streams, and message sizes in transfer/planner probes. | Complete output checks pass repeatedly; restrictions and workarounds are explicit. |
| 3 | When does fusion reduce model latency? | Compare fused/unfused paths at matched precision, shape, and resource quotas. | Separate cold compilation, warm device time, host overhead, and end-to-end latency. |
| 4 | What does each abstraction buy? | Implement matched GEMM, normalization, and attention cases across supported DSL environments. | Record programmer decisions, code size, tuning effort, generated schedules, correctness, and performance. |

Start with a small versioned regression corpus for compiler semantics and concurrency. Backend fixes may need maintainer cooperation; reproductions, generated-code analysis, and stronger public tests are accessible contributions.

Compiler-only studies need a compatible Linux toolchain and pinned wheels. Runtime validation needs supported Ascend hardware; distributed experiments need multiple devices. Cross-language performance comparisons must distinguish matched-target experiments from comparisons performed on different hardware.

## 8. Updating This Report

- Record the new evidence date, local commit, wheel/build identity, CANN version, and hardware for each refresh.
- Recheck linked issues and merge requests. Separate proposed, merged, released, and independently verified behavior.
- Refresh compatibility claims against both current source and package behavior; documentation alone may lag.
- Preserve version qualifications and attribute performance figures to their original conditions.
- Update the report and append a short dated entry below. Do not imply an automatic update schedule.

### Update Log

- **2026-10-30:** Created the consolidated report from the research discussion, including the programming-model explanation and abstraction-layer comparison. No NPU results were independently reproduced.
- **2026-10-30:** Added section 5.5 distinguishing AscendC code generation from the downstream compiler, comparing AscendC interface levels with the DSL, and qualifying synchronization and high-level Matmul claims. Consulted asc-devkit development documentation; release-specific availability remains to be verified.
- **2026-10-30:** Added section 5.6 classifying direct semantic counterparts, expanded abstractions, compile-time constructs, and host/deployment facilities. Explicitly separated documented correspondence from unverified literal translator output; checked the local environment for the DSL module and generated artifacts.
- **2026-10-30:** Added section 5.7 with a proposed compile-only probe, environment pinning, dump commands, and acceptance criteria for paired lowering evidence. The probe has not been run against a DSL wheel.
- **2026-10-30:** Added section 5.8 comparing local PyAsc v2/AscTile at 936279f2 with CANNBot-DSL, including implementation-backed transfer candidates, option defaults, integration constraints, and a staged recommendation. No PyAsc changes or compiler/NPU test runs.
- **2026-10-30:** Added the opening one-page summary and expanded section 5.8 with concrete explanations of functional interpretation and intermediate transfer-path completion, including what each mechanism does not establish.

## Sources

Source states were inspected on October 30, 2026. Commit-pinned links identify the analyzed code; live issue and documentation pages can change.

| Reference | Source | Link |
| --- | --- | --- |
| <a name="ref-1"></a>[1] | `2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c` | [Open source](https://gitcode.com/cann/cannbot-dsl/commits/detail/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c) |
| <a name="ref-2"></a>[2] | Programming-model overview | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/overview.md) |
| <a name="ref-3"></a>[3] | Repository README | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/README.md) |
| <a name="ref-4"></a>[4] | Compiler observation guide | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/compiler-options.md) |
| <a name="ref-5"></a>[5] | MoonEP design | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/samples/moonep/README.md) |
| <a name="ref-6"></a>[6] | Attention | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/284) |
| <a name="ref-7"></a>[7] | MQSMLA | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/288) |
| <a name="ref-8"></a>[8] | matmul | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/265) |
| <a name="ref-9"></a>[9] | CI runner | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/scripts/ci/run_tests.sh) |
| <a name="ref-10"></a>[10] | workflow | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/.gitcode/workflows/test_action.yml) |
| <a name="ref-11"></a>[11] | Packaging guide | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/net/native_package/README.md) |
| <a name="ref-12"></a>[12] | Homepage update | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/289) |
| <a name="ref-13"></a>[13] | Hardware model | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/hardware-model.md) |
| <a name="ref-14"></a>[14] | 910B/C discussion | [Open source](https://gitcode.com/cann/cannbot-dsl/issues/47) |
| <a name="ref-15"></a>[15] | Issue #79 | [Open source](https://gitcode.com/cann/cannbot-dsl/issues/79) |
| <a name="ref-16"></a>[16] | PR #258 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/258) |
| <a name="ref-17"></a>[17] | PR #286 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/286) |
| <a name="ref-18"></a>[18] | PR #281 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/281) |
| <a name="ref-19"></a>[19] | PR #274 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/274) |
| <a name="ref-20"></a>[20] | PR #261 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/261) |
| <a name="ref-21"></a>[21] | PR #277 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/277) |
| <a name="ref-22"></a>[22] | PR #291 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/291) |
| <a name="ref-23"></a>[23] | PR #290 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/290) |
| <a name="ref-24"></a>[24] | PR #270 | [Open source](https://gitcode.com/cann/cannbot-dsl/merge_requests/270) |
| <a name="ref-25"></a>[25] | Triton model | [Open source](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) |
| <a name="ref-26"></a>[26] | TileLang overview | [Open source](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md) |
| <a name="ref-27"></a>[27] | Gluon overview | [Open source](https://triton-lang.org/main/gluon/index.html) |
| <a name="ref-28"></a>[28] | Introduction | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/introduction.md) |
| <a name="ref-29"></a>[29] | matmul source | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/samples/matmul/matmul/matmul.py) |
| <a name="ref-30"></a>[30] | Layouts | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/data-and-layout.md) |
| <a name="ref-31"></a>[31] | Memory guide | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/onchip-memory.md) |
| <a name="ref-32"></a>[32] | compute guide | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/compute.md) |
| <a name="ref-33"></a>[33] | Add implementation | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/samples/add/add.py) |
| <a name="ref-34"></a>[34] | Synchronization guide | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/synchronization.md) |
| <a name="ref-35"></a>[35] | JIT and cache guide | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/jit-arguments.md) |
| <a name="ref-36"></a>[36] | language reference | [Open source](https://triton-lang.org/main/python-api/triton.language.html) |
| <a name="ref-37"></a>[37] | GEMM tutorial | [Open source](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) |
| <a name="ref-38"></a>[38] | pipeline controls | [Open source](https://triton-lang.org/main/python-api/generated/triton.language.range.html) |
| <a name="ref-39"></a>[39] | language basics | [Open source](https://tilelang.com/programming_guides/language_basics.html) |
| <a name="ref-40"></a>[40] | instruction reference | [Open source](https://tilelang.com/programming_guides/instructions.html) |
| <a name="ref-41"></a>[41] | software pipelines | [Open source](https://tilelang.com/programming_guides/software_pipeline.html) |
| <a name="ref-42"></a>[42] | Gluon tensor layouts | [Open source](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html) |
| <a name="ref-43"></a>[43] | Bisheng overview | [Open source](https://asc.gitcode.com/guide/programming_guide/compilation_and_execution/operator_compilation/bisheng_compiler.html) |
| <a name="ref-44"></a>[44] | API-level guide | [Open source](https://asc.gitcode.com/asc_how_to_choose_api.html) |
| <a name="ref-45"></a>[45] | C++ Tensor model | [Open source](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/cpp_tensor_programming_overview.html) |
| <a name="ref-46"></a>[46] | Reg vector APIs | [Open source](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/reg_vector_computation.html) |
| <a name="ref-47"></a>[47] | TPipe/TQue model | [Open source](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_10064.html) |
| <a name="ref-48"></a>[48] | compiler auto-sync option | [Open source](https://asc.gitcode.com/guide/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.html) |
| <a name="ref-49"></a>[49] | high-level Matmul | [Open source](https://asc.gitcode.com/guide/operator_practice/simd_operator_impl/matrix_advanced_api/operator_impl.html) |
| <a name="ref-50"></a>[50] | DSL-to-AscendC concept map | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/appendix.md) |
| <a name="ref-51"></a>[51] | RingBuffer API | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/api/data-structures/storage/ring-buffer/overview.md) |
| <a name="ref-52"></a>[52] | control flow | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/control-flow.md) |
| <a name="ref-53"></a>[53] | Staging model | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/code-generation.md) |
| <a name="ref-54"></a>[54] | type system | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/docs/zh/guide/type-system.md) |
| <a name="ref-55"></a>[55] | AscTile overview | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/README.md) |
| <a name="ref-56"></a>[56] | Operation definitions | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/include/asctile/Dialect/AscTile/IR/Ops.td) |
| <a name="ref-57"></a>[57] | compiler schedule | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/python/asctile/runtime/compiler.py) |
| <a name="ref-58"></a>[58] | allocation dispatch | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/lib/Dialect/AscendC/Transforms/DispatchAlloc.cpp) |
| <a name="ref-59"></a>[59] | interpreter execution | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/python/asctile/interpreter/function.py) |
| <a name="ref-60"></a>[60] | range implementation | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/python/asctile/language/range.py) |
| <a name="ref-61"></a>[61] | software-pipelining description | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/docs/design/software-pipelining.md) |
| <a name="ref-62"></a>[62] | Interpreter dispatch | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/python/asctile/runtime/jit.py) |
| <a name="ref-63"></a>[63] | operation implementations | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/python/asctile/interpreter/operations.py) |
| <a name="ref-64"></a>[64] | Transfer completion implementation | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/lib/Dialect/AscTile/Transforms/FulfillDataTransfer.cpp) |
| <a name="ref-65"></a>[65] | location resolution | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/lib/Dialect/AscTile/Transforms/ResolveAutoLocation.cpp) |
| <a name="ref-66"></a>[66] | location verifier | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/lib/Dialect/AscTile/Transforms/VerifyTensorLocation.cpp) |
| <a name="ref-67"></a>[67] | negative tests | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/test/Dialect/AscTile/Transforms/diagnostic/verify-tensor-location.mlir) |
| <a name="ref-68"></a>[68] | VF pipeline regression | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/test/Dialect/AscVF/Transforms/vf-fusion-pipeline.mlir) |
| <a name="ref-69"></a>[69] | ReuseTensorAllocation | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/lib/Dialect/AscendC/Transforms/ReuseTensorAllocation.cpp) |
| <a name="ref-70"></a>[70] | low-level interoperability | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/experimental/asctile/python/asctile/language/low_level_ops.py) |
| <a name="ref-71"></a>[71] | runtime compilation | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/python/asc/runtime/compiler.py) |
| <a name="ref-72"></a>[72] | launcher checks | [Open source](https://gitcode.com/compiler-team/pyasc/blob/936279f2b71ecb8cc781f438e8e51222e2e7f08f/python/asc/runtime/launcher.py) |
| <a name="ref-73"></a>[73] | Issue #77 | [Open source](https://gitcode.com/cann/cannbot-dsl/issues/77) |
| <a name="ref-74"></a>[74] | issue #50 | [Open source](https://gitcode.com/cann/cannbot-dsl/issues/50) |
| <a name="ref-75"></a>[75] | issue #84 | [Open source](https://gitcode.com/cann/cannbot-dsl/issues/84) |
| <a name="ref-76"></a>[76] | issue #85 | [Open source](https://gitcode.com/cann/cannbot-dsl/issues/85) |
| <a name="ref-77"></a>[77] | Comparator source | [Open source](https://gitcode.com/cann/cannbot-dsl/blob/2bbc139b4fe7a7162f80d5aaa49ac8bb392af16c/test/matmul/matmul/test_matmul.py) |
| <a name="ref-78"></a>[78] | issue #48 | [Open source](https://gitcode.com/cann/cannbot-dsl/issues/48) |
| <a name="ref-79"></a>[79] | Issue #75 | [Open source](https://gitcode.com/cann/cannbot-dsl/issues/75) |
