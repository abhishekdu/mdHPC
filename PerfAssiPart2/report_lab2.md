https://github.com/abhishekdu/mdHPC

# Lab Report: Accelerating Smith-Waterman Sequence Alignment

Team: \[2025MCS1120, 2025MCS2963]


## 1.  Problem Description

The objective of this lab was to optimize the Smith-Waterman algorithm,
a fundamental dynamic programming technique used in Computational
Biology for local sequence alignment. The core challenge of
Smith-Waterman is its computational complexity: the standard algorithm
scales as O(N×M) in both time and space. For large sequences (e.g., N \>
10,000), the O(NM) memory requirement quickly exceeds physical RAM, and
the computational cost becomes prohibitive for real-world biological
datasets.

Our goal was to iteratively accelerate a given Python baseline using
architecture-level optimizations. We aimed to maximize throughput
(measured in Cell Updates Per Second or CUPS) and scalability by
exploiting:

1.  Memory Hierarchy: Improving cache locality through blocking
    (tiling).

2.  Space Optimization: Reducing memory footprint from O(NM) to O(N) to
    prevent memory exhaustion.

3.  Instruction Level Parallelism: Utilizing SIMD (AVX2) vector
    instructions to process multiple cells per cycle.

## 2.  Baseline

### 2.1. Python Baseline

The provided baseline was a standard Python implementation using numpy.

Performance: For a sequence length of N = 5,000, execution time was
27.86 seconds.

Scalability: The implementation failed to scale. At N = 20,000, the
process was terminated ("Killed") by the OS, likely due to memory
exhaustion or excessive runtime.

### 2.2. Scalar C Baseline

To establish a compiled baseline, we implemented a direct translation of
the algorithm in C.

Implementation Details: The baseline employs a naive memory allocation
strategy using a pointer-to-pointer structure (int \*\*H). Each row of
the matrix is allocated via a separate calloc call. While functionally
correct, this results in non-contiguous memory allocation, causing the
CPU to perform expensive "pointer chasing" and suffering from poor
spatial locality.

Performance: For N = 5,000, execution time was 0.24 seconds,
representing a \~115x speedup over Python purely due to compilation and
static typing.

Limitations: Despite the speedup, the O(NM) memory complexity persisted.
The overhead of managing N separate heap allocations contributed to
memory fragmentation. Consequently, the baseline failed at N = 50,000
due to memory constraints (OOM).

## 3.  Optimization Implementation

To address the identified bottlenecks, we implemented the following
architectural optimizations in C.

### 3.1. Space Optimization (O(N))

The standard approach stores the entire scoring matrix. For N = 60,000,
this requires allocating \~14.4 GB of RAM.

Strategy: We transitioned from the baseline's heavy 2D allocation to a
minimal O(2×M) footprint by storing only the curr_row and prev_row.

Impact: This optimization reduced the memory requirement for N = 60,000
to mere kilobytes.

### 3.2. Cache-Aware Blocking (Tiling)

We implemented loop tiling with a BLOCK_SIZE of 64 bytes to optimize for
L1 data cache locality.

Reasoning: In the standard row-major traversal, accessing the vertical
sequence seq2 causes repeated cache evictions. By processing the matrix
in small 64×64 blocks, we ensure that relevant segments remain hot in
the cache.

### 3.3. SIMD Vectorization (AVX2)

We exploited Data Level Parallelism using Intel AVX2 intrinsics.

Strategy: We processed 8 elements at a time using 256-bit registers.

Branch Elimination: We replaced conditional logic with vector blending
and max operations, significantly reducing branch mispredictions.

## 4.  Experimental Methodology

Hardware Environment: \[Insert CPU Model & RAM\].

Compilation: gcc with flags -O3 -mavx2 -march=native.

Dataset: Random DNA sequences (A, C, G, T), lengths from 5,000 to
70,000.

## 5.  Results

| N      | Optimized C (s) | Baseline C (s) | Speedup vs C | Speedup vs Python |
|--------|-----------------|----------------|--------------|-------------------|
| 5,000  | 0.0679          | 0.2418         | 3.55x        | 409.79x           |
| 10,000 | 0.1095          | 1.1350         | 10.36x       | 1006x             |
| 20,000 | 0.4577          | 5.0847         | 11.10x       | N/A               |
| 30,000 | 0.9981          | 11.0027        | 11.02x       | N/A               |
| 40,000 | 1.8954          | 30.5130        | 16.09x       | N/A               |
| 50,000 | 2.7118          | Killed         | Infinite     | N/A               |
| 60,000 | 3.9021          | Killed         | Infinite     | N/A               |
| 70,000 | 5.2761          | Killed         | Infinite     | N/A               |
  --------------------------------------------------------------------------------

## 6.  Microarchitecture Profiling

### 6.1. Cache Hierarchy

At N = 60,000, L1 miss rate was 0.02%, LLC misses negligible.

### 6.2. Instruction Throughput

IPC = 3.72, near hardware maximum.

### 6.3. Branch Prediction

Branch miss rate: 2.88%.

## 7.  Discussion & Analysis

The dominant bottleneck is compute latency, not memory. High IPC and low
cache miss rates validate the optimization strategy.

## 8.  Reproducibility

Source Code: sw_opt.c\
Build: make\
Run: ./run.sh\
Profile: perf stat -ddd ./gemm_opt 60000
