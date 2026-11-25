# Seismic Core — 3D Stencil Performance Optimization

Step-by-step optimization of a seismic modelling kernel: a 3-axis, 16th-order, **49-point 3D stencil** in C with MPI. Starting from a slow reference code, each optimization was profiled, measured and validated against the reference output, ending **about 29× faster** than the reference.

Course project for *Techniques d'Optimisation de la Parallélisation* (TOP), M1 CHPS, Université Paris-Saclay. Pair work by **Ahmed Taleb Bechir** and **Abdelghani Ameziane**. Full write-up: [`Report.pdf`](Report.pdf).

## Optimization steps

| Step | Change | Effect |
|---|---|---|
| 0 | Fixed CMake MPI detection and mesh-initialisation error handling | Code builds and runs |
| 1 | Compiler flags `-O3`, then `-Ofast -funroll-loops` | 4028 → 2448 ms (avg) |
| 2 | Removed a hidden `sleep()` call found with `strace`; fixed an `MPI_Allreduce` bug skipped by non-root ranks | 362.9 ms (avg) |
| 3 | Pre-computed inverse powers used by the stencil coefficients | Fewer divisions per point |
| 4 | Specialised `pow` implementation | Cheaper inner loop |
| 5 | Cache blocking (64-element blocks) | Fewer cache misses |
| 6 | Mesh data-layout restructuring for locality | 63.9 ms (avg) |
| 7 | OpenMP on the Jacobi solve loop (`collapse`) | 39.8 ms with 4 threads |

Measured on an Intel Core i5-6300U (2 cores / 4 threads), 100×100×100 grid. Results are checked against `reference/` with `scripts/compare.py`.

## Tooling

`perf stat` (cycles, IPC, cache misses) · `perf record` + flame graphs · `gprof` call graphs · `strace` · MAQAO · gcc 11 and clang 14

Raw measurements, binaries and profiles for each version are kept in [`benchmarks/`](benchmarks/) (`non-optimised/`, `optimised-1/` … `optimised-5/`, with x86-64 and Apple M1 runs).

## Build and run

Requirements: CMake 3.16+, a C17 compiler, MPI 3.0+, OpenMP.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/top-stencil config.txt output.txt
```

`config.txt` sets the grid size (`dim_x`, `dim_y`, `dim_z`) and the number of iterations (`niter`).

Validate the result against the reference:

```bash
python3 scripts/compare.py reference/result_100x100x100.txt output.txt
```

## Repository layout

```
src/            main program and stencil library (config, mesh, init, solve, MPI halo exchange)
include/        headers
reference/      reference outputs used for validation
scripts/        result comparison
benchmarks/     per-version results, perf/MAQAO reports, flame graphs
Report.pdf      optimization report
```

## Next steps identified

Explicit SIMD vectorisation (128/256-bit registers) and multi-node MPI scaling on a NUMA cluster were left out for time reasons.

## Authors

**Ahmed Taleb Bechir** — [atbechir.com](https://atbechir.com) · [LinkedIn](https://www.linkedin.com/in/ahmed-taleb-bechir-97a7301a4)
**Abdelghani Ameziane**

Supervisors: Gabriel Dos Santos, Hugo Taboada.
