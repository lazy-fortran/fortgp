# Optional CUDA kernel reductions

`fortgp_reductions` applies the anisotropic squared-exponential covariance to
weights without storing a pairwise matrix. `rbf_apply` uses OpenMP across query
points; `rbf_apply_jvp` returns its directional derivative with respect to query
points, source points, amplitude, length scales and weights. The array layout is
`x(ndim,nquery)`, `z(ndim,nsource)`; amplitude is a standard deviation and every
length scale is positive. Both return an allocated vector and `fortnum_status_t`.

The optional `fortgp_cuda` module exposes the same operations as `rbf_apply_cuda`
and `rbf_apply_cuda_jvp`. It tiles source points in shared memory and fuses each
pairwise evaluation with its reduction. Inputs, accumulation and outputs use
64-bit floating point. The pointwise value and derivative code for both targets
comes from the same FortSym specification in `gen/generate.f90`; ordinary builds
need no symbolic runtime. The optional package requires an NVIDIA GPU, the CUDA
toolkit, CMake and a compatible C++ compiler. The main CPU package does not.

Build the backend and run the native Fortran tests, choosing an architecture
supported by the intended GPU:

```sh
cmake -S cuda -B build/cuda-backend -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cuda-backend --parallel 4
cd cuda
fpm clean --skip
fpm test --flag "-L$PWD/../build/cuda-backend -L/usr/local/cuda/lib64"
fpm run --profile release \
  --flag "-L$PWD/../build/cuda-backend -L/usr/local/cuda/lib64" -- 4096 4096
```

Use `CMAKE_CUDA_COMPILER` and `CMAKE_CUDA_HOST_COMPILER` to select an installed
compatible toolkit/compiler when necessary. Its runtime library directory must
also be discoverable by the dynamic loader. Repeat `fpm clean --skip` after
rebuilding the external backend so that its Fortran executables are relinked. The measured installation used
`/opt/cuda/bin/nvcc` (CUDA 13.4), `/usr/bin/g++-14`, and `/opt/cuda/lib64`.
The older installed CUDA 12.8 headers conflicted with glibc 2.44 during CMake's
compiler check, so that toolchain was not used.

The GPU methods accept optional zero-based `device` (default 0), `device_bytes`
and `kernel_ms` arguments. `device_bytes` counts the explicitly allocated device
arrays and excludes the CUDA context, allocator overhead and shared memory.
`kernel_ms` is measured with CUDA events. The benchmark separately measures total
API time, including allocation, input validation, host-to-device copies,
synchronization, device-to-host copy and release. Each call owns and releases
its device arrays. The driver must support the selected toolkit; runtime errors
return a failed status and leave the output unallocated. Shared-memory capacity
also limits the supported feature dimension, with a failed status when exceeded.

Run CPU strong scaling with:

```sh
cd benchmark/reductions_cpu
OMP_NUM_THREADS=16 fpm run --profile release \
  --flag '-O3 -flto -march=native' -- 4096 4096
```

The programs report the minimum of three timed calls after a reference/warm-up
call. Tests compare against an independent dense formula, CPU evaluation and
central finite differences, including incomplete source and query tiles.

Storage grows with the input and output arrays rather than the number of
pairwise interactions. Arithmetic still grows as `nquery*nsource*ndim`.
This is a covariance operator and its derivative, not a complete matrix-free GP
fit, uncertainty calculation, likelihood or sampler. It does not yet provide a
comparison with KeOps or GPyTorch.
