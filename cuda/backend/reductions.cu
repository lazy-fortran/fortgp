#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include "../generated/fortgp_distance_leaf.cuh"
#include "../generated/fortgp_rbf_leaf.cuh"

namespace {
constexpr int tile = 128;

template<bool tangent>
__global__ void apply_kernel(int d, int n, int m, double a,
    const double *x, const double *z, const double *ell, const double *v,
    const double *dx, const double *dz, double da, const double *dl,
    const double *dv, double *out) {
    extern __shared__ double cache[];
    double *sz = cache;
    double *sv = sz + tile*d;
    double *sdz = sv + tile;
    double *sdv = tangent ? sdz + tile*d : nullptr;
    const int lane = threadIdx.x;
    const int i = blockIdx.x*tile + lane;
    double result = 0.0;
    for (int base = 0; base < m; base += tile) {
        const int source = base + lane;
        if (source < m) {
            for (int k = 0; k < d; ++k) {
                sz[k*tile + lane] = z[source*d + k];
                if (tangent) sdz[k*tile + lane] = dz[source*d + k];
            }
            sv[lane] = v[source];
            if (tangent) sdv[lane] = dv[source];
        }
        __syncthreads();
        if (i < n) {
            const int count = min(tile, m - base);
            for (int j = 0; j < count; ++j) {
                double distance = 0.0, dq = 0.0;
                for (int k = 0; k < d; ++k) {
                    double q, gx, gz, gl;
                    fortgp_distance_leaf(x[i*d + k], sz[k*tile + j], ell[k],
                        &q, &gx, &gz, &gl);
                    distance += q;
                    if (tangent) dq += gx*dx[i*d + k] +
                        gz*sdz[k*tile + j] + gl*dl[k];
                }
                double value, ga, gq;
                fortgp_rbf_leaf(a, distance, &value, &ga, &gq);
                if (tangent) result += (ga*da + gq*dq)*sv[j] + value*sdv[j];
                else result += value*sv[j];
            }
        }
        __syncthreads();
    }
    if (i < n) out[i] = result;
}

struct workspace {
    double *p[10] = {};
    cudaEvent_t start = nullptr, stop = nullptr;
    ~workspace() {
        for (double *item : p) if (item) cudaFree(item);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
    cudaError_t copy(int slot, const double *host, size_t count) {
        cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&p[slot]),
            count*sizeof(double));
        if (e != cudaSuccess || !host) return e;
        return cudaMemcpy(p[slot], host, count*sizeof(double), cudaMemcpyHostToDevice);
    }
};
}

extern "C" int fortgp_cuda_apply(int d, int n, int m, double a,
    const double *x, const double *z, const double *ell, const double *v,
    const double *dx, const double *dz, double da, const double *dl,
    const double *dv, double *out, int derivative, int device,
    size_t *device_bytes, double *kernel_ms) {
    *device_bytes = 0;
    *kernel_ms = 0;
    if (d < 1 || n < 1 || m < 1) return cudaErrorInvalidValue;
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) return error;
    if (device < 0 || device >= count) return cudaErrorInvalidDevice;
    error = cudaSetDevice(device);
    if (error != cudaSuccess) return error;
    cudaDeviceProp properties;
    error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    const size_t shared = tile*sizeof(double)*(derivative ? 2*size_t(d)+2 : size_t(d)+1);
    if (shared > properties.sharedMemPerBlock) return cudaErrorInvalidConfiguration;
    workspace w;
    const size_t nd = size_t(n)*d, md = size_t(m)*d;
    const double *host[] = {x, z, ell, v, nullptr, dx, dz, dl, dv};
    const size_t counts[] = {nd, md, size_t(d), size_t(m), size_t(n),
        nd, md, size_t(d), size_t(m)};
    const int arrays = derivative ? 9 : 5;
    for (int k = 0; k < arrays; ++k) {
        error = w.copy(k, host[k], counts[k]);
        if (error != cudaSuccess) return error;
        *device_bytes += counts[k]*sizeof(double);
    }
    error = cudaEventCreate(&w.start);
    if (error != cudaSuccess) return error;
    error = cudaEventCreate(&w.stop);
    if (error != cudaSuccess) return error;
    error = cudaEventRecord(w.start);
    if (error != cudaSuccess) return error;
    const int blocks = (n - 1)/tile + 1;
    if (derivative) apply_kernel<true><<<blocks, tile, shared>>>(d,n,m,a,
        w.p[0],w.p[1],w.p[2],w.p[3],w.p[5],w.p[6],da,w.p[7],w.p[8],w.p[4]);
    else apply_kernel<false><<<blocks, tile, shared>>>(d,n,m,a,
        w.p[0],w.p[1],w.p[2],w.p[3],nullptr,nullptr,0,nullptr,nullptr,w.p[4]);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    error = cudaEventRecord(w.stop);
    if (error != cudaSuccess) return error;
    error = cudaEventSynchronize(w.stop);
    if (error != cudaSuccess) return error;
    float elapsed;
    error = cudaEventElapsedTime(&elapsed, w.start, w.stop);
    if (error != cudaSuccess) return error;
    *kernel_ms = elapsed;
    return cudaMemcpy(out, w.p[4], size_t(n)*sizeof(double), cudaMemcpyDeviceToHost);
}
