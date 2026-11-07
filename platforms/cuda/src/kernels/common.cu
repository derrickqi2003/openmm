/**
 * This file contains CUDA definitions for the macros and functions needed for the
 * common compute framework.
 */

#define KERNEL extern "C" __global__
#define DEVICE __device__
#define LOCAL __shared__
#define LOCAL_ARG
#define GLOBAL
#define PRIVATE
#define RESTRICT __restrict__
#define LOCAL_ID threadIdx.x
#define LOCAL_SIZE blockDim.x
#define GLOBAL_ID (blockIdx.x*blockDim.x+threadIdx.x)
#define GLOBAL_SIZE (blockDim.x*gridDim.x)
#define GROUP_ID blockIdx.x
#define NUM_GROUPS gridDim.x
#define SYNC_THREADS __syncthreads();
#define MEM_FENCE __threadfence_block();
#define ATOMIC_ADD(dest, value) atomicAdd(dest, value)
#define FLT_MAX 3.40282347e+38f

// CUDA 13 renamed the 16 byte aligned 4-vectors of 64 bit integers; before it they have no suffix.
typedef long long mm_long;
typedef longlong2 mm_long2;
typedef longlong3 mm_long3;
#if __CUDACC_VER_MAJOR__ < 13
typedef longlong4 mm_long4;
#else
typedef longlong4_16a mm_long4;
#endif
typedef unsigned long long mm_ulong;
typedef ulonglong2 mm_ulong2;
typedef ulonglong3 mm_ulong3;
#if __CUDACC_VER_MAJOR__ < 13
typedef ulonglong4 mm_ulong4;
#else
typedef ulonglong4_16a mm_ulong4;
#endif

#define SUPPORTS_64_BIT_ATOMICS 1
#define SUPPORTS_DOUBLE_PRECISION 1

__device__ inline long long realToFixedPoint(real x) {
    return static_cast<long long>(x * 0x100000000);
}
