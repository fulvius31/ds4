/* ds4_cuda_ep.cu — Expert Parallelism (EP) collectives for the CUDA backend.
 *
 * NCCL sum-all-reduce over RoCE between symmetric ranks (DGX Sparks). Compiled
 * ONLY under -DDS4_EP_BUILD (Makefile target `cuda-spark-ep`); otherwise this
 * translation unit is empty and harmless. Host-side expert partitioning is in
 * ds4_ep.c; the ABI is declared in ds4_gpu.h. See EP_IMPLEMENTATION_PLAN.md.
 *
 * It relies only on the public GPU ABI (ds4_gpu_tensor_contents returns the
 * device pointer for a CUDA tensor — struct ds4_gpu_tensor = {void*ptr; ...}),
 * so it does NOT touch ds4_cuda.cu internals.
 *
 * NOT COMPILED/TESTED on the authoring box (no CUDA/NCCL/GPU). Validate on the
 * Sparks:  make cuda-spark-ep  then the replicated-weights bit-identical smoke
 * in EP_IMPLEMENTATION_PLAN.md §10.3. Requires NCCL on the native IB path
 * (NCCL_NET_PLUGIN=none, NCCL_IB_MERGE_NICS=1, current ConnectX-7 firmware). */

#ifdef DS4_EP_BUILD

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <string.h>

static ncclComm_t   g_ep_comm   = NULL;
static cudaStream_t g_ep_stream  = NULL;
static int          g_ep_world   = 1;   /* >1 only after a successful init */

static int ep_cuda_ok(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "ds4 EP: CUDA %s: %s\n", what, cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

static int ep_nccl_ok(ncclResult_t r, const char *what) {
    if (r != ncclSuccess) {
        fprintf(stderr, "ds4 EP: NCCL %s: %s\n", what, ncclGetErrorString(r));
        return 0;
    }
    return 1;
}

/* Join the NCCL communicator. Every rank must call this with the SAME 128-byte
 * ncclUniqueId (broadcast out-of-band by rank 0 over the ds4_distributed TCP
 * rendezvous) and its own rank. world_size<=1 means EP is off (no-op success).
 * Returns 1 on success, 0 on failure. */
extern "C" int ds4_gpu_collective_init(int world_size, int rank,
                                       const void *bootstrap_id,
                                       uint64_t bootstrap_bytes) {
    if (world_size <= 1) {            /* EP disabled: single device, no comm */
        g_ep_world = 1;
        return 1;
    }
    if (g_ep_comm) return 1;          /* already initialized */
    if (!bootstrap_id || bootstrap_bytes != (uint64_t)sizeof(ncclUniqueId)) {
        fprintf(stderr, "ds4 EP: bad bootstrap id (%llu bytes, expected %zu)\n",
                (unsigned long long)bootstrap_bytes, sizeof(ncclUniqueId));
        return 0;
    }
    if (!ep_cuda_ok(cudaStreamCreateWithFlags(&g_ep_stream, cudaStreamNonBlocking),
                    "ep stream create")) {
        return 0;
    }
    ncclUniqueId id;
    memcpy(&id, bootstrap_id, sizeof(id));
    if (!ep_nccl_ok(ncclCommInitRank(&g_ep_comm, world_size, id, rank),
                    "CommInitRank")) {
        cudaStreamDestroy(g_ep_stream);
        g_ep_stream = NULL;
        g_ep_comm   = NULL;
        return 0;
    }
    g_ep_world = world_size;
    return 1;
}

/* In-place sum-all-reduce of `count` float32 elements of `tensor` across all
 * ranks. No-op when EP is off (world<=1). Synchronous: it blocks on the
 * collective stream, matching ds4's synchronous CUDA execution model (begin/
 * flush_commands are cudaDeviceSynchronize). Returns 1 on success, 0 on error. */
extern "C" int ds4_gpu_all_reduce_f32(ds4_gpu_tensor *tensor, uint64_t count) {
    if (g_ep_world <= 1) return 1;   /* EP disabled: nothing to reduce */
    if (!g_ep_comm) {
        fprintf(stderr, "ds4 EP: all_reduce before collective_init\n");
        return 0;
    }
    if (!tensor || count == 0) return 1;
    void *buf = ds4_gpu_tensor_contents(tensor);   /* CUDA tensor -> device ptr */
    if (!buf) {
        fprintf(stderr, "ds4 EP: all_reduce on null tensor buffer\n");
        return 0;
    }
    if (!ep_nccl_ok(ncclAllReduce(buf, buf, (size_t)count, ncclFloat32, ncclSum,
                                  g_ep_comm, g_ep_stream), "AllReduce")) {
        return 0;
    }
    return ep_cuda_ok(cudaStreamSynchronize(g_ep_stream), "ep stream sync");
}

extern "C" void ds4_gpu_collective_shutdown(void) {
    if (g_ep_comm)   { ncclCommDestroy(g_ep_comm);     g_ep_comm   = NULL; }
    if (g_ep_stream) { cudaStreamDestroy(g_ep_stream); g_ep_stream = NULL; }
    g_ep_world = 1;
}

#endif /* DS4_EP_BUILD */
