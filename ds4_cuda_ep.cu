/* ds4_cuda_ep.cu — Expert Parallelism (EP) collectives for the CUDA backend.
 *
 * NCCL sum-all-reduce over RoCE between symmetric ranks (DGX Sparks), plus the
 * tiny router-mask kernel that makes each rank contribute only its owned experts.
 * Compiled ONLY under -DDS4_EP_BUILD (Makefile target `cuda-spark-ep`); otherwise
 * this translation unit is empty and harmless. Host-side expert partitioning and
 * the ncclUniqueId TCP bootstrap are in ds4_ep.{c,h}; the ABI is in ds4_gpu.h.
 *
 * Stream model: ds4's compute kernels run on the DEFAULT stream (verified:
 * `kernel<<<grid,block>>>` with no stream arg). The mask and the all-reduce here
 * also run on the default stream, so they are correctly ordered after the
 * routed-MoE/router-select producer and before the shared-add consumer with no
 * extra synchronization. NCCL completion is enforced by stream ordering; errors
 * surface at ds4's next flush (cudaDeviceSynchronize).
 *
 * Validate on the Sparks (see EP_IMPLEMENTATION_PLAN.md §10.3): build with
 * `make cuda-spark-ep`, run the replicated-weights bit-identical smoke, then
 * EP-vs-pipeline. Requires NCCL on native IB (NCCL_NET_PLUGIN=none,
 * NCCL_IB_MERGE_NICS=1, current ConnectX-7 firmware). */

#ifdef DS4_EP_BUILD

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static ncclComm_t g_ep_comm  = NULL;
static int        g_ep_world  = 1;   /* >1 only after a successful init */

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

/* rank 0 mints the 128-byte ncclUniqueId; ds4_ep_bootstrap_exchange broadcasts
 * it to peers (host TCP) before every rank calls ds4_gpu_collective_init. */
extern "C" int ds4_gpu_collective_unique_id(void *out_id, uint64_t bytes) {
    if (!out_id || bytes != (uint64_t)sizeof(ncclUniqueId)) {
        fprintf(stderr, "ds4 EP: unique_id needs %zu bytes, got %llu\n",
                sizeof(ncclUniqueId), (unsigned long long)bytes);
        return 0;
    }
    ncclUniqueId id;
    if (!ep_nccl_ok(ncclGetUniqueId(&id), "GetUniqueId")) return 0;
    memcpy(out_id, &id, sizeof(id));
    return 1;
}

extern "C" int ds4_gpu_collective_init(int world_size, int rank,
                                       const void *bootstrap_id,
                                       uint64_t bootstrap_bytes) {
    if (world_size <= 1) { g_ep_world = 1; return 1; }  /* EP disabled */
    if (g_ep_comm) return 1;                            /* already initialized */
    if (!bootstrap_id || bootstrap_bytes != (uint64_t)sizeof(ncclUniqueId)) {
        fprintf(stderr, "ds4 EP: bad bootstrap id (%llu bytes, expected %zu)\n",
                (unsigned long long)bootstrap_bytes, sizeof(ncclUniqueId));
        return 0;
    }
    ncclUniqueId id;
    memcpy(&id, bootstrap_id, sizeof(id));
    if (!ep_nccl_ok(ncclCommInitRank(&g_ep_comm, world_size, id, rank),
                    "CommInitRank")) {
        g_ep_comm = NULL;
        return 0;
    }
    g_ep_world = world_size;
    return 1;
}

/* In-place sum-all-reduce of `count` f32 elements across ranks, on the default
 * stream. No-op (success) when EP is off. Returns 1 on success, 0 on error. */
extern "C" int ds4_gpu_all_reduce_f32(ds4_gpu_tensor *tensor, uint64_t count) {
    if (g_ep_world <= 1) return 1;
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
    return ep_nccl_ok(ncclAllReduce(buf, buf, (size_t)count, ncclFloat32, ncclSum,
                                    g_ep_comm, /*stream=default*/ 0), "AllReduce");
}

/* Zero the router weight for any selected slot whose GLOBAL expert id is outside
 * [owned_start, owned_start+owned_count). Run the router replicated, then mask,
 * so each rank's routed_out covers only its owned experts; the per-layer
 * all-reduce reassembles the full routed contribution. Masking the weight (not
 * the id) is required: the MoE sort clamps id<0 to expert 0. */
__global__ static void ep_router_mask_kernel(const int32_t *selected, float *weights,
                                             uint32_t n_slots, uint32_t owned_start,
                                             uint32_t owned_end) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_slots) return;
    int32_t e = selected[i];
    int owned = (e >= 0) && ((uint32_t)e >= owned_start) && ((uint32_t)e < owned_end);
    if (!owned) weights[i] = 0.0f;
}

extern "C" int ds4_gpu_router_mask_owned(ds4_gpu_tensor *selected,
                                         ds4_gpu_tensor *weights,
                                         uint32_t n_slots, uint32_t owned_start,
                                         uint32_t owned_count) {
    if (g_ep_world <= 1 || n_slots == 0) return 1;   /* EP off: nothing to mask */
    const int32_t *sel = (const int32_t *)ds4_gpu_tensor_contents(selected);
    float         *w   = (float *)ds4_gpu_tensor_contents(weights);
    if (!sel || !w) {
        fprintf(stderr, "ds4 EP: router_mask on null buffer\n");
        return 0;
    }
    uint32_t threads = 256u;
    uint32_t blocks  = (n_slots + threads - 1u) / threads;
    ep_router_mask_kernel<<<blocks, threads>>>(sel, w, n_slots, owned_start,
                                               owned_start + owned_count);
    return ep_cuda_ok(cudaGetLastError(), "router_mask launch");
}

extern "C" void ds4_gpu_collective_shutdown(void) {
    if (g_ep_comm) { ncclCommDestroy(g_ep_comm); g_ep_comm = NULL; }
    g_ep_world = 1;
}

#endif /* DS4_EP_BUILD */
