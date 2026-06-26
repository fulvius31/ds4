/* ds4_cuda_tp.cu — Tensor Parallelism (TP) collectives for the CUDA backend.
 *
 * NCCL sum-all-reduce over RoCE between symmetric ranks (DGX Sparks). Each rank
 * computes its slice of the expert FFN intermediate dim, then this all-reduce
 * reassembles the n_embd routed output per layer. Compiled ONLY under
 * -DDS4_TP_BUILD (Makefile target `cuda-spark-tp`); otherwise this translation
 * unit is empty and harmless. Host-side mid-dim partition and the ncclUniqueId
 * TCP bootstrap are in ds4_tp.{c,h}; the ABI is in ds4_gpu.h.
 *
 * Stream model: ds4's compute kernels run on the DEFAULT stream
 * (`kernel<<<grid,block>>>` with no stream arg). The all-reduce here also runs
 * on the default stream, so it is correctly ordered after the routed-MoE
 * producer and before the shared-add consumer with no extra synchronization.
 * NCCL completion is enforced by stream ordering; errors surface at ds4's next
 * cudaDeviceSynchronize.
 *
 * Requires NCCL on native IB (NCCL_NET_PLUGIN=none, NCCL_IB_MERGE_NICS=1,
 * current ConnectX-7 firmware). */

#ifdef DS4_TP_BUILD

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static ncclComm_t g_tp_comm  = NULL;
static int        g_tp_world  = 1;   /* >1 only after a successful init */

static int tp_nccl_ok(ncclResult_t r, const char *what) {
    if (r != ncclSuccess) {
        fprintf(stderr, "ds4 TP: NCCL %s: %s\n", what, ncclGetErrorString(r));
        return 0;
    }
    return 1;
}

/* rank 0 mints the 128-byte ncclUniqueId; ds4_tp_bootstrap_exchange broadcasts
 * it to peers (host TCP) before every rank calls ds4_gpu_collective_init. */
extern "C" int ds4_gpu_collective_unique_id(void *out_id, uint64_t bytes) {
    if (!out_id || bytes != (uint64_t)sizeof(ncclUniqueId)) {
        fprintf(stderr, "ds4 TP: unique_id needs %zu bytes, got %llu\n",
                sizeof(ncclUniqueId), (unsigned long long)bytes);
        return 0;
    }
    ncclUniqueId id;
    if (!tp_nccl_ok(ncclGetUniqueId(&id), "GetUniqueId")) return 0;
    memcpy(out_id, &id, sizeof(id));
    return 1;
}

extern "C" int ds4_gpu_collective_init(int world_size, int rank,
                                       const void *bootstrap_id,
                                       uint64_t bootstrap_bytes) {
    if (world_size <= 1) { g_tp_world = 1; return 1; }  /* TP disabled */
    if (g_tp_comm) return 1;                            /* already initialized */
    if (!bootstrap_id || bootstrap_bytes != (uint64_t)sizeof(ncclUniqueId)) {
        fprintf(stderr, "ds4 TP: bad bootstrap id (%llu bytes, expected %zu)\n",
                (unsigned long long)bootstrap_bytes, sizeof(ncclUniqueId));
        return 0;
    }
    ncclUniqueId id;
    memcpy(&id, bootstrap_id, sizeof(id));
    if (!tp_nccl_ok(ncclCommInitRank(&g_tp_comm, world_size, id, rank),
                    "CommInitRank")) {
        g_tp_comm = NULL;
        return 0;
    }
    g_tp_world = world_size;
    return 1;
}

/* In-place sum-all-reduce of `count` f32 elements across ranks, on the default
 * stream. No-op (success) when TP is off. Returns 1 on success, 0 on error. */
extern "C" int ds4_gpu_all_reduce_f32(ds4_gpu_tensor *tensor, uint64_t count) {
    if (g_tp_world <= 1) return 1;
    if (!g_tp_comm) {
        fprintf(stderr, "ds4 TP: all_reduce before collective_init\n");
        return 0;
    }
    if (!tensor || count == 0) return 1;
    void *buf = ds4_gpu_tensor_contents(tensor);   /* CUDA tensor -> device ptr */
    if (!buf) {
        fprintf(stderr, "ds4 TP: all_reduce on null tensor buffer\n");
        return 0;
    }
    return tp_nccl_ok(ncclAllReduce(buf, buf, (size_t)count, ncclFloat32, ncclSum,
                                    g_tp_comm, /*stream=default*/ 0), "AllReduce");
}

extern "C" void ds4_gpu_collective_shutdown(void) {
    if (g_tp_comm) { ncclCommDestroy(g_tp_comm); g_tp_comm = NULL; }
    g_tp_world = 1;
}

#endif /* DS4_TP_BUILD */
