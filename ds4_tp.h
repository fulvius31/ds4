#ifndef DS4_TP_H
#define DS4_TP_H

/* Tensor Parallelism (TP) for DeepSeek V4 across 2 symmetric ranks (DGX Sparks).
 *
 * Each rank runs ALL selected experts but only its contiguous slice of the
 * expert FFN intermediate dimension (n_ff_exp): column-parallel gate/up,
 * row-parallel down. One sum-all-reduce of the n_embd routed output per layer
 * reassembles the result. Unlike Expert Parallelism (whole-expert sharding),
 * TP is perfectly load-balanced at batch=1 and is the lever for single-stream
 * decode latency. Target: TP=2 on two Sparks for the Flash model (fits resident,
 * so no streaming). See TP_IMPLEMENTATION_PLAN.md.
 *
 * This header is pure host C (no CUDA/NCCL). The collective ops live in the CUDA
 * backend behind -DDS4_TP_BUILD (ds4_cuda_tp.cu); the ABI is in ds4_gpu.h. */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_tp_context {
    int      enabled;     /* 0 = single device (no TP); 1 = TP active           */
    int      world_size;  /* number of ranks; 1 when disabled                   */
    int      rank;        /* this process's rank, 0..world_size-1               */
    uint32_t mid_dim;     /* full expert intermediate dim (n_ff_exp)            */
    uint32_t mid_start;   /* first mid-dim index this rank computes             */
    uint32_t mid_count;   /* number of mid-dim elements this rank computes      */
} ds4_tp_context;

/* Contiguous, `align`-aligned mid-dim slice for `rank` of `world_size`. Splits
 * mid_dim into mid_dim/align chunks and spreads them evenly (remainder to the
 * lowest ranks) so every rank's start and count are multiples of `align` (the
 * quant block, QK_K=256) — keeping the quantized gate/up row and down column
 * byte offsets block-aligned. Requires mid_dim % align == 0 and
 * world_size <= mid_dim/align. start_out/count_out may be NULL. Returns 0 on
 * success, -1 on invalid args. */
int ds4_tp_mid_range(uint32_t mid_dim, uint32_t align, int world_size, int rank,
                     uint32_t *start_out, uint32_t *count_out);

/* Initialize `tp`. world_size<=1 disables TP (owns the whole mid_dim). Returns
 * 0 on success, -1 on invalid args (e.g. unalignable split). */
int ds4_tp_context_init(ds4_tp_context *tp, int world_size, int rank,
                        uint32_t mid_dim, uint32_t align);

/* Initialize from DS4_TP_WORLD_SIZE / DS4_TP_RANK (default world_size=1 =
 * disabled). Returns 0 on success, -1 on failure. */
int ds4_tp_context_from_env(ds4_tp_context *tp, uint32_t mid_dim, uint32_t align);

/* Broadcast the `cap`-byte ncclUniqueId across ranks over TCP, env-driven:
 * DS4_TP_MASTER_ADDR (default 127.0.0.1) and DS4_TP_MASTER_PORT (default 29500).
 * rank 0 listens and sends `local_id` to every peer; other ranks connect (with
 * retry) and receive into `id_out`. On the single-rank case `local_id` is copied
 * straight to `id_out`. Pure host C. Returns 0 on success, -1 on error. */
int ds4_tp_bootstrap_exchange(int world_size, int rank,
                              const void *local_id, void *id_out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* DS4_TP_H */
