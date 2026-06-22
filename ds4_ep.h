#ifndef DS4_EP_H
#define DS4_EP_H

/* Expert Parallelism (EP) for DeepSeek V4 PRO across N symmetric ranks (Sparks).
 *
 * Each rank owns a CONTIGUOUS id-range of the routed MoE experts; everything
 * else (router, shared expert, attention, KV cache, hyper-connections, indexer,
 * output head, sampling) is REPLICATED on every rank. One sum-all-reduce of the
 * n_embd routed FFN output per layer reassembles the result. See
 * EP_IMPLEMENTATION_PLAN.md and DUAL_SPARK_NOTES.md.
 *
 * This header is pure host C (no CUDA/NCCL) and is always safe to include. The
 * actual collective ops live in the CUDA backend behind -DDS4_EP_BUILD. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_ep_context {
    int      enabled;        /* 0 = single device (no EP); 1 = EP active        */
    int      world_size;     /* number of ranks (Sparks); 1 when disabled       */
    int      rank;           /* this process's rank, 0..world_size-1            */
    uint32_t expert_start;   /* first routed-expert id owned by this rank        */
    uint32_t expert_count;   /* number of routed experts owned by this rank      */
    uint32_t n_total_expert; /* total routed experts in the model (256/384)      */
} ds4_ep_context;

/* Contiguous owned expert range for `rank` of `world_size`, remainder spread
 * over the lowest ranks (rank r<rem gets one extra). A contiguous id-range maps
 * to a contiguous byte-range in the per-expert tensors, so a rank can load only
 * its slice. Writes *start_out/*count_out (either may be NULL).
 * Returns 0 on success, -1 on invalid args (world_size<=0, rank out of range,
 * or n_total_expert==0). */
int ds4_ep_expert_range(uint32_t n_total_expert, int world_size, int rank,
                        uint32_t *start_out, uint32_t *count_out);

/* True iff this rank owns `expert_id`. A disabled context owns every expert
 * (single-device fallback), so callers can branch unconditionally on this. */
int ds4_ep_owns_expert(const ds4_ep_context *ep, uint32_t expert_id);

/* Initialize `ep`. world_size<=1 disables EP (owns all experts). Returns 0 on
 * success, -1 on invalid args. */
int ds4_ep_context_init(ds4_ep_context *ep, int world_size, int rank,
                        uint32_t n_total_expert);

/* Initialize from DS4_EP_WORLD_SIZE / DS4_EP_RANK (default world_size=1 =
 * disabled). Returns 0 on success, -1 if rank>=world_size or partition fails. */
int ds4_ep_context_from_env(ds4_ep_context *ep, uint32_t n_total_expert);

#ifdef __cplusplus
}
#endif

#endif /* DS4_EP_H */
