/* Expert Parallelism (EP) host-side partition logic. Pure C, no CUDA/NCCL.
 * See ds4_ep.h and EP_IMPLEMENTATION_PLAN.md.
 *
 * Standalone self-test (run wherever a C compiler exists):
 *   cc -DDS4_EP_SELFTEST ds4_ep.c -o ds4_ep_selftest && ./ds4_ep_selftest
 */

#include "ds4_ep.h"

#include <stdlib.h>
#include <string.h>

int ds4_ep_expert_range(uint32_t n_total_expert, int world_size, int rank,
                        uint32_t *start_out, uint32_t *count_out) {
    if (start_out) *start_out = 0;
    if (count_out) *count_out = 0;
    if (world_size <= 0 || rank < 0 || rank >= world_size) return -1;
    if (n_total_expert == 0) return -1;

    /* Even split with the remainder handed to the lowest-numbered ranks. This
     * keeps each rank's range contiguous (so it maps to one contiguous byte
     * span of the per-expert tensors) and balances counts to within one. */
    const uint32_t w    = (uint32_t)world_size;
    const uint32_t r    = (uint32_t)rank;
    const uint32_t base = n_total_expert / w;
    const uint32_t rem  = n_total_expert % w;
    const uint32_t start = r * base + (r < rem ? r : rem);
    const uint32_t count = base + (r < rem ? 1u : 0u);

    if (start_out) *start_out = start;
    if (count_out) *count_out = count;
    return 0;
}

int ds4_ep_owns_expert(const ds4_ep_context *ep, uint32_t expert_id) {
    if (!ep || !ep->enabled) return 1; /* single-device: owns everything */
    return expert_id >= ep->expert_start &&
           expert_id <  ep->expert_start + ep->expert_count;
}

int ds4_ep_context_init(ds4_ep_context *ep, int world_size, int rank,
                        uint32_t n_total_expert) {
    if (!ep) return -1;
    memset(ep, 0, sizeof(*ep));
    ep->n_total_expert = n_total_expert;

    if (world_size <= 1) {
        ep->enabled      = 0;
        ep->world_size   = 1;
        ep->rank         = 0;
        ep->expert_start = 0;
        ep->expert_count = n_total_expert;
        return 0;
    }

    uint32_t start = 0, count = 0;
    if (ds4_ep_expert_range(n_total_expert, world_size, rank, &start, &count) != 0)
        return -1;

    ep->enabled      = 1;
    ep->world_size   = world_size;
    ep->rank         = rank;
    ep->expert_start = start;
    ep->expert_count = count;
    return 0;
}

int ds4_ep_context_from_env(ds4_ep_context *ep, uint32_t n_total_expert) {
    int world_size = 1, rank = 0;
    const char *ws = getenv("DS4_EP_WORLD_SIZE");
    const char *rk = getenv("DS4_EP_RANK");
    if (ws && ws[0]) { long v = strtol(ws, NULL, 10); if (v >= 1 && v <= 64) world_size = (int)v; }
    if (rk && rk[0]) { long v = strtol(rk, NULL, 10); if (v >= 0 && v <  64) rank = (int)v; }
    if (rank >= world_size) return -1;
    return ds4_ep_context_init(ep, world_size, rank, n_total_expert);
}

#ifdef DS4_EP_SELFTEST
#include <stdio.h>
#include <assert.h>

int main(void) {
    uint32_t s, c;

    /* PRO: 384 experts / 2 ranks -> 0..191, 192..383 */
    assert(ds4_ep_expert_range(384, 2, 0, &s, &c) == 0 && s == 0   && c == 192);
    assert(ds4_ep_expert_range(384, 2, 1, &s, &c) == 0 && s == 192 && c == 192);

    /* Flash: 256 experts / 2 -> 128 each */
    assert(ds4_ep_expert_range(256, 2, 0, &s, &c) == 0 && s == 0   && c == 128);
    assert(ds4_ep_expert_range(256, 2, 1, &s, &c) == 0 && s == 128 && c == 128);

    /* remainder spread: 7 / 2 -> rank0 owns 4 (0..3), rank1 owns 3 (4..6) */
    assert(ds4_ep_expert_range(7, 2, 0, &s, &c) == 0 && s == 0 && c == 4);
    assert(ds4_ep_expert_range(7, 2, 1, &s, &c) == 0 && s == 4 && c == 3);

    /* 384 across 3 ranks -> 128 each */
    assert(ds4_ep_expert_range(384, 3, 2, &s, &c) == 0 && s == 256 && c == 128);

    /* coverage + disjointness for many configs */
    for (int W = 1; W <= 8; ++W) {
        uint32_t prev_end = 0, covered = 0;
        for (int r = 0; r < W; ++r) {
            assert(ds4_ep_expert_range(384, W, r, &s, &c) == 0);
            assert(s == prev_end);          /* contiguous, no gaps/overlap */
            prev_end = s + c;
            covered += c;
        }
        assert(covered == 384 && prev_end == 384);
    }

    /* invalid args */
    assert(ds4_ep_expert_range(384, 2, 2, &s, &c) == -1); /* rank out of range */
    assert(ds4_ep_expert_range(0,   2, 0, &s, &c) == -1); /* no experts        */

    /* context: disabled owns everything; enabled honors the range */
    ds4_ep_context ep;
    assert(ds4_ep_context_init(&ep, 1, 0, 384) == 0 && ep.enabled == 0);
    assert(ds4_ep_owns_expert(&ep, 300) == 1);
    assert(ds4_ep_context_init(&ep, 2, 1, 384) == 0 && ep.enabled == 1 &&
           ep.expert_start == 192 && ep.expert_count == 192);
    assert(ds4_ep_owns_expert(&ep, 191) == 0);
    assert(ds4_ep_owns_expert(&ep, 192) == 1);
    assert(ds4_ep_owns_expert(&ep, 383) == 1);

    printf("ds4_ep selftest: OK\n");
    return 0;
}
#endif /* DS4_EP_SELFTEST */
