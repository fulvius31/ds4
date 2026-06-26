/* Tensor Parallelism (TP) host-side mid-dim partition + ncclUniqueId TCP
 * bootstrap. Pure C, no CUDA/NCCL. See ds4_tp.h and TP_IMPLEMENTATION_PLAN.md.
 *
 * Standalone self-test:
 *   cc -DDS4_TP_SELFTEST ds4_tp.c -o ds4_tp_selftest && ./ds4_tp_selftest
 */

#include "ds4_tp.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>

/* ---- mid-dim partition ----------------------------------------------------- */

int ds4_tp_mid_range(uint32_t mid_dim, uint32_t align, int world_size, int rank,
                     uint32_t *start_out, uint32_t *count_out) {
    if (start_out) *start_out = 0;
    if (count_out) *count_out = 0;
    if (world_size <= 0 || rank < 0 || rank >= world_size) return -1;
    if (mid_dim == 0 || align == 0 || (mid_dim % align) != 0) return -1;

    /* Split into align-sized chunks and spread them evenly (remainder to the
     * lowest ranks). Every rank's start/count stays a multiple of align, so the
     * quantized byte offsets (start/QK_K block boundaries) stay exact. */
    const uint32_t n_chunks = mid_dim / align;
    if ((uint32_t)world_size > n_chunks) return -1; /* can't split finer than a chunk */
    const uint32_t w = (uint32_t)world_size;
    const uint32_t r = (uint32_t)rank;
    const uint32_t base = n_chunks / w;
    const uint32_t rem  = n_chunks % w;
    const uint32_t chunk_start = r * base + (r < rem ? r : rem);
    const uint32_t chunk_count = base + (r < rem ? 1u : 0u);

    if (start_out) *start_out = chunk_start * align;
    if (count_out) *count_out = chunk_count * align;
    return 0;
}

int ds4_tp_context_init(ds4_tp_context *tp, int world_size, int rank,
                        uint32_t mid_dim, uint32_t align) {
    if (!tp) return -1;
    memset(tp, 0, sizeof(*tp));
    tp->mid_dim = mid_dim;

    if (world_size <= 1) {
        tp->enabled    = 0;
        tp->world_size = 1;
        tp->rank       = 0;
        tp->mid_start  = 0;
        tp->mid_count  = mid_dim;
        return 0;
    }

    uint32_t start = 0, count = 0;
    if (ds4_tp_mid_range(mid_dim, align, world_size, rank, &start, &count) != 0)
        return -1;

    tp->enabled    = 1;
    tp->world_size = world_size;
    tp->rank       = rank;
    tp->mid_start  = start;
    tp->mid_count  = count;
    return 0;
}

int ds4_tp_context_from_env(ds4_tp_context *tp, uint32_t mid_dim, uint32_t align) {
    int world_size = 1, rank = 0;
    const char *ws = getenv("DS4_TP_WORLD_SIZE");
    const char *rk = getenv("DS4_TP_RANK");
    if (ws && ws[0]) { long v = strtol(ws, NULL, 10); if (v >= 1 && v <= 64) world_size = (int)v; }
    if (rk && rk[0]) { long v = strtol(rk, NULL, 10); if (v >= 0 && v <  64) rank = (int)v; }
    if (rank >= world_size) return -1;
    return ds4_tp_context_init(tp, world_size, rank, mid_dim, align);
}

/* ---- ncclUniqueId TCP bootstrap ------------------------------------------- */

static int tp_write_all(int fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n) {
        ssize_t k = send(fd, p, n, 0);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}

static int tp_read_all(int fd, void *buf, size_t n) {
    char *p = (char *)buf;
    while (n) {
        ssize_t k = recv(fd, p, n, 0);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) return -1;   /* EOF before the full id */
        p += k; n -= (size_t)k;
    }
    return 0;
}

static int tp_listen_fd(const char *host, const char *port, int backlog) {
    struct addrinfo hints, *res = NULL, *ai;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    if (getaddrinfo(host, port, &hints, &res) != 0) return -1;
    int fd = -1;
    for (ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, backlog) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static int tp_connect_fd(const char *host, const char *port) {
    /* The master (rank 0) listener may not be up yet; retry ~60s. */
    for (int attempt = 0; attempt < 600; ++attempt) {
        struct addrinfo hints, *res = NULL, *ai;
        memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host, port, &hints, &res) == 0) {
            for (ai = res; ai; ai = ai->ai_next) {
                int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0) continue;
                if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    return fd;
                }
                close(fd);
            }
            freeaddrinfo(res);
        }
        usleep(100000); /* 100 ms */
    }
    return -1;
}

int ds4_tp_bootstrap_exchange(int world_size, int rank,
                              const void *local_id, void *id_out, size_t cap) {
    if (!id_out || cap == 0) return -1;
    if (world_size <= 1) {
        if (local_id) memcpy(id_out, local_id, cap);
        return 0;
    }

    const char *addr = getenv("DS4_TP_MASTER_ADDR");
    if (!addr || !addr[0]) addr = "127.0.0.1";
    const char *port = getenv("DS4_TP_MASTER_PORT");
    if (!port || !port[0]) port = "29500";

    if (rank == 0) {
        if (!local_id) return -1;
        memcpy(id_out, local_id, cap);
        int lfd = tp_listen_fd(addr, port, world_size);
        if (lfd < 0) {
            fprintf(stderr, "ds4 TP: bootstrap listen %s:%s failed: %s\n",
                    addr, port, strerror(errno));
            return -1;
        }
        int rc = 0;
        for (int peers = world_size - 1; peers > 0; --peers) {
            int cfd = accept(lfd, NULL, NULL);
            if (cfd < 0) { if (errno == EINTR) { ++peers; continue; } rc = -1; break; }
            if (tp_write_all(cfd, local_id, cap) != 0) rc = -1;
            close(cfd);
            if (rc != 0) break;
        }
        close(lfd);
        return rc;
    }

    int cfd = tp_connect_fd(addr, port);
    if (cfd < 0) {
        fprintf(stderr, "ds4 TP: bootstrap connect %s:%s failed\n", addr, port);
        return -1;
    }
    int rc = tp_read_all(cfd, id_out, cap);
    close(cfd);
    if (rc != 0) fprintf(stderr, "ds4 TP: bootstrap recv id failed\n");
    return rc;
}

#ifdef DS4_TP_SELFTEST
#include <assert.h>

int main(void) {
    uint32_t s, c;
    const uint32_t QK = 256;

    /* Flash n_ff_exp 2048 / 2 ranks -> [0,1024) and [1024,1024), both 4*256 */
    assert(ds4_tp_mid_range(2048, QK, 2, 0, &s, &c) == 0 && s == 0    && c == 1024);
    assert(ds4_tp_mid_range(2048, QK, 2, 1, &s, &c) == 0 && s == 1024 && c == 1024);

    /* PRO n_ff_exp 3072 / 2 -> [0,1536) and [1536,1536), both 6*256 */
    assert(ds4_tp_mid_range(3072, QK, 2, 0, &s, &c) == 0 && s == 0    && c == 1536);
    assert(ds4_tp_mid_range(3072, QK, 2, 1, &s, &c) == 0 && s == 1536 && c == 1536);

    /* every start and count is a multiple of QK across world sizes 1..8 */
    for (int W = 1; W <= 8; ++W) {
        uint32_t prev_end = 0, covered = 0;
        for (int r = 0; r < W; ++r) {
            assert(ds4_tp_mid_range(3072, QK, W, r, &s, &c) == 0);
            assert((s % QK) == 0 && (c % QK) == 0);
            assert(s == prev_end);
            prev_end = s + c;
            covered += c;
        }
        assert(covered == 3072 && prev_end == 3072);
    }

    /* remainder spread: 2048 = 8 chunks / 3 ranks -> 3,3,2 chunks */
    assert(ds4_tp_mid_range(2048, QK, 3, 0, &s, &c) == 0 && s == 0    && c == 3*QK);
    assert(ds4_tp_mid_range(2048, QK, 3, 1, &s, &c) == 0 && s == 3*QK && c == 3*QK);
    assert(ds4_tp_mid_range(2048, QK, 3, 2, &s, &c) == 0 && s == 6*QK && c == 2*QK);

    /* invalid args */
    assert(ds4_tp_mid_range(2000, QK, 2, 0, &s, &c) == -1); /* 2000 % 256 != 0 */
    assert(ds4_tp_mid_range(2048, QK, 2, 2, &s, &c) == -1); /* rank out of range */
    assert(ds4_tp_mid_range(2048, QK, 9, 0, &s, &c) == -1); /* 9 > 8 chunks */

    /* context: disabled owns whole mid_dim; enabled honors the slice */
    ds4_tp_context tp;
    assert(ds4_tp_context_init(&tp, 1, 0, 2048, QK) == 0 && tp.enabled == 0 &&
           tp.mid_start == 0 && tp.mid_count == 2048);
    assert(ds4_tp_context_init(&tp, 2, 1, 2048, QK) == 0 && tp.enabled == 1 &&
           tp.mid_start == 1024 && tp.mid_count == 1024);

    printf("ds4_tp selftest: OK\n");
    return 0;
}
#endif /* DS4_TP_SELFTEST */
