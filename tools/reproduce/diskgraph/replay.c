// SPDX-License-Identifier: MIT
#include <errno.h>
#include <liburing.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>

#define PAGE_BYTES 4096

struct request {
    uint64_t offset;
    uint32_t length;
    uint32_t expected;
    int fd;
};

static int wait_for_batch(struct io_uring *ring, const struct request *requests,
                          size_t count)
{
    if (io_uring_submit(ring) != (int)count)
        return -EIO;
    int error = 0;
    for (size_t i = 0; i < count; i++) {
        struct io_uring_cqe *completion;
        int result;
        do {
            result = io_uring_wait_cqe(ring, &completion);
        } while (result == -EINTR);
        if (result)
            return result;
        uint64_t id = io_uring_cqe_get_data64(completion);
        if (id >= count || completion->res != (int)requests[id].expected)
            error = completion->res < 0 ? completion->res : -EIO;
        io_uring_cqe_seen(ring, completion);
    }
    return error;
}

static uint64_t hash_bytes(uint64_t hash, const unsigned char *data, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int diskgraph_replay(const struct request *requests, const uint32_t *waves, size_t wave_count,
                     unsigned depth, uint32_t largest_read, int use_uring, int verify,
                     uint64_t *bytes, uint64_t *digest)
{
    size_t slots = use_uring ? depth : 1;
    unsigned char *buffer;
    int allocation = posix_memalign((void **)&buffer, PAGE_BYTES, slots * largest_read);
    if (allocation)
        return -allocation;
    for (size_t pos = 0; pos < slots * largest_read; pos += PAGE_BYTES)
        ((volatile unsigned char *)buffer)[pos] = 0;
    struct io_uring ring;
    struct iovec *iov = NULL;
    int error = 0;
    size_t pages_per_slot = (largest_read + PAGE_BYTES - 1) / PAGE_BYTES;
    if (use_uring) {
        iov = calloc(slots * pages_per_slot, sizeof(*iov));
        if (!iov) {
            free(buffer);
            return -ENOMEM;
        }
        error = io_uring_queue_init(depth, &ring, 0);
        if (error) {
            free(iov);
            free(buffer);
            return error;
        }
    }
    *bytes = 0;
    *digest = UINT64_C(14695981039346656037);
    for (size_t wave = 0; wave < wave_count && !error; wave++) {
        if (waves[wave] > depth) {
            error = -EINVAL;
            break;
        }
        for (size_t id = 0; id < waves[wave] && !error; id++) {
            const struct request *req = &requests[id];
            if (use_uring) {
                struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                if (!sqe) {
                    error = -EIO;
                    break;
                }
                unsigned char *dst = buffer + id * largest_read;
                if (req->length > PAGE_BYTES) {
                    size_t pages = req->length / PAGE_BYTES;
                    struct iovec *vec = iov + id * pages_per_slot;
                    for (size_t j = 0; j < pages; j++) {
                        vec[j].iov_base = dst + j * PAGE_BYTES;
                        vec[j].iov_len = PAGE_BYTES;
                    }
                    io_uring_prep_readv(sqe, req->fd, vec, pages, req->offset);
                } else {
                    io_uring_prep_read(sqe, req->fd, dst, req->length, req->offset);
                }
                io_uring_sqe_set_data64(sqe, id);
            } else {
                ssize_t result;
                do {
                    result = pread(req->fd, buffer, req->length, req->offset);
                } while (result < 0 && errno == EINTR);
                if (result < 0)
                    error = -errno;
                else if (result != req->expected)
                    error = -EIO;
                else if (verify)
                    *digest = hash_bytes(*digest, buffer, req->expected);
            }
            *bytes += req->expected;
        }
        if (use_uring && !error) {
            error = wait_for_batch(&ring, requests, waves[wave]);
            if (verify && !error) {
                for (size_t id = 0; id < waves[wave]; id++)
                    *digest = hash_bytes(*digest, buffer + id * largest_read, requests[id].expected);
            }
        }
        requests += waves[wave];
    }
    if (use_uring)
        io_uring_queue_exit(&ring);
    free(iov);
    free(buffer);
    return error;
}
