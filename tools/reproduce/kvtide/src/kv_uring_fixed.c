// SPDX-License-Identifier: MIT
//
// KV perf tool over raw io_uring NVMe passthrough with POOL-BACKED fixed
// buffers: registers its I/O buffers from the block device's blk_iobuf
// pool via IORING_REGISTER_BUFFERS_ALLOC_FOR_FILE (opcode 38) and
// submits KV Store/Retrieve as IORING_OP_URING_CMD | IORING_URING_CMD_FIXED.
//
// This is the opt-in data path the pool is designed for: deterministic
// folio-backed buffers, zero-copy DMA, no bounce anywhere. The buffers
// are kernel-backed (no userspace mapping in this series revision), so
// payload content is opaque -- fine for a perf tool.
//
// The device queue must have an active pool: boot with
// blk_iobuf.iobuf_pool_force_order=N or write the runtime param before
// attaching the device, else registration fails with EOPNOTSUPP.
//
// Build: gcc -O2 kv_uring_fixed.c -luring -o kv_uring_fixed
// Run:   sudo ./kv_uring_fixed /dev/ng2n1 store 64 4096 8

#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <linux/nvme_ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAX_QD 256
#define NKEYS 1024
#define MAX_SAMPLES (8 * 1024 * 1024)

#define KV_OPC_STORE    0x01
#define KV_OPC_RETRIEVE 0x02

#ifndef IORING_REGISTER_BUFFERS_ALLOC_FOR_FILE
#define IORING_REGISTER_BUFFERS_ALLOC_FOR_FILE 38
#endif
#ifndef IORING_BUF_ALLOC_READ
#define IORING_BUF_ALLOC_READ  (1U << 3)
#define IORING_BUF_ALLOC_WRITE (1U << 4)
#endif

struct io_uring_buf_alloc_for_file {
	__u32 fd;
	__u32 index;
	__u32 nr_buffers;
	__u32 flags;
	__u64 buffer_size;
	__u32 min_order;
	__u32 pref_order;
	__u32 reserved[2];
};

// Direction is per-slot in the kernel (READ -> destination,
// WRITE -> source), so one slot per direction.
#define SLOT_STORE_SRC 0
#define SLOT_RETR_DST  1

struct slot {
	int busy;
	struct timespec t0;
};

static struct slot g_slots[MAX_QD];
static double *g_lat_us;
static size_t g_nsamples;
static uint64_t g_completed, g_errors, g_inflight;

static double
ts_diff_us(const struct timespec *a, const struct timespec *b)
{
	return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

static int
pool_buf_register(struct io_uring *ring, int dev_fd, unsigned index,
		  unsigned dir_flag, uint64_t size)
{
	struct io_uring_buf_alloc_for_file req = {
		.fd          = (unsigned)dev_fd,
		.index       = index,
		.nr_buffers  = 1,
		.flags       = dir_flag,
		.buffer_size = size,
	};
	return syscall(__NR_io_uring_register, ring->ring_fd,
		       IORING_REGISTER_BUFFERS_ALLOC_FOR_FILE, &req, 1);
}

static void
fill_kv_cmd(struct nvme_uring_cmd *cmd, int is_store, uint32_t nsid,
	    unsigned key_idx, uint32_t vsize)
{
	char key[9];

	snprintf(key, sizeof(key), "pk%06x", key_idx % NKEYS);
	memset(cmd, 0, sizeof(*cmd));
	cmd->opcode = is_store ? KV_OPC_STORE : KV_OPC_RETRIEVE;
	cmd->nsid = nsid;
	memcpy(&cmd->cdw2, key, 4);      // key bytes 0-3
	memcpy(&cmd->cdw3, key + 4, 4);  // key bytes 4-7
	cmd->cdw11 = 8;                  // key length
	cmd->cdw10 = vsize;              // value size / host buffer size
	cmd->addr = 0;                   // offset 0 within the fixed buffer
	cmd->data_len = vsize;
}

static int
submit_one(struct io_uring *ring, int dev_fd, int is_store, uint32_t nsid,
	   int slot_idx, uint32_t vsize, unsigned key_idx)
{
	struct io_uring_sqe *sqe = io_uring_get_sqe(ring);

	if (!sqe)
		return -EAGAIN;
	memset(sqe, 0, sizeof(*sqe));
	sqe->opcode = IORING_OP_URING_CMD;
	sqe->fd = dev_fd;
	sqe->cmd_op = NVME_URING_CMD_IO;
	sqe->uring_cmd_flags = IORING_URING_CMD_FIXED;
	sqe->buf_index = is_store ? SLOT_STORE_SRC : SLOT_RETR_DST;
	sqe->user_data = slot_idx;
	fill_kv_cmd((struct nvme_uring_cmd *)sqe->cmd, is_store, nsid,
		    key_idx, vsize);
	clock_gettime(CLOCK_MONOTONIC, &g_slots[slot_idx].t0);
	g_slots[slot_idx].busy = 1;
	g_inflight++;
	return 0;
}

static void
reap(struct io_uring *ring, int wait)
{
	struct io_uring_cqe *cqe;
	struct timespec t1;
	unsigned head, n = 0;

	if (wait && g_inflight)
		io_uring_wait_cqe(ring, &cqe);
	io_uring_for_each_cqe(ring, head, cqe) {
		struct slot *s = &g_slots[cqe->user_data];

		clock_gettime(CLOCK_MONOTONIC, &t1);
		if (cqe->res != 0) {
			g_errors++;
		} else if (g_nsamples < MAX_SAMPLES) {
			g_lat_us[g_nsamples++] = ts_diff_us(&s->t0, &t1);
		}
		g_completed++;
		g_inflight--;
		s->busy = 0;
		n++;
	}
	io_uring_cq_advance(ring, n);
}

static int
cmp_double(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static double
pctile(double *sorted, size_t n, double p)
{
	if (!n)
		return 0;
	return sorted[(size_t)(p * (n - 1))];
}

int
main(int argc, char **argv)
{
	if (argc < 6) {
		fprintf(stderr,
			"usage: %s <dev> store|retrieve <qd> <vsize> <seconds>\n",
			argv[0]);
		return 1;
	}
	const char *dev = argv[1];
	const int is_store = strcmp(argv[2], "store") == 0;
	const int qd = atoi(argv[3]);
	const uint32_t vsize = atoi(argv[4]);
	const int seconds = atoi(argv[5]);

	if (qd < 1 || qd > MAX_QD) {
		fprintf(stderr, "qd must be 1..%d\n", MAX_QD);
		return 1;
	}

	g_lat_us = malloc(MAX_SAMPLES * sizeof(double));
	if (!g_lat_us)
		return 1;

	int dev_fd = open(dev, O_RDWR);
	if (dev_fd < 0) {
		perror("open");
		return 1;
	}
	int nsid = ioctl(dev_fd, NVME_IOCTL_ID);
	if (nsid < 0) {
		perror("NVME_IOCTL_ID");
		return 1;
	}

	struct io_uring ring;
	unsigned entries = 8;
	while (entries < (unsigned)qd * 2 && entries < 4096)
		entries <<= 1;
	struct io_uring_params p = {
		.flags = IORING_SETUP_SQE128 | IORING_SETUP_CQE32,
	};
	if (io_uring_queue_init_params(entries, &ring, &p)) {
		perror("io_uring_queue_init_params");
		return 1;
	}

	// Sparse fixed-buffer table, then pool-backed allocation into it.
	if (io_uring_register_buffers_sparse(&ring, 2)) {
		perror("register_buffers_sparse");
		return 1;
	}
	if (pool_buf_register(&ring, dev_fd, SLOT_STORE_SRC,
			      IORING_BUF_ALLOC_WRITE, vsize)) {
		fprintf(stderr,
			"pool buffer registration failed (%s) -- does the "
			"queue have a pool? (blk_iobuf.iobuf_pool_force_order)\n",
			strerror(errno));
		return 1;
	}
	if (pool_buf_register(&ring, dev_fd, SLOT_RETR_DST,
			      IORING_BUF_ALLOC_READ, vsize)) {
		fprintf(stderr, "pool read-buffer registration failed (%s)\n",
			strerror(errno));
		return 1;
	}

	// Pre-populate all keys (required for retrieve; harmless for store).
	for (unsigned i = 0; i < NKEYS; i++) {
		if (submit_one(&ring, dev_fd, 1, nsid, 0, vsize, i))
			return 1;
		io_uring_submit(&ring);
		while (g_slots[0].busy)
			reap(&ring, 1);
	}
	if (g_errors) {
		fprintf(stderr, "pre-population errors: %lu\n", g_errors);
		return 1;
	}
	g_completed = 0;
	g_nsamples = 0;

	struct timespec start, now;
	struct rusage ru0, ru1;
	getrusage(RUSAGE_SELF, &ru0);
	clock_gettime(CLOCK_MONOTONIC, &start);
	const double deadline_us = seconds * 1e6;
	unsigned key_seq = 0;

	for (;;) {
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (ts_diff_us(&start, &now) >= deadline_us)
			break;
		int queued = 0;
		for (int i = 0; i < qd; i++) {
			if (!g_slots[i].busy) {
				if (submit_one(&ring, dev_fd, is_store, nsid,
					       i, vsize, key_seq++))
					break;
				queued++;
			}
		}
		if (queued)
			io_uring_submit(&ring);
		reap(&ring, g_inflight >= (uint64_t)qd);
	}
	while (g_inflight)
		reap(&ring, 1);
	clock_gettime(CLOCK_MONOTONIC, &now);
	getrusage(RUSAGE_SELF, &ru1);

	const double wall_s = ts_diff_us(&start, &now) / 1e6;
	const double cpu_s = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) +
		(ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1e6 +
		(ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) +
		(ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e6;

	qsort(g_lat_us, g_nsamples, sizeof(double), cmp_double);
	printf("kv_uring_fixed,%s,%s,qd=%d,vsize=%u,secs=%.2f,ios=%lu,errs=%lu,"
	       "iops=%.0f,MBps=%.1f,p50=%.1f,p95=%.1f,p99=%.1f,p999=%.1f,cpu_pct=%.1f\n",
	       dev, is_store ? "store" : "retrieve", qd, vsize, wall_s,
	       g_completed, g_errors, g_completed / wall_s,
	       g_completed / wall_s * vsize / 1048576.0,
	       pctile(g_lat_us, g_nsamples, 0.50),
	       pctile(g_lat_us, g_nsamples, 0.95),
	       pctile(g_lat_us, g_nsamples, 0.99),
	       pctile(g_lat_us, g_nsamples, 0.999),
	       100.0 * cpu_s / wall_s);

	io_uring_queue_exit(&ring);
	close(dev_fd);
	return g_errors ? 1 : 0;
}
