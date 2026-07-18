// SPDX-License-Identifier: Apache-2.0
//
// Minimal sustained-queue-depth KV perf tool over xNVMe io_uring_cmd —
// the xNVMe-side counterpart of spdk_nvme_perf's KV mode for an
// initiator bake-off against the same NVMe-oF KV target.
//
// Single thread, one xNVMe queue, N keys cycled round-robin, fixed value
// size, timed run. Reports IOPS, MB/s, latency percentiles and CPU use.
//
// Build: gcc -O2 xnvme_kv_perf.c $(pkg-config --cflags --libs xnvme) -o xnvme_kv_perf
// Run:   sudo ./xnvme_kv_perf /dev/ng5n1 store 64 4096 10

#include <libxnvme.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define MAX_QD 256
#define NKEYS 1024
#define MAX_SAMPLES (8 * 1024 * 1024)

struct slot {
	int busy;
	int idx; // slot index
	struct timespec t0;
};

static struct slot g_slots[MAX_QD];
static double *g_lat_us;
static size_t g_nsamples;
static uint64_t g_completed, g_errors;
static struct xnvme_queue *g_queue;

static double
ts_diff_us(const struct timespec *a, const struct timespec *b)
{
	return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

static void
cb(struct xnvme_cmd_ctx *ctx, void *arg)
{
	struct slot *s = arg;
	struct timespec t1;

	clock_gettime(CLOCK_MONOTONIC, &t1);
	if (xnvme_cmd_ctx_cpl_status(ctx)) {
		g_errors++;
	} else if (g_nsamples < MAX_SAMPLES) {
		g_lat_us[g_nsamples++] = ts_diff_us(&s->t0, &t1);
	}
	g_completed++;
	s->busy = 0;
	xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
}

static void
make_key(char *buf, unsigned i)
{
	snprintf(buf, 16, "pk%06x", i % NKEYS);
}

static int
submit_one(struct xnvme_dev *dev, uint32_t nsid, int is_store, struct slot *s,
	   char *vbuf, uint32_t vsize, unsigned key_idx)
{
	char key[16];
	int err;

	make_key(key, key_idx);
	for (;;) {
		struct xnvme_cmd_ctx *ctx = xnvme_queue_get_cmd_ctx(g_queue);
		if (!ctx) {
			xnvme_queue_poke(g_queue, 0);
			continue;
		}
		xnvme_cmd_ctx_set_cb(ctx, cb, s);
		clock_gettime(CLOCK_MONOTONIC, &s->t0);
		if (is_store)
			err = xnvme_kvs_store(ctx, nsid, key, strlen(key), vbuf, vsize, 0);
		else
			err = xnvme_kvs_retrieve(ctx, nsid, key, strlen(key), vbuf, vsize, 0);
		if (!err) {
			s->busy = 1;
			return 0;
		}
		xnvme_queue_put_cmd_ctx(g_queue, ctx);
		if (err == -EBUSY || err == -EAGAIN) {
			xnvme_queue_poke(g_queue, 0);
			continue;
		}
		fprintf(stderr, "submit failed: %d\n", err);
		return err;
	}
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
		fprintf(stderr, "usage: %s <uri> store|retrieve <qd> <vsize> <seconds>\n", argv[0]);
		return 1;
	}
	const char *uri = argv[1];
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

	struct xnvme_opts opts = xnvme_opts_default();
	opts.be = "io_uring_cmd";
	opts.admin = "nvme";
	struct xnvme_dev *dev = xnvme_dev_open(uri, &opts);
	if (!dev) {
		perror("xnvme_dev_open");
		return 1;
	}
	uint32_t nsid = xnvme_dev_get_nsid(dev);

	// queue capacity: next pow2 >= 2*qd, max 2048
	uint16_t cap = 1;
	while (cap < qd * 2 && cap < 2048)
		cap <<= 1;
	if (xnvme_queue_init(dev, cap, 0, &g_queue)) {
		fprintf(stderr, "queue_init failed\n");
		return 1;
	}

	char *vbufs[MAX_QD];
	for (int i = 0; i < qd; i++) {
		vbufs[i] = aligned_alloc(4096, vsize);
		memset(vbufs[i], 0x5a + i, vsize);
		g_slots[i].idx = i;
	}

	// Pre-populate all keys (required for retrieve; harmless for store).
	for (unsigned i = 0; i < NKEYS; i++) {
		if (submit_one(dev, nsid, 1, &g_slots[0], vbufs[0], vsize, i))
			return 1;
		while (g_slots[0].busy)
			xnvme_queue_poke(g_queue, 0);
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
		for (int i = 0; i < qd; i++) {
			if (!g_slots[i].busy) {
				if (submit_one(dev, nsid, is_store, &g_slots[i], vbufs[i],
					       vsize, key_seq++))
					return 1;
			}
		}
		xnvme_queue_poke(g_queue, 0);
	}
	// drain
	while (xnvme_queue_get_outstanding(g_queue))
		xnvme_queue_poke(g_queue, 0);
	clock_gettime(CLOCK_MONOTONIC, &now);
	getrusage(RUSAGE_SELF, &ru1);

	const double wall_s = ts_diff_us(&start, &now) / 1e6;
	const double cpu_s = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) +
		(ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1e6 +
		(ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) +
		(ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e6;

	qsort(g_lat_us, g_nsamples, sizeof(double), cmp_double);
	printf("xnvme_kv_perf,%s,%s,qd=%d,vsize=%u,secs=%.2f,ios=%lu,errs=%lu,"
	       "iops=%.0f,MBps=%.1f,p50=%.1f,p95=%.1f,p99=%.1f,p999=%.1f,cpu_pct=%.1f\n",
	       uri, is_store ? "store" : "retrieve", qd, vsize, wall_s, g_completed, g_errors,
	       g_completed / wall_s, g_completed / wall_s * vsize / 1048576.0,
	       pctile(g_lat_us, g_nsamples, 0.50), pctile(g_lat_us, g_nsamples, 0.95),
	       pctile(g_lat_us, g_nsamples, 0.99), pctile(g_lat_us, g_nsamples, 0.999),
	       100.0 * cpu_s / wall_s);

	xnvme_queue_term(g_queue);
	xnvme_dev_close(dev);
	return g_errors ? 1 : 0;
}
