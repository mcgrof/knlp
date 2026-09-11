// SPDX-License-Identifier: MIT
//
// Drive read-only NVMe passthrough through ordinary, registered, or
// blk_iobuf-premapped buffers. The queue and random-LBA rules mirror the
// one-queue-per-device xnvmeperf setup used by KVTide's userspace arms.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <liburing.h>
#include <limits.h>
#include <linux/types.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

struct nvme_uring_cmd {
	__u8 opcode;
	__u8 flags;
	__u16 rsvd1;
	__u32 nsid;
	__u32 cdw2;
	__u32 cdw3;
	__u64 metadata;
	__u64 addr;
	__u32 metadata_len;
	__u32 data_len;
	__u32 cdw10;
	__u32 cdw11;
	__u32 cdw12;
	__u32 cdw13;
	__u32 cdw14;
	__u32 cdw15;
	__u32 timeout_ms;
	__u32 rsvd2;
};

#define NVME_URING_CMD_IO _IOWR('N', 0x80, struct nvme_uring_cmd)
#define NVME_IOCTL_ID _IO('N', 0x40)

#ifndef BLOCK_URING_CMD_ALLOC_IOBUF
#define BLOCK_URING_CMD_ALLOC_IOBUF _IO(0x12, 1)
#endif
#ifndef IORING_URING_CMD_FIXED
#define IORING_URING_CMD_FIXED (1U << 0)
#endif

enum buffer_mode {
	MODE_PLAIN,
	MODE_FIXED,
	MODE_PREMAP,
};

struct options {
	enum buffer_mode mode;
	unsigned int qdepth;
	unsigned int iosize;
	unsigned int lba_size;
	unsigned int seconds;
	uint64_t namespace_bytes;
	int *cpus;
	int ncpus;
	char **devices;
	int ndevices;
};

struct job {
	const struct options *opts;
	const char *device;
	int fd;
	struct io_uring ring;
	struct iovec iov;
	void *buffer;
	uint32_t nsid;
	uint32_t nlb;
	uint64_t nio_blocks;
	unsigned int seed;
	unsigned int outstanding;
	uint64_t completed;
	uint64_t failed;
	bool ring_ready;
	bool buffers_registered;
};

struct start_gate {
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	int ready;
	bool start;
};

struct worker {
	const struct options *opts;
	struct job *jobs;
	int first_job;
	int njobs;
	int cpu;
	int error;
	double elapsed;
	struct start_gate *gate;
};

static void usage(FILE *out, const char *program)
{
	fprintf(out,
		"usage: %s --mode plain|fixed|premap --qd N --iosize B "
		"--lba-size B --namespace-bytes B --seconds N --cpus C[,C] "
		"/dev/ngXnY [...]\n",
		program);
}

static int parse_u64(const char *text, uint64_t max, uint64_t *value)
{
	char *end = NULL;
	unsigned long long parsed;

	errno = 0;
	parsed = strtoull(text, &end, 0);
	if (errno || !text[0] || !end || end[0] || !parsed || parsed > max)
		return -EINVAL;
	*value = parsed;
	return 0;
}

static int parse_cpus(const char *text, int **cpus_out, int *count_out)
{
	char *copy = strdup(text);
	char *save = NULL;
	char *token;
	int *cpus = NULL;
	int count = 0;

	if (!copy)
		return -ENOMEM;
	for (token = strtok_r(copy, ",", &save); token;
	     token = strtok_r(NULL, ",", &save)) {
		char *end = NULL;
		unsigned long cpu;
		int *grown;

		errno = 0;
		cpu = strtoul(token, &end, 0);
		if (errno || !token[0] || !end || end[0] || cpu >= CPU_SETSIZE) {
			free(cpus);
			free(copy);
			return -EINVAL;
		}
		grown = realloc(cpus, (size_t)(count + 1) * sizeof(*cpus));
		if (!grown) {
			free(cpus);
			free(copy);
			return -ENOMEM;
		}
		cpus = grown;
		cpus[count++] = (int)cpu;
	}
	free(copy);
	if (!count) {
		free(cpus);
		return -EINVAL;
	}
	*cpus_out = cpus;
	*count_out = count;
	return 0;
}

static int parse_options(int argc, char **argv, struct options *opts)
{
	enum {
		OPT_MODE = 256,
		OPT_QD,
		OPT_IOSIZE,
		OPT_LBA_SIZE,
		OPT_NAMESPACE_BYTES,
		OPT_SECONDS,
		OPT_CPUS,
	};
	static const struct option long_options[] = {
		{"mode", required_argument, NULL, OPT_MODE},
		{"qd", required_argument, NULL, OPT_QD},
		{"iosize", required_argument, NULL, OPT_IOSIZE},
		{"lba-size", required_argument, NULL, OPT_LBA_SIZE},
		{"namespace-bytes", required_argument, NULL, OPT_NAMESPACE_BYTES},
		{"seconds", required_argument, NULL, OPT_SECONDS},
		{"cpus", required_argument, NULL, OPT_CPUS},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	bool have_mode = false;
	int option;

	while ((option = getopt_long(argc, argv, "h", long_options, NULL)) != -1) {
		uint64_t value;
		int error = 0;

		switch (option) {
		case OPT_MODE:
			have_mode = true;
			if (!strcmp(optarg, "plain"))
				opts->mode = MODE_PLAIN;
			else if (!strcmp(optarg, "fixed"))
				opts->mode = MODE_FIXED;
			else if (!strcmp(optarg, "premap"))
				opts->mode = MODE_PREMAP;
			else
				error = -EINVAL;
			break;
		case OPT_QD:
			error = parse_u64(optarg, UINT_MAX, &value);
			if (!error)
				opts->qdepth = (unsigned int)value;
			break;
		case OPT_IOSIZE:
			error = parse_u64(optarg, UINT_MAX, &value);
			if (!error)
				opts->iosize = (unsigned int)value;
			break;
		case OPT_LBA_SIZE:
			error = parse_u64(optarg, UINT_MAX, &value);
			if (!error)
				opts->lba_size = (unsigned int)value;
			break;
		case OPT_NAMESPACE_BYTES:
			error = parse_u64(optarg, UINT64_MAX, &opts->namespace_bytes);
			break;
		case OPT_SECONDS:
			error = parse_u64(optarg, UINT_MAX, &value);
			if (!error)
				opts->seconds = (unsigned int)value;
			break;
		case OPT_CPUS:
			error = parse_cpus(optarg, &opts->cpus, &opts->ncpus);
			break;
		case 'h':
			return 1;
		default:
			return -EINVAL;
		}
		if (error)
			return error;
	}

	opts->devices = argv + optind;
	opts->ndevices = argc - optind;
	if (!have_mode || !opts->qdepth || !opts->iosize || !opts->lba_size ||
	    !opts->namespace_bytes || !opts->seconds || !opts->ncpus ||
	    !opts->ndevices)
		return -EINVAL;
	if (opts->ncpus > opts->ndevices || opts->iosize % opts->lba_size ||
	    opts->namespace_bytes < opts->iosize ||
	    opts->iosize / opts->lba_size > 65536U)
		return -EINVAL;
	return 0;
}

static uint64_t monotonic_ns(void)
{
	struct timespec now;

	if (clock_gettime(CLOCK_MONOTONIC, &now))
		return 0;
	return (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

static int alloc_iobuf(struct job *job)
{
	struct io_uring_sqe *sqe = io_uring_get_sqe(&job->ring);
	struct io_uring_cqe *cqe;
	int result;

	if (!sqe)
		return -ENOSPC;
	memset(sqe, 0, 128);
	sqe->opcode = IORING_OP_URING_CMD;
	sqe->fd = job->fd;
	sqe->cmd_op = BLOCK_URING_CMD_ALLOC_IOBUF;
	sqe->addr = 0;
	sqe->addr3 = job->opts->iosize;
	result = io_uring_submit_and_wait(&job->ring, 1);
	if (result < 0)
		return result;
	result = io_uring_wait_cqe(&job->ring, &cqe);
	if (result < 0)
		return result;
	result = cqe->res;
	io_uring_cqe_seen(&job->ring, cqe);
	return result > 0 ? -EPROTO : result;
}

static int setup_job(struct job *job, const struct options *opts,
		     const char *device, unsigned int seed)
{
	struct io_uring_params params = {};
	long page_size = sysconf(_SC_PAGESIZE);
	int result;

	job->opts = opts;
	job->device = device;
	job->seed = seed;
	job->nlb = opts->iosize / opts->lba_size;
	job->nio_blocks = opts->namespace_bytes / opts->iosize;
	job->fd = open(device, O_RDONLY);
	if (job->fd < 0)
		return -errno;
	result = ioctl(job->fd, NVME_IOCTL_ID);
	if (result <= 0)
		return result < 0 ? -errno : -ENODEV;
	job->nsid = (uint32_t)result;
	params.flags = IORING_SETUP_SQE128 | IORING_SETUP_CQE32;
	result = io_uring_queue_init_params(opts->qdepth, &job->ring, &params);
	if (result < 0)
		return result;
	job->ring_ready = true;

	if (opts->mode == MODE_PREMAP) {
		result = io_uring_register_buffers_sparse(&job->ring, 1);
		if (result < 0)
			return result;
		job->buffers_registered = true;
		return alloc_iobuf(job);
	}

	if (page_size <= 0)
		return -EINVAL;
	result = posix_memalign(&job->buffer, (size_t)page_size, opts->iosize);
	if (result)
		return -result;
	memset(job->buffer, 0, opts->iosize);
	if (opts->mode == MODE_FIXED) {
		job->iov.iov_base = job->buffer;
		job->iov.iov_len = opts->iosize;
		result = io_uring_register_buffers(&job->ring, &job->iov, 1);
		if (result < 0)
			return result;
		job->buffers_registered = true;
	}
	return 0;
}

static void cleanup_job(struct job *job)
{
	if (job->buffers_registered)
		io_uring_unregister_buffers(&job->ring);
	if (job->ring_ready)
		io_uring_queue_exit(&job->ring);
	free(job->buffer);
	if (job->fd >= 0)
		close(job->fd);
}

static int prepare_read(struct job *job)
{
	struct io_uring_sqe *sqe = io_uring_get_sqe(&job->ring);
	struct nvme_uring_cmd *command;
	uint64_t slba;

	if (!sqe)
		return -EAGAIN;
	memset(sqe, 0, 128);
	sqe->opcode = IORING_OP_URING_CMD;
	sqe->fd = job->fd;
	sqe->cmd_op = NVME_URING_CMD_IO;
	if (job->opts->mode != MODE_PLAIN) {
		sqe->uring_cmd_flags = IORING_URING_CMD_FIXED;
		sqe->buf_index = 0;
	}
	command = (struct nvme_uring_cmd *)sqe->cmd;
	command->opcode = 0x02;
	command->nsid = job->nsid;
	command->addr = job->opts->mode == MODE_PREMAP ?
		0 : (uint64_t)(uintptr_t)job->buffer;
	command->data_len = job->opts->iosize;
	slba = ((uint64_t)rand_r(&job->seed) % job->nio_blocks) * job->nlb;
	command->cdw10 = (uint32_t)slba;
	command->cdw11 = (uint32_t)(slba >> 32);
	command->cdw12 = job->nlb - 1U;
	job->outstanding++;
	return 0;
}

static int fill_queue(struct job *job)
{
	int result;

	while (job->outstanding < job->opts->qdepth) {
		result = prepare_read(job);
		if (result)
			break;
	}
	result = io_uring_submit(&job->ring);
	return result < 0 ? result : 0;
}

static int reap_job(struct job *job, bool wait)
{
	struct io_uring_cqe *cqe;
	int count = 0;
	int result;

	for (;;) {
		result = wait && !count ? io_uring_wait_cqe(&job->ring, &cqe) :
			io_uring_peek_cqe(&job->ring, &cqe);
		if (result == -EAGAIN)
			return count;
		if (result < 0)
			return result;
		if (cqe->res)
			job->failed++;
		else
			job->completed++;
		job->outstanding--;
		io_uring_cqe_seen(&job->ring, cqe);
		count++;
	}
}

static int pin_to_cpu(int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void gate_wait(struct start_gate *gate)
{
	pthread_mutex_lock(&gate->mutex);
	gate->ready++;
	pthread_cond_broadcast(&gate->cond);
	while (!gate->start)
		pthread_cond_wait(&gate->cond, &gate->mutex);
	pthread_mutex_unlock(&gate->mutex);
}

static void *run_worker(void *argument)
{
	struct worker *worker = argument;
	uint64_t start, deadline, end;
	int result;

	result = pin_to_cpu(worker->cpu);
	if (result)
		worker->error = -result;
	gate_wait(worker->gate);
	if (worker->error)
		return NULL;
	start = monotonic_ns();
	deadline = start + (uint64_t)worker->opts->seconds * 1000000000ULL;
	for (int i = 0; i < worker->njobs; i++) {
		worker->error = fill_queue(&worker->jobs[worker->first_job + i]);
		if (worker->error)
			return NULL;
	}
	while (monotonic_ns() < deadline) {
		for (int i = 0; i < worker->njobs; i++) {
			struct job *job = &worker->jobs[worker->first_job + i];
			int result = reap_job(job, false);

			if (result < 0) {
				worker->error = result;
				return NULL;
			}
			if (result) {
				worker->error = fill_queue(job);
				if (worker->error)
					return NULL;
			}
		}
	}
	for (int i = 0; i < worker->njobs; i++) {
		struct job *job = &worker->jobs[worker->first_job + i];

		while (job->outstanding) {
			int result = reap_job(job, true);

			if (result < 0) {
				worker->error = result;
				return NULL;
			}
		}
	}
	end = monotonic_ns();
	worker->elapsed = (double)(end - start) / 1000000000.0;
	return NULL;
}

static const char *mode_name(enum buffer_mode mode)
{
	switch (mode) {
	case MODE_PLAIN:
		return "plain";
	case MODE_FIXED:
		return "fixed";
	case MODE_PREMAP:
		return "premap";
	}
	return "unknown";
}

int main(int argc, char **argv)
{
	struct options opts = {};
	struct start_gate gate = {
		.mutex = PTHREAD_MUTEX_INITIALIZER,
		.cond = PTHREAD_COND_INITIALIZER,
	};
	struct job *jobs = NULL;
	struct worker *workers = NULL;
	pthread_t *threads = NULL;
	uint64_t completed = 0, failed = 0;
	double elapsed = 0.0;
	int created = 0;
	int result;

	result = parse_options(argc, argv, &opts);
	if (result > 0) {
		usage(stdout, argv[0]);
		return 0;
	}
	if (result < 0) {
		usage(stderr, argv[0]);
		return 2;
	}
	jobs = calloc((size_t)opts.ndevices, sizeof(*jobs));
	workers = calloc((size_t)opts.ncpus, sizeof(*workers));
	threads = calloc((size_t)opts.ncpus, sizeof(*threads));
	if (!jobs || !workers || !threads) {
		result = -ENOMEM;
		goto out;
	}
	for (int i = 0; i < opts.ndevices; i++)
		jobs[i].fd = -1;
	for (int i = 0; i < opts.ncpus; i++) {
		int base = opts.ndevices / opts.ncpus;
		int extra = opts.ndevices % opts.ncpus;
		int first = i * base + (i < extra ? i : extra);
		int count = base + (i < extra ? 1 : 0);

		workers[i].opts = &opts;
		workers[i].jobs = jobs;
		workers[i].first_job = first;
		workers[i].njobs = count;
		workers[i].cpu = opts.cpus[i];
		workers[i].gate = &gate;
		for (int j = 0; j < count; j++) {
			result = setup_job(&jobs[first + j], &opts,
					   opts.devices[first + j],
					   (unsigned int)(opts.cpus[i] * 1000 + j));
			if (result) {
				fprintf(stderr, "setup %s: %s\n",
					opts.devices[first + j], strerror(-result));
				goto out;
			}
		}
	}
	for (int i = 0; i < opts.ncpus; i++) {
		result = pthread_create(&threads[i], NULL, run_worker, &workers[i]);
		if (result) {
			result = -result;
			goto out;
		}
		created++;
	}
	pthread_mutex_lock(&gate.mutex);
	while (gate.ready != opts.ncpus)
		pthread_cond_wait(&gate.cond, &gate.mutex);
	gate.start = true;
	pthread_cond_broadcast(&gate.cond);
	pthread_mutex_unlock(&gate.mutex);
	for (int i = 0; i < created; i++)
		pthread_join(threads[i], NULL);
	created = 0;
	result = 0;
	for (int i = 0; i < opts.ncpus; i++) {
		if (workers[i].error && !result)
			result = workers[i].error;
		if (workers[i].elapsed > elapsed)
			elapsed = workers[i].elapsed;
	}
	for (int i = 0; i < opts.ndevices; i++) {
		completed += jobs[i].completed;
		failed += jobs[i].failed;
	}
	printf("mode=%s completed=%" PRIu64 " failed=%" PRIu64
	       " elapsed=%.6f iops=%.3f MiBps=%.3f\n",
	       mode_name(opts.mode), completed, failed, elapsed,
	       elapsed ? (double)completed / elapsed : 0.0,
	       elapsed ? ((double)completed * opts.iosize) /
				 (elapsed * 1024.0 * 1024.0) : 0.0);

out:
	if (created) {
		pthread_mutex_lock(&gate.mutex);
		gate.start = true;
		pthread_cond_broadcast(&gate.cond);
		pthread_mutex_unlock(&gate.mutex);
		for (int i = 0; i < created; i++)
			pthread_join(threads[i], NULL);
	}
	if (jobs) {
		for (int i = 0; i < opts.ndevices; i++)
			cleanup_job(&jobs[i]);
	}
	free(threads);
	free(workers);
	free(jobs);
	free(opts.cpus);
	if (result) {
		fprintf(stderr, "uring_nvm_perf: %s\n", strerror(-result));
		return 1;
	}
	return failed ? 1 : 0;
}
