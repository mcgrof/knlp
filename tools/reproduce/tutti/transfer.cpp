// SPDX-License-Identifier: MIT
// Synthetic, single-device Tutti round trips. Every byte is verified.
#include <tutti/tutti_runtime.h>
#include <tutti/storage_runtime.h>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <random>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace tutti;
using Clock = std::chrono::steady_clock;

[[noreturn]] static void fail(const std::string& message) {
    // Do not run caller-owned buffer/target destructors on an unknown I/O
    // outcome. A failed run is not authorization to reuse its files/device.
    std::fprintf(stderr, "FAIL: %s\nNo PASS receipt; retain files and establish "
                         "device quiescence before retrying.\n", message.c_str());
    std::fflush(stderr);
    std::_Exit(2);
}

static void check(bool ok, const std::string& message) {
    if (!ok) fail(message);
}
static void gpu(cudaError_t code) {
    check(code == cudaSuccess, cudaGetErrorString(code));
}
static void status(const Status& s) { check(s.ok(), s.message()); }
static double cpu_seconds() {
    rusage r{};
    check(getrusage(RUSAGE_SELF, &r) == 0, "getrusage");
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec +
           (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6;
}
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static uint64_t number(const char* value) {
    check(value[0] >= '0' && value[0] <= '9', "invalid unsigned integer");
    char* end = nullptr;
    errno = 0;
    auto n = std::strtoull(value, &end, 10);
    check(!errno && end && !*end, "invalid integer");
    return n;
}
static uint64_t pattern(uint64_t word, uint64_t seed) {
    uint64_t x = word + seed + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Measurement {
    double wall = 0, cpu = 0;
    std::vector<double> batch_us;
};

static Measurement transfer(StorageRuntime* rt, MemoryHandle memory,
                            TargetHandle target, const HostSubmitContext& context,
                            const std::vector<uint64_t>& order, uint64_t bytes,
                            size_t depth, IoDirection direction) {
    Measurement m;
    m.batch_us.reserve((order.size() + depth - 1) / depth);
    std::vector<IoRequest> requests;
    requests.reserve(depth);
    gpu(cudaStreamSynchronize(context.stream));
    const double cpu_start = cpu_seconds();
    const auto start = Clock::now();
    for (size_t base = 0; base < order.size(); base += depth) {
        requests.clear();
        const size_t count = std::min(depth, order.size() - base);
        for (size_t i = 0; i < count; ++i) {
            const uint64_t offset = order[base + i] * bytes;
            requests.push_back({direction, memory, offset, target, offset, bytes});
        }
        const auto batch_start = Clock::now();
        auto submitted = rt->submit(requests.data(), requests.size(), context);
        // Drain accepted work even if submission rejected another entry.
        if (submitted.io) {
            auto done = rt->wait(*submitted.io, 60000);
            check(done.observation_status.ok() && done.result.has_value(),
                  "I/O outcome unknown or wait failed");
            check(done.result->state == IoState::COMPLETED &&
                  done.result->status.ok() && !done.result->detail.timeout_seen &&
                  done.result->detail.failure_kind == IoFailureKind::NONE &&
                  done.result->detail.confirmed_bytes == count * bytes,
                  "I/O failed or confirmed-byte count differs");
            status(rt->release_io(*submitted.io));
        }
        check(submitted.io.has_value() && submitted.status.ok() &&
              submitted.initial_states.size() == count, "batch was not fully accepted");
        for (const auto& item : submitted.initial_states)
            check(item.state == IoRequestState::ACCEPTED && item.status.ok(),
                  "request rejected; lower application batch depth");
        m.batch_us.push_back(seconds(batch_start) * 1e6);
    }
    gpu(cudaStreamSynchronize(context.stream));
    m.wall = seconds(start);
    m.cpu = cpu_seconds() - cpu_start;
    check(m.wall > 0 && std::isfinite(m.wall) && m.cpu >= 0, "invalid timing");
    return m;
}

static void emit(const Measurement& m, const char* operation, uint64_t span,
                 uint64_t bytes, uint64_t depth, uint64_t repeat, uint64_t seed,
                 bool random) {
    auto samples = m.batch_us;
    std::sort(samples.begin(), samples.end());
    auto percentile = [&](double p) {
        return samples[static_cast<size_t>(std::ceil(p * samples.size())) - 1];
    };
    // Emit only after full write/read verification and normal cleanup.
    std::printf("KNLP_TUTTI_JSON {\"schema_version\":1,\"backend\":\"tutti\","
                "\"workload\":\"synthetic-transfer\",\"operation\":\"%s\","
                "\"scheduling\":\"submit-and-drain\",\"access\":\"%s\","
                "\"io_bytes\":%llu,\"span_bytes\":%llu,\"batch_depth\":%llu,"
                "\"repeat\":%llu,\"seed\":%llu,\"completed_bytes\":%llu,"
                "\"operations\":%llu,\"wall_seconds\":%.9f,"
                "\"process_cpu_seconds\":%.9f,\"batch_count\":%zu,"
                "\"batch_p50_us\":%.6f,\"batch_p99_us\":%.6f,"
                "\"verified_bytes\":%llu,\"correctness\":\"full-readback\"}\n",
                operation, random ? "random-permutation" : "sequential",
                (unsigned long long)bytes, (unsigned long long)span,
                (unsigned long long)depth, (unsigned long long)repeat,
                (unsigned long long)seed, (unsigned long long)span,
                (unsigned long long)(span / bytes), m.wall, m.cpu, samples.size(),
                percentile(0.5), percentile(0.99), (unsigned long long)span);
}

int main(int argc, char** argv) {
    std::string config, file;
    uint64_t bytes = 65536, span = 16ULL << 20, depth = 8;
    uint64_t warmups = 0, repeats = 1, seed = 42;
    bool random = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            std::puts("knlp_tutti_io --config YAML --file NEW_FILE [--io-bytes N] "
                      "[--span-bytes N] [--batch-depth N] [--warmups N] "
                      "[--repeats N] [--seed N] [--random]");
            return 0;
        }
        if (option == "--random") { random = true; continue; }
        check(i + 1 < argc, "missing argument for " + option);
        const char* value = argv[++i];
        if (option == "--config") config = value;
        else if (option == "--file") file = value;
        else if (option == "--io-bytes") bytes = number(value);
        else if (option == "--span-bytes") span = number(value);
        else if (option == "--batch-depth") depth = number(value);
        else if (option == "--warmups") warmups = number(value);
        else if (option == "--repeats") repeats = number(value);
        else if (option == "--seed") seed = number(value);
        else fail("unknown option " + option);
    }
    check(!config.empty() && !file.empty(), "config and new file are required");
    check(bytes && bytes % 4096 == 0 && span >= bytes && span % bytes == 0 &&
          span <= (1ULL << 40) && depth && depth <= span / bytes &&
          depth <= 4096 && repeats && repeats <= 1000 && warmups <= 100,
          "invalid size/depth/repetition geometry");
    check(std::string(TUTTI_COMPILED_ACCELERATOR_PROFILE) == "CUDA",
          "this benchmark requires the real CUDA profile");
    auto created = TuttiRuntime::create(config);
    check(created.ok(), created.status().message());
    auto owner = std::move(created).value();
    auto* rt = owner->storage_runtime();
    check(rt && rt->accel_id() >= 0, "no accelerator runtime");
    gpu(cudaSetDevice(rt->accel_id()));

    // Materialize extents before resolving them. Exclusive creation prevents
    // overwriting an existing file even when invoked outside the Python runner.
    int fd = open(file.c_str(), O_RDWR | O_CREAT | O_EXCL | O_DIRECT | O_NOFOLLOW, 0600);
    check(fd >= 0, "exclusive backing-file create failed");
    check(posix_fallocate(fd, 0, span) == 0, "preallocation failed");
    void* zeros = nullptr;
    check(posix_memalign(&zeros, 4096, 1 << 20) == 0, "host initialization allocation");
    std::memset(zeros, 0, 1 << 20);
    for (uint64_t off = 0; off < span;) {
        const size_t n = std::min<uint64_t>(1 << 20, span - off);
        check(pwrite(fd, zeros, n, off) == static_cast<ssize_t>(n), "initialization write");
        off += n;
    }
    std::free(zeros);
    check(fsync(fd) == 0, "initialization fsync");
    check(close(fd) == 0, "initialization close");

    auto target_result = rt->open("file://" + file, OpenOptions{"file"});
    check(target_result.ok(), target_result.status().message());
    auto target = target_result.value();
    void* raw = nullptr;
    gpu(cudaMalloc(&raw, span + 65536));
    void* buffer = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw) + 65535) &
                                          ~uintptr_t(65535));
    auto registered = rt->register_memory({buffer, span, MemoryKind::DEVICE,
        MemoryOwnership::CALLER_OWNED, rt->accel_id(), "CUDA", bytes});
    check(registered.ok(), registered.status().message());
    auto memory = registered.value();
    cudaStream_t stream;
    gpu(cudaStreamCreate(&stream));
    HostSubmitContext context{ExecutionDomain::DEVICE_EXECUTION, rt->accel_id(), stream};
    std::vector<uint64_t> expected(span / 8), actual(span / 8), order(span / bytes);
    std::iota(order.begin(), order.end(), 0);
    if (random) {
        std::mt19937_64 rng(seed);
        std::shuffle(order.begin(), order.end(), rng);
    }
    std::vector<std::pair<Measurement, Measurement>> results;
    for (uint64_t rep = 0; rep < warmups + repeats; ++rep) {
        for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = pattern(i, seed + rep);
        gpu(cudaMemcpy(buffer, expected.data(), span, cudaMemcpyHostToDevice));
        auto write = transfer(rt, memory, target, context, order, bytes, depth, IoDirection::WRITE);
        gpu(cudaMemset(buffer, 0xA5, span));
        auto read = transfer(rt, memory, target, context, order, bytes, depth, IoDirection::READ);
        gpu(cudaMemcpy(actual.data(), buffer, span, cudaMemcpyDeviceToHost));
        check(actual == expected, "payload mismatch in full round-trip readback");
        if (rep >= warmups) results.emplace_back(std::move(write), std::move(read));
    }
    status(rt->unregister_memory(memory));
    status(rt->close(target));
    status(owner->shutdown());
    gpu(cudaStreamDestroy(stream));
    gpu(cudaFree(raw));
    check(unlink(file.c_str()) == 0, "backing-file cleanup failed");
    for (size_t i = 0; i < results.size(); ++i) {
        emit(results[i].first, "write", span, bytes, depth, i, seed, random);
        emit(results[i].second, "read", span, bytes, depth, i, seed, random);
    }
    return 0;
}
