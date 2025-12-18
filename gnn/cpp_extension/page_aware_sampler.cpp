/**
 * Page-Aware Neighbor Sampler - C++ Extension
 *
 * High-performance implementation of page-aware neighbor sampling for GNN training.
 * Uses OpenMP for parallel execution across frontier nodes.
 */

#include <torch/extension.h>
#include <omp.h>
#include <random>
#include <vector>
#include <algorithm>
#include <cstdint>


/**
 * Page-aware neighbor sampling for a batch of frontier nodes.
 */
std::tuple<torch::Tensor, torch::Tensor> sample_neighbors_page_aware(
    torch::Tensor frontier,
    torch::Tensor intra_indptr,
    torch::Tensor intra_indices,
    torch::Tensor inter_indptr,
    torch::Tensor inter_indices,
    int64_t fanout,
    double intra_ratio,
    int64_t seed,
    int64_t num_threads
) {
    // Input validation
    TORCH_CHECK(frontier.dim() == 1, "frontier must be 1D");
    TORCH_CHECK(intra_indptr.dim() == 1, "intra_indptr must be 1D");

    // Move to CPU and ensure int64
    frontier = frontier.to(torch::kCPU).to(torch::kInt64).contiguous();
    intra_indptr = intra_indptr.to(torch::kCPU).to(torch::kInt64).contiguous();
    intra_indices = intra_indices.to(torch::kCPU).to(torch::kInt64).contiguous();
    inter_indptr = inter_indptr.to(torch::kCPU).to(torch::kInt64).contiguous();
    inter_indices = inter_indices.to(torch::kCPU).to(torch::kInt64).contiguous();

    int64_t N = frontier.size(0);
    int64_t intra_budget = static_cast<int64_t>(fanout * intra_ratio);
    int64_t inter_budget = fanout - intra_budget;

    // Set number of threads
    int actual_threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    omp_set_num_threads(actual_threads);

    // Thread-local storage for results
    std::vector<std::vector<int64_t>> thread_neighbors(actual_threads);
    std::vector<std::vector<int64_t>> thread_sources(actual_threads);

    // Pre-allocate thread-local buffers
    int64_t estimated_per_thread = (N * fanout / actual_threads) + fanout;
    for (int t = 0; t < actual_threads; t++) {
        thread_neighbors[t].reserve(estimated_per_thread);
        thread_sources[t].reserve(estimated_per_thread);
    }

    // Initialize thread-local RNGs
    std::vector<std::mt19937_64> thread_rngs(actual_threads);
    for (int t = 0; t < actual_threads; t++) {
        uint64_t thread_seed = (seed >= 0) ? static_cast<uint64_t>(seed + t * 12345) : std::random_device{}();
        thread_rngs[t].seed(thread_seed);
    }

    // Get raw pointers
    auto frontier_ptr = frontier.data_ptr<int64_t>();
    auto intra_indptr_ptr = intra_indptr.data_ptr<int64_t>();
    auto intra_indices_ptr = intra_indices.data_ptr<int64_t>();
    auto inter_indptr_ptr = inter_indptr.data_ptr<int64_t>();
    auto inter_indices_ptr = inter_indices.data_ptr<int64_t>();

    // Parallel sampling loop
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_neighbors = thread_neighbors[tid];
        auto& local_sources = thread_sources[tid];
        auto& rng = thread_rngs[tid];

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N; i++) {
            int64_t node = frontier_ptr[i];

            // Get CSR ranges for this node
            int64_t intra_start = intra_indptr_ptr[node];
            int64_t intra_end = intra_indptr_ptr[node + 1];
            int64_t inter_start = inter_indptr_ptr[node];
            int64_t inter_end = inter_indptr_ptr[node + 1];

            int64_t n_intra_avail = intra_end - intra_start;
            int64_t n_inter_avail = inter_end - inter_start;

            // Compute actual samples
            int64_t n_intra = std::min(intra_budget, n_intra_avail);
            int64_t actual_inter = inter_budget + (intra_budget - n_intra);
            int64_t n_inter = std::min(actual_inter, n_inter_avail);

            // Sample intra-page neighbors
            if (n_intra > 0) {
                if (n_intra < n_intra_avail) {
                    // Fisher-Yates partial shuffle
                    std::vector<int64_t> pool(intra_indices_ptr + intra_start,
                                               intra_indices_ptr + intra_end);
                    for (int64_t j = 0; j < n_intra; j++) {
                        std::uniform_int_distribution<int64_t> dist(j, n_intra_avail - 1);
                        int64_t swap_idx = dist(rng);
                        std::swap(pool[j], pool[swap_idx]);
                        local_neighbors.push_back(pool[j]);
                        local_sources.push_back(i);
                    }
                } else {
                    for (int64_t j = intra_start; j < intra_end; j++) {
                        local_neighbors.push_back(intra_indices_ptr[j]);
                        local_sources.push_back(i);
                    }
                }
            }

            // Sample inter-page neighbors
            if (n_inter > 0) {
                if (n_inter < n_inter_avail) {
                    std::vector<int64_t> pool(inter_indices_ptr + inter_start,
                                               inter_indices_ptr + inter_end);
                    for (int64_t j = 0; j < n_inter; j++) {
                        std::uniform_int_distribution<int64_t> dist(j, n_inter_avail - 1);
                        int64_t swap_idx = dist(rng);
                        std::swap(pool[j], pool[swap_idx]);
                        local_neighbors.push_back(pool[j]);
                        local_sources.push_back(i);
                    }
                } else {
                    for (int64_t j = inter_start; j < inter_end; j++) {
                        local_neighbors.push_back(inter_indices_ptr[j]);
                        local_sources.push_back(i);
                    }
                }
            }
        }
    }

    // Merge thread-local results
    int64_t total = 0;
    for (int t = 0; t < actual_threads; t++) {
        total += static_cast<int64_t>(thread_neighbors[t].size());
    }

    // Create output tensors
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    auto sampled_neighbors = torch::empty({total}, options);
    auto source_indices = torch::empty({total}, options);

    auto neighbors_ptr = sampled_neighbors.data_ptr<int64_t>();
    auto sources_ptr = source_indices.data_ptr<int64_t>();

    // Copy results
    int64_t offset = 0;
    for (int t = 0; t < actual_threads; t++) {
        std::copy(thread_neighbors[t].begin(), thread_neighbors[t].end(),
                  neighbors_ptr + offset);
        std::copy(thread_sources[t].begin(), thread_sources[t].end(),
                  sources_ptr + offset);
        offset += static_cast<int64_t>(thread_neighbors[t].size());
    }

    return std::make_tuple(sampled_neighbors, source_indices);
}


/**
 * Simple CSR neighbor sampling (no page-awareness) for benchmarking.
 */
std::tuple<torch::Tensor, torch::Tensor> sample_neighbors_simple(
    torch::Tensor frontier,
    torch::Tensor indptr,
    torch::Tensor indices,
    int64_t fanout,
    int64_t seed,
    int64_t num_threads
) {
    frontier = frontier.to(torch::kCPU).to(torch::kInt64).contiguous();
    indptr = indptr.to(torch::kCPU).to(torch::kInt64).contiguous();
    indices = indices.to(torch::kCPU).to(torch::kInt64).contiguous();

    int64_t N = frontier.size(0);

    int actual_threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    omp_set_num_threads(actual_threads);

    std::vector<std::vector<int64_t>> thread_neighbors(actual_threads);
    std::vector<std::vector<int64_t>> thread_sources(actual_threads);

    int64_t estimated_per_thread = (N * fanout / actual_threads) + fanout;
    for (int t = 0; t < actual_threads; t++) {
        thread_neighbors[t].reserve(estimated_per_thread);
        thread_sources[t].reserve(estimated_per_thread);
    }

    std::vector<std::mt19937_64> thread_rngs(actual_threads);
    for (int t = 0; t < actual_threads; t++) {
        uint64_t thread_seed = (seed >= 0) ? static_cast<uint64_t>(seed + t * 12345) : std::random_device{}();
        thread_rngs[t].seed(thread_seed);
    }

    auto frontier_ptr = frontier.data_ptr<int64_t>();
    auto indptr_ptr = indptr.data_ptr<int64_t>();
    auto indices_ptr = indices.data_ptr<int64_t>();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_neighbors = thread_neighbors[tid];
        auto& local_sources = thread_sources[tid];
        auto& rng = thread_rngs[tid];

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N; i++) {
            int64_t node = frontier_ptr[i];
            int64_t start = indptr_ptr[node];
            int64_t end = indptr_ptr[node + 1];
            int64_t degree = end - start;

            if (degree == 0) continue;

            int64_t n_sample = std::min(fanout, degree);

            if (n_sample < degree) {
                std::vector<int64_t> pool(indices_ptr + start, indices_ptr + end);
                for (int64_t j = 0; j < n_sample; j++) {
                    std::uniform_int_distribution<int64_t> dist(j, degree - 1);
                    int64_t swap_idx = dist(rng);
                    std::swap(pool[j], pool[swap_idx]);
                    local_neighbors.push_back(pool[j]);
                    local_sources.push_back(i);
                }
            } else {
                for (int64_t j = start; j < end; j++) {
                    local_neighbors.push_back(indices_ptr[j]);
                    local_sources.push_back(i);
                }
            }
        }
    }

    int64_t total = 0;
    for (int t = 0; t < actual_threads; t++) {
        total += static_cast<int64_t>(thread_neighbors[t].size());
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64);
    auto sampled_neighbors = torch::empty({total}, options);
    auto source_indices = torch::empty({total}, options);

    auto neighbors_ptr = sampled_neighbors.data_ptr<int64_t>();
    auto sources_ptr = source_indices.data_ptr<int64_t>();

    int64_t offset = 0;
    for (int t = 0; t < actual_threads; t++) {
        std::copy(thread_neighbors[t].begin(), thread_neighbors[t].end(),
                  neighbors_ptr + offset);
        std::copy(thread_sources[t].begin(), thread_sources[t].end(),
                  sources_ptr + offset);
        offset += static_cast<int64_t>(thread_neighbors[t].size());
    }

    return std::make_tuple(sampled_neighbors, source_indices);
}


// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_neighbors_page_aware", &sample_neighbors_page_aware,
          "Page-aware neighbor sampling with OpenMP parallelization",
          py::arg("frontier"),
          py::arg("intra_indptr"),
          py::arg("intra_indices"),
          py::arg("inter_indptr"),
          py::arg("inter_indices"),
          py::arg("fanout"),
          py::arg("intra_ratio") = 0.7,
          py::arg("seed") = -1,
          py::arg("num_threads") = 0);

    m.def("sample_neighbors_simple", &sample_neighbors_simple,
          "Simple CSR neighbor sampling with OpenMP parallelization",
          py::arg("frontier"),
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("fanout"),
          py::arg("seed") = -1,
          py::arg("num_threads") = 0);
}
