/**
 * AMEX Credit-Default Streaming Benchmark - C++ Samplers
 *
 * High-performance entity window sampling for streaming inference.
 * Implements three policies: random, page_aware, fim_importance.
 *
 * Key features:
 * - Ring-buffer state for per-entity windows
 * - Contiguous tensor output ready for GPU
 * - OpenMP parallelization for batch assembly
 * - Stable latency (no Python GIL overhead)
 */

#include <torch/extension.h>
#include <omp.h>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <unordered_map>
#include <chrono>


/**
 * EntityWindowManager - Per-entity rolling window state
 *
 * Manages a ring buffer of feature vectors for each entity (customer_id).
 * Thread-safe updates and efficient batch retrieval.
 */
class EntityWindowManager {
public:
    EntityWindowManager(int64_t num_entities, int64_t window_size, int64_t feature_dim)
        : num_entities_(num_entities),
          window_size_(window_size),
          feature_dim_(feature_dim) {

        // Allocate contiguous storage for all entity windows
        // Shape: [num_entities, window_size, feature_dim]
        windows_ = torch::zeros({num_entities, window_size, feature_dim},
                                torch::TensorOptions().dtype(torch::kFloat32));

        // Track how many events each entity has seen
        counts_ = std::vector<int64_t>(num_entities, 0);

        // Ring buffer write positions
        positions_ = std::vector<int64_t>(num_entities, 0);
    }

    /**
     * Update entity state with new event features.
     * Returns the entity index for latency tracking.
     */
    int64_t ingest_event(int64_t entity_idx, torch::Tensor features) {
        TORCH_CHECK(entity_idx >= 0 && entity_idx < num_entities_,
                    "Entity index out of bounds");
        TORCH_CHECK(features.size(0) == feature_dim_,
                    "Feature dimension mismatch");

        // Get write position in ring buffer
        int64_t pos = positions_[entity_idx];

        // Copy features into window
        auto window_accessor = windows_.accessor<float, 3>();
        auto feat_accessor = features.accessor<float, 1>();

        for (int64_t f = 0; f < feature_dim_; f++) {
            window_accessor[entity_idx][pos][f] = feat_accessor[f];
        }

        // Update ring buffer position
        positions_[entity_idx] = (pos + 1) % window_size_;
        counts_[entity_idx]++;

        return entity_idx;
    }

    /**
     * Get feature window for a single entity.
     * Returns tensor of shape [window_size, feature_dim].
     */
    torch::Tensor get_entity_window(int64_t entity_idx) {
        TORCH_CHECK(entity_idx >= 0 && entity_idx < num_entities_,
                    "Entity index out of bounds");

        // Return a view of the entity's window
        return windows_.index({entity_idx});
    }

    /**
     * Get feature windows for a batch of entities.
     * Returns contiguous tensor of shape [batch_size, window_size, feature_dim].
     */
    torch::Tensor get_batch_windows(torch::Tensor entity_indices) {
        int64_t batch_size = entity_indices.size(0);
        auto indices_accessor = entity_indices.accessor<int64_t, 1>();

        // Allocate output tensor
        auto batch = torch::empty({batch_size, window_size_, feature_dim_},
                                  torch::TensorOptions().dtype(torch::kFloat32));

        // Parallel copy
        #pragma omp parallel for
        for (int64_t i = 0; i < batch_size; i++) {
            int64_t idx = indices_accessor[i];
            batch.index({i}).copy_(windows_.index({idx}));
        }

        return batch;
    }

    int64_t num_entities() const { return num_entities_; }
    int64_t window_size() const { return window_size_; }
    int64_t feature_dim() const { return feature_dim_; }
    int64_t get_event_count(int64_t entity_idx) const { return counts_[entity_idx]; }

private:
    int64_t num_entities_;
    int64_t window_size_;
    int64_t feature_dim_;
    torch::Tensor windows_;
    std::vector<int64_t> counts_;
    std::vector<int64_t> positions_;
};


/**
 * Random Sampler - Worst-case I/O baseline
 *
 * Randomly samples entities without locality consideration.
 */
class RandomSampler {
public:
    RandomSampler(int64_t num_entities, int64_t batch_size, int64_t seed = -1)
        : num_entities_(num_entities),
          batch_size_(batch_size),
          current_pos_(0) {

        // Initialize entity order
        entity_order_.resize(num_entities);
        std::iota(entity_order_.begin(), entity_order_.end(), 0);

        // Initialize RNG
        if (seed >= 0) {
            rng_.seed(static_cast<uint64_t>(seed));
        } else {
            rng_.seed(std::random_device{}());
        }

        // Initial shuffle
        shuffle();
    }

    void shuffle() {
        std::shuffle(entity_order_.begin(), entity_order_.end(), rng_);
        current_pos_ = 0;
    }

    /**
     * Get next batch of entity indices.
     * Returns tensor of shape [batch_size].
     */
    torch::Tensor next_batch() {
        if (current_pos_ >= num_entities_) {
            shuffle();
        }

        int64_t actual_batch = std::min(batch_size_, num_entities_ - current_pos_);

        auto batch = torch::empty({actual_batch},
                                  torch::TensorOptions().dtype(torch::kInt64));
        auto batch_ptr = batch.data_ptr<int64_t>();

        for (int64_t i = 0; i < actual_batch; i++) {
            batch_ptr[i] = entity_order_[current_pos_ + i];
        }

        current_pos_ += actual_batch;
        return batch;
    }

    bool epoch_complete() const { return current_pos_ >= num_entities_; }

private:
    int64_t num_entities_;
    int64_t batch_size_;
    int64_t current_pos_;
    std::vector<int64_t> entity_order_;
    std::mt19937_64 rng_;
};


/**
 * Page-Aware Sampler - Maximize cache reuse
 *
 * Batches entities by their storage order (first-seen position).
 * Equivalent to page-aware batching from DGraphFin.
 */
class PageAwareSampler {
public:
    PageAwareSampler(torch::Tensor entity_first_row_indices, int64_t batch_size)
        : batch_size_(batch_size),
          current_pos_(0) {

        int64_t num_entities = entity_first_row_indices.size(0);
        auto indices_accessor = entity_first_row_indices.accessor<int64_t, 1>();

        // Create (first_row, entity_idx) pairs
        std::vector<std::pair<int64_t, int64_t>> pairs;
        pairs.reserve(num_entities);
        for (int64_t i = 0; i < num_entities; i++) {
            pairs.emplace_back(indices_accessor[i], i);
        }

        // Sort by first_row (storage order)
        std::sort(pairs.begin(), pairs.end());

        // Extract sorted entity order
        entity_order_.resize(num_entities);
        for (size_t i = 0; i < pairs.size(); i++) {
            entity_order_[i] = pairs[i].second;
        }

        num_entities_ = num_entities;
    }

    void reset() {
        current_pos_ = 0;
    }

    torch::Tensor next_batch() {
        if (current_pos_ >= num_entities_) {
            reset();
        }

        int64_t actual_batch = std::min(batch_size_, num_entities_ - current_pos_);

        auto batch = torch::empty({actual_batch},
                                  torch::TensorOptions().dtype(torch::kInt64));
        auto batch_ptr = batch.data_ptr<int64_t>();

        for (int64_t i = 0; i < actual_batch; i++) {
            batch_ptr[i] = entity_order_[current_pos_ + i];
        }

        current_pos_ += actual_batch;
        return batch;
    }

    bool epoch_complete() const { return current_pos_ >= num_entities_; }

private:
    int64_t num_entities_;
    int64_t batch_size_;
    int64_t current_pos_;
    std::vector<int64_t> entity_order_;
};


/**
 * FIM Importance Sampler - Importance-guided entity revisiting
 *
 * Tracks importance scores per entity and prioritizes high-importance
 * entities for revisiting. Does NOT improve convergence speed - used
 * for stability evaluation under drift.
 */
class FIMImportanceSampler {
public:
    FIMImportanceSampler(int64_t num_entities, int64_t batch_size,
                         double ema = 0.99, double budget = 0.2, int64_t seed = -1)
        : num_entities_(num_entities),
          batch_size_(batch_size),
          ema_(ema),
          budget_(budget) {

        // Initialize importance scores to 1.0
        importance_.resize(num_entities, 1.0f);

        // Initialize RNG
        if (seed >= 0) {
            rng_.seed(static_cast<uint64_t>(seed));
        } else {
            rng_.seed(std::random_device{}());
        }

        // Build initial batches
        rebuild_batches();
    }

    /**
     * Update importance scores using gradient/output magnitudes.
     */
    void update_importance(torch::Tensor entity_indices, torch::Tensor magnitudes) {
        auto idx_accessor = entity_indices.accessor<int64_t, 1>();
        auto mag_accessor = magnitudes.accessor<float, 1>();

        int64_t n = entity_indices.size(0);
        for (int64_t i = 0; i < n; i++) {
            int64_t idx = idx_accessor[i];
            float mag = mag_accessor[i];
            importance_[idx] = ema_ * importance_[idx] + (1.0 - ema_) * mag;
        }
    }

    void rebuild_batches() {
        // Split into priority (high importance) and regular
        int64_t n_priority = static_cast<int64_t>(num_entities_ * budget_);

        // Get indices sorted by importance (descending)
        std::vector<int64_t> sorted_indices(num_entities_);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [this](int64_t a, int64_t b) {
                      return importance_[a] > importance_[b];
                  });

        // Split into priority and regular
        priority_entities_.assign(sorted_indices.begin(),
                                  sorted_indices.begin() + n_priority);
        regular_entities_.assign(sorted_indices.begin() + n_priority,
                                 sorted_indices.end());

        // Shuffle within groups
        std::shuffle(priority_entities_.begin(), priority_entities_.end(), rng_);
        std::shuffle(regular_entities_.begin(), regular_entities_.end(), rng_);

        priority_pos_ = 0;
        regular_pos_ = 0;
    }

    torch::Tensor next_batch() {
        // Interleave priority and regular entities
        int64_t priority_per_batch = std::max(1L, batch_size_ / 5);
        int64_t regular_per_batch = batch_size_ - priority_per_batch;

        std::vector<int64_t> batch_indices;
        batch_indices.reserve(batch_size_);

        // Add priority entities
        for (int64_t i = 0; i < priority_per_batch && priority_pos_ < static_cast<int64_t>(priority_entities_.size()); i++) {
            batch_indices.push_back(priority_entities_[priority_pos_++]);
        }

        // Add regular entities
        for (int64_t i = 0; i < regular_per_batch && regular_pos_ < static_cast<int64_t>(regular_entities_.size()); i++) {
            batch_indices.push_back(regular_entities_[regular_pos_++]);
        }

        if (batch_indices.empty()) {
            rebuild_batches();
            return next_batch();
        }

        auto batch = torch::empty({static_cast<int64_t>(batch_indices.size())},
                                  torch::TensorOptions().dtype(torch::kInt64));
        auto batch_ptr = batch.data_ptr<int64_t>();

        for (size_t i = 0; i < batch_indices.size(); i++) {
            batch_ptr[i] = batch_indices[i];
        }

        return batch;
    }

    bool epoch_complete() const {
        return priority_pos_ >= static_cast<int64_t>(priority_entities_.size()) &&
               regular_pos_ >= static_cast<int64_t>(regular_entities_.size());
    }

    torch::Tensor get_importance_scores() {
        auto scores = torch::empty({num_entities_},
                                   torch::TensorOptions().dtype(torch::kFloat32));
        std::copy(importance_.begin(), importance_.end(),
                  scores.data_ptr<float>());
        return scores;
    }

private:
    int64_t num_entities_;
    int64_t batch_size_;
    double ema_;
    double budget_;
    std::vector<float> importance_;
    std::vector<int64_t> priority_entities_;
    std::vector<int64_t> regular_entities_;
    int64_t priority_pos_ = 0;
    int64_t regular_pos_ = 0;
    std::mt19937_64 rng_;
};


/**
 * Latency tracker for streaming inference benchmarks.
 * Records per-event latencies for p50/p95/p99 computation.
 */
class LatencyTracker {
public:
    LatencyTracker(int64_t capacity = 1000000)
        : capacity_(capacity) {
        latencies_.reserve(capacity);
    }

    void record(double latency_ms) {
        if (static_cast<int64_t>(latencies_.size()) < capacity_) {
            latencies_.push_back(latency_ms);
        }
    }

    void reset() {
        latencies_.clear();
    }

    torch::Tensor get_percentiles() {
        if (latencies_.empty()) {
            return torch::zeros({4});
        }

        // Sort for percentile computation
        std::vector<double> sorted = latencies_;
        std::sort(sorted.begin(), sorted.end());

        auto result = torch::empty({4}, torch::TensorOptions().dtype(torch::kFloat64));
        auto ptr = result.data_ptr<double>();

        size_t n = sorted.size();
        ptr[0] = sorted[n * 50 / 100];   // p50
        ptr[1] = sorted[n * 95 / 100];   // p95
        ptr[2] = sorted[n * 99 / 100];   // p99
        ptr[3] = sorted[n - 1];           // max

        return result;
    }

    int64_t count() const { return static_cast<int64_t>(latencies_.size()); }
    double mean() const {
        if (latencies_.empty()) return 0.0;
        return std::accumulate(latencies_.begin(), latencies_.end(), 0.0) / latencies_.size();
    }

private:
    int64_t capacity_;
    std::vector<double> latencies_;
};


// High-resolution timer for latency measurement
inline double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}


// Python bindings
PYBIND11_MODULE(amex_sampler_cpp, m) {
    m.doc() = "AMEX Credit-Default Streaming Benchmark - C++ Samplers";

    // EntityWindowManager
    py::class_<EntityWindowManager>(m, "EntityWindowManager")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("num_entities"),
             py::arg("window_size"),
             py::arg("feature_dim"))
        .def("ingest_event", &EntityWindowManager::ingest_event)
        .def("get_entity_window", &EntityWindowManager::get_entity_window)
        .def("get_batch_windows", &EntityWindowManager::get_batch_windows)
        .def("num_entities", &EntityWindowManager::num_entities)
        .def("window_size", &EntityWindowManager::window_size)
        .def("feature_dim", &EntityWindowManager::feature_dim)
        .def("get_event_count", &EntityWindowManager::get_event_count);

    // RandomSampler
    py::class_<RandomSampler>(m, "RandomSampler")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("num_entities"),
             py::arg("batch_size"),
             py::arg("seed") = -1)
        .def("shuffle", &RandomSampler::shuffle)
        .def("next_batch", &RandomSampler::next_batch)
        .def("epoch_complete", &RandomSampler::epoch_complete);

    // PageAwareSampler
    py::class_<PageAwareSampler>(m, "PageAwareSampler")
        .def(py::init<torch::Tensor, int64_t>(),
             py::arg("entity_first_row_indices"),
             py::arg("batch_size"))
        .def("reset", &PageAwareSampler::reset)
        .def("next_batch", &PageAwareSampler::next_batch)
        .def("epoch_complete", &PageAwareSampler::epoch_complete);

    // FIMImportanceSampler
    py::class_<FIMImportanceSampler>(m, "FIMImportanceSampler")
        .def(py::init<int64_t, int64_t, double, double, int64_t>(),
             py::arg("num_entities"),
             py::arg("batch_size"),
             py::arg("ema") = 0.99,
             py::arg("budget") = 0.2,
             py::arg("seed") = -1)
        .def("update_importance", &FIMImportanceSampler::update_importance)
        .def("rebuild_batches", &FIMImportanceSampler::rebuild_batches)
        .def("next_batch", &FIMImportanceSampler::next_batch)
        .def("epoch_complete", &FIMImportanceSampler::epoch_complete)
        .def("get_importance_scores", &FIMImportanceSampler::get_importance_scores);

    // LatencyTracker
    py::class_<LatencyTracker>(m, "LatencyTracker")
        .def(py::init<int64_t>(),
             py::arg("capacity") = 1000000)
        .def("record", &LatencyTracker::record)
        .def("reset", &LatencyTracker::reset)
        .def("get_percentiles", &LatencyTracker::get_percentiles)
        .def("count", &LatencyTracker::count)
        .def("mean", &LatencyTracker::mean);

    // Utility functions
    m.def("get_time_ms", &get_time_ms, "Get current time in milliseconds");
}
