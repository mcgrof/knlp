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
#include <memory>
#include <unordered_map>
#include <unordered_set>
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


/**
 * RelatedEntityLookup - Cross-entity retrieval for locality stress testing
 *
 * Simulates production patterns where each event requires fetching related
 * entities' states (e.g., for risk aggregation, pattern detection).
 *
 * Uses a simple hash-based mapping to assign k related entities per entity.
 * This exercises data locality without requiring actual relationship data.
 */
class RelatedEntityLookup {
public:
    RelatedEntityLookup(int64_t num_entities, int64_t k_related, int64_t seed = 42)
        : num_entities_(num_entities),
          k_related_(k_related) {

        if (k_related <= 0) {
            return;  // No related entities needed
        }

        // Pre-compute related entities for each entity using hash-based mapping
        // This creates stable, reproducible relationships
        related_.resize(num_entities);
        std::mt19937_64 rng(seed);

        for (int64_t i = 0; i < num_entities; i++) {
            related_[i].reserve(k_related);

            // Generate k unique related entities (excluding self)
            std::unordered_set<int64_t> seen;
            seen.insert(i);  // Exclude self

            while (static_cast<int64_t>(related_[i].size()) < k_related) {
                // Hash-based selection with RNG fallback
                int64_t candidate = rng() % num_entities;
                if (seen.find(candidate) == seen.end()) {
                    related_[i].push_back(candidate);
                    seen.insert(candidate);
                }
            }
        }
    }

    /**
     * Get related entity indices for a single entity.
     * Returns tensor of shape [k_related].
     */
    torch::Tensor get_related(int64_t entity_idx) {
        if (k_related_ <= 0 || entity_idx < 0 || entity_idx >= num_entities_) {
            return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
        }

        auto result = torch::empty({k_related_},
                                   torch::TensorOptions().dtype(torch::kInt64));
        auto ptr = result.data_ptr<int64_t>();

        for (int64_t i = 0; i < k_related_; i++) {
            ptr[i] = related_[entity_idx][i];
        }

        return result;
    }

    /**
     * Get related entity indices for a batch of entities.
     * Returns tensor of shape [batch_size, k_related].
     */
    torch::Tensor get_batch_related(torch::Tensor entity_indices) {
        entity_indices = entity_indices.to(torch::kCPU).to(torch::kInt64).contiguous();
        int64_t batch_size = entity_indices.size(0);

        if (k_related_ <= 0) {
            return torch::empty({batch_size, 0},
                               torch::TensorOptions().dtype(torch::kInt64));
        }

        auto idx_ptr = entity_indices.data_ptr<int64_t>();
        auto result = torch::empty({batch_size, k_related_},
                                   torch::TensorOptions().dtype(torch::kInt64));
        auto result_ptr = result.data_ptr<int64_t>();

        #pragma omp parallel for
        for (int64_t i = 0; i < batch_size; i++) {
            int64_t entity_idx = idx_ptr[i];
            if (entity_idx >= 0 && entity_idx < num_entities_) {
                for (int64_t j = 0; j < k_related_; j++) {
                    result_ptr[i * k_related_ + j] = related_[entity_idx][j];
                }
            }
        }

        return result;
    }

    /**
     * Get all unique related entities for a batch (for cache tracking).
     * Returns flattened tensor of unique entity indices.
     */
    torch::Tensor get_unique_related(torch::Tensor entity_indices) {
        entity_indices = entity_indices.to(torch::kCPU).to(torch::kInt64).contiguous();
        int64_t batch_size = entity_indices.size(0);

        if (k_related_ <= 0) {
            return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
        }

        std::unordered_set<int64_t> unique_set;
        auto idx_ptr = entity_indices.data_ptr<int64_t>();

        for (int64_t i = 0; i < batch_size; i++) {
            int64_t entity_idx = idx_ptr[i];
            if (entity_idx >= 0 && entity_idx < num_entities_) {
                for (int64_t j = 0; j < k_related_; j++) {
                    unique_set.insert(related_[entity_idx][j]);
                }
            }
        }

        auto result = torch::empty({static_cast<int64_t>(unique_set.size())},
                                   torch::TensorOptions().dtype(torch::kInt64));
        auto ptr = result.data_ptr<int64_t>();

        int64_t idx = 0;
        for (int64_t entity : unique_set) {
            ptr[idx++] = entity;
        }

        return result;
    }

    int64_t k_related() const { return k_related_; }
    int64_t num_entities() const { return num_entities_; }

private:
    int64_t num_entities_;
    int64_t k_related_;
    std::vector<std::vector<int64_t>> related_;
    std::unordered_set<int64_t> seen_;  // Temp for unique computation
};


/**
 * HotsetRelatedLookup - Locality-aware related entity lookup
 *
 * Maps entities to buckets and draws related entities from the same bucket.
 * This creates high locality (cache reuse) for stress testing.
 */
class HotsetRelatedLookup {
public:
    HotsetRelatedLookup(int64_t num_entities, int64_t k_related,
                        int64_t hotset_size, int64_t seed = 42)
        : num_entities_(num_entities),
          k_related_(k_related),
          hotset_size_(hotset_size) {

        if (k_related <= 0) {
            return;
        }

        // Compute number of buckets
        num_buckets_ = (num_entities + hotset_size - 1) / hotset_size;

        // Pre-compute related entities for each entity from its bucket
        related_.resize(num_entities);
        std::mt19937_64 rng(seed);

        for (int64_t i = 0; i < num_entities; i++) {
            int64_t bucket = i / hotset_size;
            int64_t bucket_start = bucket * hotset_size;
            int64_t bucket_end = std::min(bucket_start + hotset_size, num_entities);
            int64_t bucket_size = bucket_end - bucket_start;

            related_[i].reserve(k_related);

            // Generate k related entities from same bucket
            std::unordered_set<int64_t> seen;
            seen.insert(i);  // Exclude self

            int64_t attempts = 0;
            while (static_cast<int64_t>(related_[i].size()) < k_related && attempts < k_related * 10) {
                attempts++;
                int64_t candidate = bucket_start + (rng() % bucket_size);
                if (seen.find(candidate) == seen.end()) {
                    related_[i].push_back(candidate);
                    seen.insert(candidate);
                }
            }

            // If bucket too small, fill with repeated entities
            while (static_cast<int64_t>(related_[i].size()) < k_related && bucket_size > 1) {
                int64_t candidate = bucket_start + (rng() % bucket_size);
                if (candidate != i) {
                    related_[i].push_back(candidate);
                }
            }
        }
    }

    torch::Tensor get_related(int64_t entity_idx) {
        if (k_related_ <= 0 || entity_idx < 0 || entity_idx >= num_entities_) {
            return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
        }

        int64_t actual_k = static_cast<int64_t>(related_[entity_idx].size());
        auto result = torch::empty({actual_k},
                                   torch::TensorOptions().dtype(torch::kInt64));
        auto ptr = result.data_ptr<int64_t>();

        for (int64_t i = 0; i < actual_k; i++) {
            ptr[i] = related_[entity_idx][i];
        }

        return result;
    }

    torch::Tensor get_batch_related(torch::Tensor entity_indices) {
        entity_indices = entity_indices.to(torch::kCPU).to(torch::kInt64).contiguous();
        int64_t batch_size = entity_indices.size(0);

        if (k_related_ <= 0) {
            return torch::empty({batch_size, 0},
                               torch::TensorOptions().dtype(torch::kInt64));
        }

        auto idx_ptr = entity_indices.data_ptr<int64_t>();
        auto result = torch::empty({batch_size, k_related_},
                                   torch::TensorOptions().dtype(torch::kInt64));
        auto result_ptr = result.data_ptr<int64_t>();

        #pragma omp parallel for
        for (int64_t i = 0; i < batch_size; i++) {
            int64_t entity_idx = idx_ptr[i];
            if (entity_idx >= 0 && entity_idx < num_entities_) {
                int64_t actual_k = static_cast<int64_t>(related_[entity_idx].size());
                for (int64_t j = 0; j < k_related_; j++) {
                    if (j < actual_k) {
                        result_ptr[i * k_related_ + j] = related_[entity_idx][j];
                    } else {
                        result_ptr[i * k_related_ + j] = related_[entity_idx][j % actual_k];
                    }
                }
            }
        }

        return result;
    }

    int64_t k_related() const { return k_related_; }
    int64_t num_entities() const { return num_entities_; }
    int64_t hotset_size() const { return hotset_size_; }
    int64_t num_buckets() const { return num_buckets_; }

private:
    int64_t num_entities_;
    int64_t k_related_;
    int64_t hotset_size_;
    int64_t num_buckets_;
    std::vector<std::vector<int64_t>> related_;
};


/**
 * StreamingFeatureExtractor - C++ hot path for streaming inference
 *
 * Combines entity window lookup + related entity lookup + cache tracking
 * into a single C++ call to minimize Python overhead.
 */
class StreamingFeatureExtractor {
public:
    StreamingFeatureExtractor(
        torch::Tensor entity_windows,      // [num_entities, window_size, feature_dim]
        int64_t k_related,
        bool use_hotset,
        int64_t hotset_size,
        int64_t page_size,
        int64_t feature_bytes,
        int64_t seed = 42
    ) : k_related_(k_related),
        use_hotset_(use_hotset),
        page_size_(page_size),
        feature_bytes_(feature_bytes) {

        // Store windows
        windows_ = entity_windows.to(torch::kCPU).to(torch::kFloat32).contiguous();
        num_entities_ = windows_.size(0);
        window_size_ = windows_.size(1);
        feature_dim_ = windows_.size(2);

        // Compute rows per page
        rows_per_page_ = std::max(1L, page_size / feature_bytes);

        // Create related lookup
        if (k_related > 0) {
            if (use_hotset) {
                hotset_lookup_ = std::make_unique<HotsetRelatedLookup>(
                    num_entities_, k_related, hotset_size, seed);
            } else {
                random_lookup_ = std::make_unique<RelatedEntityLookup>(
                    num_entities_, k_related, seed);
            }
        }

        // Initialize cache tracking
        pages_seen_.reserve(num_entities_ * window_size_ / rows_per_page_ + 1);
    }

    /**
     * Extract features for a single event.
     * Returns tuple: (entity_window, related_windows, metrics)
     *
     * Metrics tensor: [bytes_read, unique_pages, cache_hits, unique_related]
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    extract_single(int64_t entity_idx) {
        TORCH_CHECK(entity_idx >= 0 && entity_idx < num_entities_,
                    "Entity index out of bounds");

        // Get entity window
        auto entity_window = windows_.index({entity_idx}).clone();

        // Track pages for primary entity
        int64_t bytes_read = 0;
        int64_t unique_pages = 0;
        int64_t cache_hits = 0;

        for (int64_t row = 0; row < window_size_; row++) {
            // Simulate row -> page mapping
            int64_t global_row = entity_idx * window_size_ + row;
            int64_t page = global_row / rows_per_page_;

            if (pages_seen_.find(page) != pages_seen_.end()) {
                cache_hits++;
            } else {
                pages_seen_.insert(page);
                unique_pages++;
            }
        }
        bytes_read += window_size_ * feature_bytes_;

        // Get related entity windows
        torch::Tensor related_windows;
        int64_t unique_related = 0;

        if (k_related_ > 0) {
            torch::Tensor related_indices;
            if (use_hotset_) {
                related_indices = hotset_lookup_->get_related(entity_idx);
            } else {
                related_indices = random_lookup_->get_related(entity_idx);
            }

            int64_t actual_k = related_indices.size(0);
            if (actual_k > 0) {
                // Dedupe related entities
                std::unordered_set<int64_t> unique_set;
                auto rel_ptr = related_indices.data_ptr<int64_t>();
                for (int64_t i = 0; i < actual_k; i++) {
                    unique_set.insert(rel_ptr[i]);
                }
                unique_related = static_cast<int64_t>(unique_set.size());

                // Extract related windows
                related_windows = torch::empty({actual_k, window_size_, feature_dim_},
                                               torch::TensorOptions().dtype(torch::kFloat32));

                for (int64_t i = 0; i < actual_k; i++) {
                    int64_t rel_idx = rel_ptr[i];
                    related_windows.index({i}).copy_(windows_.index({rel_idx}));

                    // Track pages for related entities
                    for (int64_t row = 0; row < window_size_; row++) {
                        int64_t global_row = rel_idx * window_size_ + row;
                        int64_t page = global_row / rows_per_page_;

                        if (pages_seen_.find(page) != pages_seen_.end()) {
                            cache_hits++;
                        } else {
                            pages_seen_.insert(page);
                            unique_pages++;
                        }
                    }
                    bytes_read += window_size_ * feature_bytes_;
                }
            } else {
                related_windows = torch::empty({0, window_size_, feature_dim_},
                                               torch::TensorOptions().dtype(torch::kFloat32));
            }
        } else {
            related_windows = torch::empty({0, window_size_, feature_dim_},
                                           torch::TensorOptions().dtype(torch::kFloat32));
        }

        // Create metrics tensor
        auto metrics = torch::tensor({
            static_cast<double>(bytes_read),
            static_cast<double>(unique_pages),
            static_cast<double>(cache_hits),
            static_cast<double>(unique_related)
        }, torch::TensorOptions().dtype(torch::kFloat64));

        return std::make_tuple(entity_window, related_windows, metrics);
    }

    /**
     * Extract features for a batch of events (microbatch).
     * Returns tuple: (entity_windows, related_windows, metrics)
     *
     * entity_windows: [batch_size, window_size, feature_dim]
     * related_windows: [batch_size, k_related, window_size, feature_dim]
     * metrics: [batch_size, 4] - per-event metrics
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    extract_batch(torch::Tensor entity_indices) {
        entity_indices = entity_indices.to(torch::kCPU).to(torch::kInt64).contiguous();
        int64_t batch_size = entity_indices.size(0);
        auto idx_ptr = entity_indices.data_ptr<int64_t>();

        // Allocate outputs
        auto entity_windows = torch::empty({batch_size, window_size_, feature_dim_},
                                           torch::TensorOptions().dtype(torch::kFloat32));
        auto metrics = torch::zeros({batch_size, 4},
                                    torch::TensorOptions().dtype(torch::kFloat64));

        torch::Tensor related_windows;
        if (k_related_ > 0) {
            related_windows = torch::empty({batch_size, k_related_, window_size_, feature_dim_},
                                           torch::TensorOptions().dtype(torch::kFloat32));
        } else {
            related_windows = torch::empty({batch_size, 0, window_size_, feature_dim_},
                                           torch::TensorOptions().dtype(torch::kFloat32));
        }

        auto metrics_ptr = metrics.data_ptr<double>();

        // Process each event (could parallelize, but cache tracking needs sync)
        for (int64_t b = 0; b < batch_size; b++) {
            int64_t entity_idx = idx_ptr[b];

            // Copy entity window
            entity_windows.index({b}).copy_(windows_.index({entity_idx}));

            // Track pages for primary entity
            int64_t bytes_read = 0;
            int64_t unique_pages = 0;
            int64_t cache_hits = 0;

            for (int64_t row = 0; row < window_size_; row++) {
                int64_t global_row = entity_idx * window_size_ + row;
                int64_t page = global_row / rows_per_page_;

                if (pages_seen_.find(page) != pages_seen_.end()) {
                    cache_hits++;
                } else {
                    pages_seen_.insert(page);
                    unique_pages++;
                }
            }
            bytes_read += window_size_ * feature_bytes_;

            // Get related entities
            int64_t unique_related = 0;
            if (k_related_ > 0) {
                torch::Tensor related_indices;
                if (use_hotset_) {
                    related_indices = hotset_lookup_->get_related(entity_idx);
                } else {
                    related_indices = random_lookup_->get_related(entity_idx);
                }

                int64_t actual_k = related_indices.size(0);
                auto rel_ptr = related_indices.data_ptr<int64_t>();

                // Dedupe
                std::unordered_set<int64_t> unique_set;
                for (int64_t i = 0; i < actual_k; i++) {
                    unique_set.insert(rel_ptr[i]);
                }
                unique_related = static_cast<int64_t>(unique_set.size());

                // Copy related windows
                for (int64_t i = 0; i < k_related_; i++) {
                    int64_t rel_idx = (i < actual_k) ? rel_ptr[i] : rel_ptr[i % actual_k];
                    related_windows.index({b, i}).copy_(windows_.index({rel_idx}));

                    // Track pages
                    for (int64_t row = 0; row < window_size_; row++) {
                        int64_t global_row = rel_idx * window_size_ + row;
                        int64_t page = global_row / rows_per_page_;

                        if (pages_seen_.find(page) != pages_seen_.end()) {
                            cache_hits++;
                        } else {
                            pages_seen_.insert(page);
                            unique_pages++;
                        }
                    }
                    bytes_read += window_size_ * feature_bytes_;
                }
            }

            // Store metrics
            metrics_ptr[b * 4 + 0] = static_cast<double>(bytes_read);
            metrics_ptr[b * 4 + 1] = static_cast<double>(unique_pages);
            metrics_ptr[b * 4 + 2] = static_cast<double>(cache_hits);
            metrics_ptr[b * 4 + 3] = static_cast<double>(unique_related);
        }

        return std::make_tuple(entity_windows, related_windows, metrics);
    }

    void reset_cache() {
        pages_seen_.clear();
    }

    int64_t get_total_pages_seen() const {
        return static_cast<int64_t>(pages_seen_.size());
    }

    int64_t num_entities() const { return num_entities_; }
    int64_t window_size() const { return window_size_; }
    int64_t feature_dim() const { return feature_dim_; }
    int64_t k_related() const { return k_related_; }
    bool use_hotset() const { return use_hotset_; }

private:
    torch::Tensor windows_;
    int64_t num_entities_;
    int64_t window_size_;
    int64_t feature_dim_;
    int64_t k_related_;
    bool use_hotset_;
    int64_t page_size_;
    int64_t feature_bytes_;
    int64_t rows_per_page_;

    std::unique_ptr<RelatedEntityLookup> random_lookup_;
    std::unique_ptr<HotsetRelatedLookup> hotset_lookup_;
    std::unordered_set<int64_t> pages_seen_;
};


/**
 * QueueingMetrics - Track producer/consumer queue statistics
 */
class QueueingMetrics {
public:
    QueueingMetrics(int64_t capacity = 10000)
        : capacity_(capacity),
          total_enqueued_(0),
          total_dequeued_(0),
          total_dropped_(0),
          max_queue_depth_(0),
          current_depth_(0) {
        queue_depth_samples_.reserve(capacity);
    }

    void record_enqueue(bool dropped) {
        if (dropped) {
            total_dropped_++;
        } else {
            total_enqueued_++;
            current_depth_++;
            if (current_depth_ > max_queue_depth_) {
                max_queue_depth_ = current_depth_;
            }
        }
    }

    void record_dequeue() {
        total_dequeued_++;
        if (current_depth_ > 0) {
            current_depth_--;
        }
    }

    void sample_queue_depth() {
        if (static_cast<int64_t>(queue_depth_samples_.size()) < capacity_) {
            queue_depth_samples_.push_back(current_depth_);
        }
    }

    torch::Tensor get_summary() {
        // Returns: [enqueued, dequeued, dropped, max_depth, avg_depth, drop_rate]
        double avg_depth = 0.0;
        if (!queue_depth_samples_.empty()) {
            double sum = 0.0;
            for (int64_t d : queue_depth_samples_) {
                sum += d;
            }
            avg_depth = sum / queue_depth_samples_.size();
        }

        double drop_rate = 0.0;
        if (total_enqueued_ + total_dropped_ > 0) {
            drop_rate = static_cast<double>(total_dropped_) /
                       (total_enqueued_ + total_dropped_);
        }

        return torch::tensor({
            static_cast<double>(total_enqueued_),
            static_cast<double>(total_dequeued_),
            static_cast<double>(total_dropped_),
            static_cast<double>(max_queue_depth_),
            avg_depth,
            drop_rate
        }, torch::TensorOptions().dtype(torch::kFloat64));
    }

    void reset() {
        total_enqueued_ = 0;
        total_dequeued_ = 0;
        total_dropped_ = 0;
        max_queue_depth_ = 0;
        current_depth_ = 0;
        queue_depth_samples_.clear();
    }

    int64_t current_depth() const { return current_depth_; }
    int64_t max_queue_depth() const { return max_queue_depth_; }
    int64_t total_dropped() const { return total_dropped_; }

private:
    int64_t capacity_;
    int64_t total_enqueued_;
    int64_t total_dequeued_;
    int64_t total_dropped_;
    int64_t max_queue_depth_;
    int64_t current_depth_;
    std::vector<int64_t> queue_depth_samples_;
};


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

    // RelatedEntityLookup
    py::class_<RelatedEntityLookup>(m, "RelatedEntityLookup")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("num_entities"),
             py::arg("k_related"),
             py::arg("seed") = 42)
        .def("get_related", &RelatedEntityLookup::get_related)
        .def("get_batch_related", &RelatedEntityLookup::get_batch_related)
        .def("get_unique_related", &RelatedEntityLookup::get_unique_related)
        .def("k_related", &RelatedEntityLookup::k_related)
        .def("num_entities", &RelatedEntityLookup::num_entities);

    // HotsetRelatedLookup
    py::class_<HotsetRelatedLookup>(m, "HotsetRelatedLookup")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("num_entities"),
             py::arg("k_related"),
             py::arg("hotset_size"),
             py::arg("seed") = 42)
        .def("get_related", &HotsetRelatedLookup::get_related)
        .def("get_batch_related", &HotsetRelatedLookup::get_batch_related)
        .def("k_related", &HotsetRelatedLookup::k_related)
        .def("num_entities", &HotsetRelatedLookup::num_entities)
        .def("hotset_size", &HotsetRelatedLookup::hotset_size)
        .def("num_buckets", &HotsetRelatedLookup::num_buckets);

    // StreamingFeatureExtractor
    py::class_<StreamingFeatureExtractor>(m, "StreamingFeatureExtractor")
        .def(py::init<torch::Tensor, int64_t, bool, int64_t, int64_t, int64_t, int64_t>(),
             py::arg("entity_windows"),
             py::arg("k_related"),
             py::arg("use_hotset"),
             py::arg("hotset_size"),
             py::arg("page_size"),
             py::arg("feature_bytes"),
             py::arg("seed") = 42)
        .def("extract_single", &StreamingFeatureExtractor::extract_single)
        .def("extract_batch", &StreamingFeatureExtractor::extract_batch)
        .def("reset_cache", &StreamingFeatureExtractor::reset_cache)
        .def("get_total_pages_seen", &StreamingFeatureExtractor::get_total_pages_seen)
        .def("num_entities", &StreamingFeatureExtractor::num_entities)
        .def("window_size", &StreamingFeatureExtractor::window_size)
        .def("feature_dim", &StreamingFeatureExtractor::feature_dim)
        .def("k_related", &StreamingFeatureExtractor::k_related)
        .def("use_hotset", &StreamingFeatureExtractor::use_hotset);

    // QueueingMetrics
    py::class_<QueueingMetrics>(m, "QueueingMetrics")
        .def(py::init<int64_t>(),
             py::arg("capacity") = 10000)
        .def("record_enqueue", &QueueingMetrics::record_enqueue)
        .def("record_dequeue", &QueueingMetrics::record_dequeue)
        .def("sample_queue_depth", &QueueingMetrics::sample_queue_depth)
        .def("get_summary", &QueueingMetrics::get_summary)
        .def("reset", &QueueingMetrics::reset)
        .def("current_depth", &QueueingMetrics::current_depth)
        .def("max_queue_depth", &QueueingMetrics::max_queue_depth)
        .def("total_dropped", &QueueingMetrics::total_dropped);

    // Utility functions
    m.def("get_time_ms", &get_time_ms, "Get current time in milliseconds");
}
