/**
 * Page-Batch Sampler - C++ Extension
 *
 * High-performance implementation of page-batch sampling for GNN training.
 * Processes entire memory pages as batches with only intra-page edges.
 *
 * Key operations:
 * 1. build_page_structures() - One-time setup of page-level data structures
 * 2. sample_batch() - Fast batch assembly from selected pages
 */

#include <torch/extension.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>

/**
 * Build page-level data structures for efficient batch sampling.
 *
 * Returns:
 *   - page_node_offsets: CSR-style offsets for nodes per page
 *   - page_node_indices: Flattened node IDs grouped by page
 *   - page_edge_offsets: CSR-style offsets for edges per page
 *   - page_edge_src: Flattened intra-page edge sources (global IDs)
 *   - page_edge_dst: Flattened intra-page edge destinations (global IDs)
 *   - page_train_offsets: CSR-style offsets for training nodes per page
 *   - page_train_indices: Flattened training node IDs grouped by page
 */
std::tuple<
    torch::Tensor,  // page_node_offsets
    torch::Tensor,  // page_node_indices
    torch::Tensor,  // page_edge_offsets
    torch::Tensor,  // page_edge_src
    torch::Tensor,  // page_edge_dst
    torch::Tensor,  // page_train_offsets
    torch::Tensor   // page_train_indices
>
build_page_structures(
    torch::Tensor page_id,        // [num_nodes] page ID for each node
    torch::Tensor edge_src,       // [num_edges] source nodes
    torch::Tensor edge_dst,       // [num_edges] destination nodes
    torch::Tensor train_mask,     // [num_nodes] boolean mask for training nodes
    int64_t num_pages
) {
    // Move to CPU and ensure correct types
    page_id = page_id.to(torch::kCPU).to(torch::kInt32).contiguous();
    edge_src = edge_src.to(torch::kCPU).to(torch::kInt64).contiguous();
    edge_dst = edge_dst.to(torch::kCPU).to(torch::kInt64).contiguous();
    train_mask = train_mask.to(torch::kCPU).to(torch::kBool).contiguous();

    const int32_t* page_id_ptr = page_id.data_ptr<int32_t>();
    const int64_t* src_ptr = edge_src.data_ptr<int64_t>();
    const int64_t* dst_ptr = edge_dst.data_ptr<int64_t>();
    const bool* train_ptr = train_mask.data_ptr<bool>();

    int64_t num_nodes = page_id.size(0);
    int64_t num_edges = edge_src.size(0);

    // Step 1: Count nodes and training nodes per page
    std::vector<int64_t> node_counts(num_pages, 0);
    std::vector<int64_t> train_counts(num_pages, 0);

    #pragma omp parallel
    {
        std::vector<int64_t> local_node_counts(num_pages, 0);
        std::vector<int64_t> local_train_counts(num_pages, 0);

        #pragma omp for nowait
        for (int64_t i = 0; i < num_nodes; i++) {
            int32_t pg = page_id_ptr[i];
            if (pg >= 0 && pg < num_pages) {
                local_node_counts[pg]++;
                if (train_ptr[i]) {
                    local_train_counts[pg]++;
                }
            }
        }

        #pragma omp critical
        {
            for (int64_t p = 0; p < num_pages; p++) {
                node_counts[p] += local_node_counts[p];
                train_counts[p] += local_train_counts[p];
            }
        }
    }

    // Step 2: Count intra-page edges per page
    std::vector<int64_t> edge_counts(num_pages, 0);

    #pragma omp parallel
    {
        std::vector<int64_t> local_edge_counts(num_pages, 0);

        #pragma omp for nowait
        for (int64_t i = 0; i < num_edges; i++) {
            int64_t s = src_ptr[i];
            int64_t d = dst_ptr[i];
            int32_t src_page = page_id_ptr[s];
            int32_t dst_page = page_id_ptr[d];
            if (src_page == dst_page && src_page >= 0 && src_page < num_pages) {
                local_edge_counts[src_page]++;
            }
        }

        #pragma omp critical
        {
            for (int64_t p = 0; p < num_pages; p++) {
                edge_counts[p] += local_edge_counts[p];
            }
        }
    }

    // Step 3: Build CSR offsets
    std::vector<int64_t> node_offsets(num_pages + 1, 0);
    std::vector<int64_t> edge_offsets(num_pages + 1, 0);
    std::vector<int64_t> train_offsets(num_pages + 1, 0);

    for (int64_t p = 0; p < num_pages; p++) {
        node_offsets[p + 1] = node_offsets[p] + node_counts[p];
        edge_offsets[p + 1] = edge_offsets[p] + edge_counts[p];
        train_offsets[p + 1] = train_offsets[p] + train_counts[p];
    }

    int64_t total_nodes = node_offsets[num_pages];
    int64_t total_edges = edge_offsets[num_pages];
    int64_t total_train = train_offsets[num_pages];

    // Step 4: Allocate output arrays
    auto node_indices = torch::empty({total_nodes}, torch::kInt64);
    auto intra_edge_src = torch::empty({total_edges}, torch::kInt64);
    auto intra_edge_dst = torch::empty({total_edges}, torch::kInt64);
    auto train_indices = torch::empty({total_train}, torch::kInt64);

    int64_t* node_indices_ptr = node_indices.data_ptr<int64_t>();
    int64_t* intra_src_ptr = intra_edge_src.data_ptr<int64_t>();
    int64_t* intra_dst_ptr = intra_edge_dst.data_ptr<int64_t>();
    int64_t* train_indices_ptr = train_indices.data_ptr<int64_t>();

    // Step 5: Fill node indices (need atomic counters per page)
    std::vector<int64_t> node_write_pos(num_pages);
    std::vector<int64_t> train_write_pos(num_pages);
    for (int64_t p = 0; p < num_pages; p++) {
        node_write_pos[p] = node_offsets[p];
        train_write_pos[p] = train_offsets[p];
    }

    // Single-threaded for correctness (ordering within page doesn't matter)
    for (int64_t i = 0; i < num_nodes; i++) {
        int32_t pg = page_id_ptr[i];
        if (pg >= 0 && pg < num_pages) {
            node_indices_ptr[node_write_pos[pg]++] = i;
            if (train_ptr[i]) {
                train_indices_ptr[train_write_pos[pg]++] = i;
            }
        }
    }

    // Step 6: Fill intra-page edges
    std::vector<int64_t> edge_write_pos(num_pages);
    for (int64_t p = 0; p < num_pages; p++) {
        edge_write_pos[p] = edge_offsets[p];
    }

    for (int64_t i = 0; i < num_edges; i++) {
        int64_t s = src_ptr[i];
        int64_t d = dst_ptr[i];
        int32_t src_page = page_id_ptr[s];
        int32_t dst_page = page_id_ptr[d];
        if (src_page == dst_page && src_page >= 0 && src_page < num_pages) {
            int64_t pos = edge_write_pos[src_page]++;
            intra_src_ptr[pos] = s;
            intra_dst_ptr[pos] = d;
        }
    }

    // Convert offsets to tensors
    auto page_node_offsets = torch::from_blob(
        node_offsets.data(), {num_pages + 1}, torch::kInt64
    ).clone();
    auto page_edge_offsets = torch::from_blob(
        edge_offsets.data(), {num_pages + 1}, torch::kInt64
    ).clone();
    auto page_train_offsets = torch::from_blob(
        train_offsets.data(), {num_pages + 1}, torch::kInt64
    ).clone();

    return std::make_tuple(
        page_node_offsets,
        node_indices,
        page_edge_offsets,
        intra_edge_src,
        intra_edge_dst,
        page_train_offsets,
        train_indices
    );
}


/**
 * Assemble a batch from selected pages.
 *
 * Given pre-built page structures and a list of page IDs, efficiently
 * concatenate all nodes and edges, remapping edge indices to local space.
 *
 * Returns:
 *   - batch_nodes: Global node IDs in this batch
 *   - batch_edge_index: [2, num_edges] local edge indices
 *   - batch_train_mask: Boolean mask indicating training nodes in batch
 *   - num_train_nodes: Count of training nodes for loss weighting
 */
std::tuple<
    torch::Tensor,  // batch_nodes (global IDs)
    torch::Tensor,  // batch_edge_index (local, [2, E])
    torch::Tensor,  // batch_train_mask (bool, [N])
    int64_t         // num_train_nodes
>
sample_batch(
    torch::Tensor batch_pages,        // [batch_size] page IDs to sample
    torch::Tensor page_node_offsets,  // CSR offsets for nodes
    torch::Tensor page_node_indices,  // Flattened node IDs
    torch::Tensor page_edge_offsets,  // CSR offsets for edges
    torch::Tensor page_edge_src,      // Flattened edge sources (global)
    torch::Tensor page_edge_dst,      // Flattened edge destinations (global)
    torch::Tensor page_train_offsets, // CSR offsets for training nodes
    torch::Tensor page_train_indices, // Flattened training node IDs
    int64_t num_nodes_total           // Total nodes in graph (for bounds check)
) {
    // Ensure CPU and correct types
    batch_pages = batch_pages.to(torch::kCPU).to(torch::kInt64).contiguous();
    page_node_offsets = page_node_offsets.to(torch::kCPU).to(torch::kInt64).contiguous();
    page_node_indices = page_node_indices.to(torch::kCPU).to(torch::kInt64).contiguous();
    page_edge_offsets = page_edge_offsets.to(torch::kCPU).to(torch::kInt64).contiguous();
    page_edge_src = page_edge_src.to(torch::kCPU).to(torch::kInt64).contiguous();
    page_edge_dst = page_edge_dst.to(torch::kCPU).to(torch::kInt64).contiguous();
    page_train_offsets = page_train_offsets.to(torch::kCPU).to(torch::kInt64).contiguous();
    page_train_indices = page_train_indices.to(torch::kCPU).to(torch::kInt64).contiguous();

    const int64_t* pages_ptr = batch_pages.data_ptr<int64_t>();
    const int64_t* node_off_ptr = page_node_offsets.data_ptr<int64_t>();
    const int64_t* node_idx_ptr = page_node_indices.data_ptr<int64_t>();
    const int64_t* edge_off_ptr = page_edge_offsets.data_ptr<int64_t>();
    const int64_t* edge_src_ptr = page_edge_src.data_ptr<int64_t>();
    const int64_t* edge_dst_ptr = page_edge_dst.data_ptr<int64_t>();
    const int64_t* train_off_ptr = page_train_offsets.data_ptr<int64_t>();
    const int64_t* train_idx_ptr = page_train_indices.data_ptr<int64_t>();

    int64_t num_batch_pages = batch_pages.size(0);

    // Step 1: Count total nodes and edges in batch
    int64_t total_batch_nodes = 0;
    int64_t total_batch_edges = 0;
    int64_t total_batch_train = 0;

    for (int64_t i = 0; i < num_batch_pages; i++) {
        int64_t pg = pages_ptr[i];
        total_batch_nodes += node_off_ptr[pg + 1] - node_off_ptr[pg];
        total_batch_edges += edge_off_ptr[pg + 1] - edge_off_ptr[pg];
        total_batch_train += train_off_ptr[pg + 1] - train_off_ptr[pg];
    }

    // Step 2: Allocate output tensors
    auto batch_nodes = torch::empty({total_batch_nodes}, torch::kInt64);
    auto batch_edge_src = torch::empty({total_batch_edges}, torch::kInt64);
    auto batch_edge_dst = torch::empty({total_batch_edges}, torch::kInt64);
    auto batch_train_mask = torch::zeros({total_batch_nodes}, torch::kBool);

    int64_t* batch_nodes_ptr = batch_nodes.data_ptr<int64_t>();
    int64_t* batch_src_ptr = batch_edge_src.data_ptr<int64_t>();
    int64_t* batch_dst_ptr = batch_edge_dst.data_ptr<int64_t>();
    bool* batch_train_ptr = batch_train_mask.data_ptr<bool>();

    // Step 3: Build global-to-local mapping and copy nodes
    std::unordered_map<int64_t, int64_t> global_to_local;
    global_to_local.reserve(total_batch_nodes);

    int64_t local_idx = 0;
    for (int64_t i = 0; i < num_batch_pages; i++) {
        int64_t pg = pages_ptr[i];
        int64_t start = node_off_ptr[pg];
        int64_t end = node_off_ptr[pg + 1];

        for (int64_t j = start; j < end; j++) {
            int64_t global_node = node_idx_ptr[j];
            batch_nodes_ptr[local_idx] = global_node;
            global_to_local[global_node] = local_idx;
            local_idx++;
        }
    }

    // Step 4: Mark training nodes in batch
    for (int64_t i = 0; i < num_batch_pages; i++) {
        int64_t pg = pages_ptr[i];
        int64_t start = train_off_ptr[pg];
        int64_t end = train_off_ptr[pg + 1];

        for (int64_t j = start; j < end; j++) {
            int64_t global_train_node = train_idx_ptr[j];
            auto it = global_to_local.find(global_train_node);
            if (it != global_to_local.end()) {
                batch_train_ptr[it->second] = true;
            }
        }
    }

    // Step 5: Copy and remap edges to local indices
    int64_t edge_idx = 0;
    for (int64_t i = 0; i < num_batch_pages; i++) {
        int64_t pg = pages_ptr[i];
        int64_t start = edge_off_ptr[pg];
        int64_t end = edge_off_ptr[pg + 1];

        for (int64_t j = start; j < end; j++) {
            int64_t global_src = edge_src_ptr[j];
            int64_t global_dst = edge_dst_ptr[j];

            // Remap to local indices
            auto src_it = global_to_local.find(global_src);
            auto dst_it = global_to_local.find(global_dst);

            if (src_it != global_to_local.end() && dst_it != global_to_local.end()) {
                batch_src_ptr[edge_idx] = src_it->second;
                batch_dst_ptr[edge_idx] = dst_it->second;
                edge_idx++;
            }
        }
    }

    // Trim edges if some were skipped (shouldn't happen with correct data)
    if (edge_idx < total_batch_edges) {
        batch_edge_src = batch_edge_src.slice(0, 0, edge_idx);
        batch_edge_dst = batch_edge_dst.slice(0, 0, edge_idx);
    }

    // Stack into [2, E] edge_index format
    auto batch_edge_index = torch::stack({batch_edge_src, batch_edge_dst}, 0);

    return std::make_tuple(
        batch_nodes,
        batch_edge_index,
        batch_train_mask,
        total_batch_train
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_page_structures", &build_page_structures,
          "Build page-level CSR structures for batch sampling",
          py::arg("page_id"),
          py::arg("edge_src"),
          py::arg("edge_dst"),
          py::arg("train_mask"),
          py::arg("num_pages"));

    m.def("sample_batch", &sample_batch,
          "Assemble a batch from selected pages",
          py::arg("batch_pages"),
          py::arg("page_node_offsets"),
          py::arg("page_node_indices"),
          py::arg("page_edge_offsets"),
          py::arg("page_edge_src"),
          py::arg("page_edge_dst"),
          py::arg("page_train_offsets"),
          py::arg("page_train_indices"),
          py::arg("num_nodes_total"));
}
