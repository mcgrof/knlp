"""Pure QK router: score blocks using Q_t against K-summaries."""

import numpy as np


def score_blocks_cosine(
    q_vector: np.ndarray,
    k_summaries: np.ndarray,
    aggregation: str = "mean_across_layers",
) -> np.ndarray:
    """Score each block using cosine similarity between Q and K-summary.

    Args:
        q_vector: [num_layers, head_dim] query vector for current step
        k_summaries: [num_blocks, num_layers, head_dim] summary index
        aggregation: how to aggregate across layers

    Returns:
        [num_blocks] scores
    """
    num_blocks, num_layers, head_dim = k_summaries.shape

    # Per-layer cosine similarity
    scores_per_layer = np.zeros((num_blocks, num_layers))
    for l in range(num_layers):
        q = q_vector[l]  # [head_dim]
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            continue
        q_normed = q / q_norm

        for b in range(num_blocks):
            k = k_summaries[b, l]  # [head_dim] already normalized
            scores_per_layer[b, l] = np.dot(q_normed, k)

    # Aggregate
    if aggregation == "mean_across_layers":
        scores = scores_per_layer.mean(axis=1)
    elif aggregation == "max_across_layers":
        scores = scores_per_layer.max(axis=1)
    elif aggregation.startswith("last_"):
        n = int(aggregation.split("_")[1])
        scores = scores_per_layer[:, -n:].mean(axis=1)
    else:
        scores = scores_per_layer.mean(axis=1)

    return scores


def router_predict(
    q_vector: np.ndarray,
    k_summaries: np.ndarray,
    top_m: int = 8,
    score_fn: str = "cosine",
    aggregation: str = "mean_across_layers",
) -> tuple[list[int], np.ndarray]:
    """Router prediction: return top-M block indices and scores.

    Returns:
        (top_m_indices, all_scores)
    """
    if score_fn == "cosine":
        scores = score_blocks_cosine(q_vector, k_summaries, aggregation)
    elif score_fn == "dot":
        # Dot product (no normalization of Q)
        num_blocks = k_summaries.shape[0]
        num_layers = k_summaries.shape[1]
        scores_per_layer = np.zeros((num_blocks, num_layers))
        for l in range(num_layers):
            for b in range(num_blocks):
                scores_per_layer[b, l] = np.dot(q_vector[l], k_summaries[b, l])
        scores = scores_per_layer.mean(axis=1)
    else:
        raise ValueError(f"Unknown score_fn: {score_fn}")

    top_indices = np.argsort(scores)[::-1][:top_m].tolist()
    return top_indices, scores


def compute_router_metrics(
    predicted_blocks: list[int],
    needed_blocks: list[int],
    all_scores: np.ndarray,
    num_blocks: int,
    m_values: list[int] = None,
) -> dict:
    """Compute router-only metrics.

    Returns dict with recall@M, precision@M, score separation.
    """
    if m_values is None:
        m_values = [2, 4, 8, 16]

    needed_set = set(needed_blocks)
    sorted_indices = np.argsort(all_scores)[::-1]

    metrics = {}
    for m in m_values:
        top_m = set(sorted_indices[:m].tolist())
        if len(needed_set) > 0:
            recall = len(top_m & needed_set) / len(needed_set)
            precision = len(top_m & needed_set) / m if m > 0 else 0
        else:
            recall = 1.0
            precision = 1.0
        metrics[f"recall@{m}"] = recall
        metrics[f"precision@{m}"] = precision

    # Score separation
    if len(needed_set) > 0 and len(needed_set) < num_blocks:
        needed_scores = all_scores[list(needed_set)]
        not_needed = [i for i in range(num_blocks) if i not in needed_set]
        not_needed_scores = all_scores[not_needed]
        metrics["mean_needed_score"] = float(needed_scores.mean())
        metrics["mean_not_needed_score"] = float(not_needed_scores.mean())
        metrics["score_separation"] = float(
            needed_scores.mean() - not_needed_scores.mean()
        )
    else:
        metrics["score_separation"] = 0.0

    return metrics
