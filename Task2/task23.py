import numpy as np
from fifa_common import load_fifa_data, clean_and_select, get_feature_matrix, standardize, ALL_FEATURES

TARGET_NAME = "M. Salah"

def load_pool():
    raw = load_fifa_data()
    df = clean_and_select(raw)
    X = get_feature_matrix(df)    # raw (unstandardized) feature matrix
    Z, mean, std = standardize(X)   # standardized (z-score) feature matrix

    target_idx = df.index[df["Name"] == TARGET_NAME][0]
    print(f"[scout] target: {TARGET_NAME} (row {target_idx}), "
          f"pool size: {len(df)} players")
    return df, X, Z, target_idx

def cosine_similarity(X, target_vec):
    """
    Cosine similarity measures the angle between two vectors, ignoring
    their magnitude: 1.0 = identical direction (same "shape" of profile),
    0 = orthogonal, -1 = opposite direction.

        cos(a, b) = (a . b) / (||a|| * ||b||)

    Vectorized: compute the dot product of every row in X with the target
    in one matrix-vector multiply, and every row's norm in one call.
    """
    dot_products = X @ target_vec                      # (n_players,)
    row_norms = np.linalg.norm(X, axis=1)               # (n_players,)
    target_norm = np.linalg.norm(target_vec)
    return dot_products / (row_norms * target_norm)

def euclidean_distance(X, target_vec):
    """
    Euclidean distance is straight-line distance in feature space:
    it measures absolute differences in level across every attribute.

        d(a, b) = sqrt( sum( (a_i - b_i)^2 ) )

    Vectorized via broadcasting: (X - target_vec) subtracts the target
    from every row at once, no Python loop over players.
    Lower value = more similar.
    """
    diff = X - target_vec
    return np.sqrt((diff ** 2).sum(axis=1))

def manhattan_distance(X, target_vec):
    """
    Manhattan (city-block) distance sums absolute differences instead of
    squaring them, so it is less sensitive to any single large gap in
    one attribute than Euclidean distance is.

        d(a, b) = sum( |a_i - b_i| )

    Lower value = more similar.
    """
    diff = X - target_vec
    return np.abs(diff).sum(axis=1)

def top5(df, scores, target_idx, higher_is_better):
    """
    Return the 5 most similar players to the target, excluding the
    target itself. For similarity scores, higher is better (cosine);
    for distances, lower is better (euclidean/manhattan).
    """
    scores = scores.copy()
    scores[target_idx] = -np.inf if higher_is_better else np.inf  # exclude target

    order = np.argsort(scores)
    order = order[::-1] if higher_is_better else order
    top_idx = order[:5]

    result = df.loc[top_idx, ["Name", "Position"]].copy()
    result["score"] = scores[top_idx]
    return result.reset_index(drop=True)


def rank_of(scores, higher_is_better):
    """
    Convert a raw score array into a rank array (0 = most similar), so
    that scores from different metrics with different scales/directions
    (a cosine similarity of 0.99 vs a Euclidean distance of 2.3) can be
    combined on a common footing.
    """
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks


def consensus_shortlist(df, Z, target_idx, metrics, top_n=5):
    """
    Combine all three standardized metrics into one final shortlist by
    averaging each player's rank across cosine/euclidean/manhattan, then
    taking the 5 players with the best (lowest) average rank. A player
    who shows up near the top of all three metrics is a much safer
    recommendation than one who is #1 on a single metric only.
    """
    target_vec = Z[target_idx]
    all_ranks = []
    for metric_name, (fn, higher_is_better) in metrics.items():
        scores = fn(Z, target_vec)
        all_ranks.append(rank_of(scores, higher_is_better))

    avg_rank = np.mean(all_ranks, axis=0)
    avg_rank[target_idx] = np.inf  # exclude the target itself

    top_idx = np.argsort(avg_rank)[:top_n]
    result = df.loc[top_idx, ["Name", "Position"]].copy()
    result["avg_rank"] = avg_rank[top_idx]
    return result.reset_index(drop=True)


# main function
def main():
    df, X, Z, target_idx = load_pool()

    metrics = {
        "cosine": (cosine_similarity, True),
        "euclidean": (euclidean_distance, False),
        "manhattan": (manhattan_distance, False),
    }

    results = {"standardized": {}, "raw": {}}

    for feature_matrix, label in [(Z, "standardized"), (X, "raw")]:
        target_vec = feature_matrix[target_idx]
        print(f"\n{'='*70}\n{label.upper()} FEATURES\n{'='*70}")
        for metric_name, (fn, higher_is_better) in metrics.items():
            scores = fn(feature_matrix, target_vec)
            shortlist = top5(df, scores, target_idx, higher_is_better)
            results[label][metric_name] = shortlist
            print(f"\n-- {metric_name} ({label}) --")
            print(shortlist.to_string(index=False))

    print(f"\n{'='*70}\nFINAL CONSENSUS SHORTLIST (standardized, rank-averaged)\n{'='*70}")
    final_shortlist = consensus_shortlist(df, Z, target_idx, metrics)
    print(final_shortlist.to_string(index=False))
    results["consensus"] = final_shortlist

    return results


if __name__ == "__main__":
    main()
