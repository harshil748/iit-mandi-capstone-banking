"""Part 3 — Segmentation and dimensionality reduction."""
from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

import numpy as np

from helpers import (
    OUTPUT_DIR,
    load_records,
    sample_mean,
    sample_sd,
    write_csv,
    write_results,
)


def kmeans_wcss(Xz: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
    n = Xz.shape[0]
    step = n // k
    centroids = Xz[[i * step for i in range(k)]].copy()
    labels = np.zeros(n, dtype=int)
    while True:
        new_labels = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.sum((centroids - Xz[i]) ** 2, axis=1)
            # ties -> lower-numbered centroid
            new_labels[i] = int(np.argmin(dists))
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = Xz[labels == j]
            if members.size > 0:
                centroids[j] = np.mean(members, axis=0)
    wcss = 0.0
    for i in range(n):
        wcss += float(np.sum((Xz[i] - centroids[labels[i]]) ** 2))
    return wcss, labels


def cluster_label(members: Set[str]) -> str:
    ordered = sorted(members, key=lambda s: int(s[1:]))
    return "+".join(ordered)


def linkage_distance(
    a: Set[str], b: Set[str], D: Dict[Tuple[str, str], float], rule: str
) -> float:
    vals = []
    for i in a:
        for j in b:
            key = (i, j) if (i, j) in D else (j, i)
            vals.append(D[key])
    if rule == "single":
        return min(vals)
    if rule == "complete":
        return max(vals)
    return float(sum(vals) / len(vals))


def agglomerative(rule: str) -> Tuple[List[List[object]], int]:
    names = [f"E{i}" for i in range(1, 7)]
    # pairwise table from the brief
    mat = {
        ("E1", "E2"): 7.9,
        ("E1", "E3"): 7.4,
        ("E1", "E4"): 2.9,
        ("E1", "E5"): 2.9,
        ("E1", "E6"): 6.1,
        ("E2", "E3"): 1.4,
        ("E2", "E4"): 5.9,
        ("E2", "E5"): 5.3,
        ("E2", "E6"): 2.0,
        ("E3", "E4"): 5.8,
        ("E3", "E5"): 4.6,
        ("E3", "E6"): 1.3,
        ("E4", "E5"): 2.8,
        ("E4", "E6"): 4.5,
        ("E5", "E6"): 3.4,
    }
    D: Dict[Tuple[str, str], float] = {}
    for (a, b), v in mat.items():
        D[(a, b)] = v
        D[(b, a)] = v
    for n in names:
        D[(n, n)] = 0.0

    clusters: List[Set[str]] = [{n} for n in names]
    merges: List[List[object]] = []
    heights: List[float] = []

    for merge_no in range(1, 6):
        best = None
        best_i = best_j = -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = linkage_distance(clusters[i], clusters[j], D, rule)
                if best is None or dist < best:
                    best = dist
                    best_i, best_j = i, j
        ca = clusters[best_i]
        cb = clusters[best_j]
        # cluster_a is the one whose smallest member number is smaller
        min_a = min(int(x[1:]) for x in ca)
        min_b = min(int(x[1:]) for x in cb)
        if min_a <= min_b:
            left, right = ca, cb
        else:
            left, right = cb, ca
        merges.append(
            [rule, merge_no, cluster_label(left), cluster_label(right), float(best)]
        )
        heights.append(float(best))
        new_c = ca | cb
        # remove higher index first
        for idx in sorted([best_i, best_j], reverse=True):
            clusters.pop(idx)
        clusters.append(new_c)

    # cut: largest gap between consecutive merge heights
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    # cut lies in the largest gap; number of clusters = 6 - merges below the cut
    # merges below the cut = index of largest gap + 1? 
    # "the cut lies in the largest gap, and the number of clusters is 6 minus the number of merges below the cut"
    # If largest gap is between merge i and merge i+1 (0-based gap index g),
    # merges below the cut are merges 1..(g+1), so g+1 merges below cut.
    g = int(np.argmax(gaps))
    merges_below = g + 1
    n_clusters = 6 - merges_below
    return merges, n_clusters


def main() -> None:
    X, y, value = load_records()
    X5 = X[:, :5]
    n = X5.shape[0]
    means = np.array([sample_mean(X5[:, j]) for j in range(5)])
    sds = np.array([sample_sd(X5[:, j]) for j in range(5)])
    Xz = (X5 - means) / sds

    # 3.1 K-means
    wcss = {}
    labels_by_k = {}
    for k in range(2, 7):
        w, labels = kmeans_wcss(Xz, k)
        wcss[k] = w
        labels_by_k[k] = labels

    # 3.2 elbow
    elbow_k = None
    for k in (3, 4, 5):
        drop = wcss[k] - wcss[k + 1]
        prev = wcss[k - 1] - wcss[k]
        if drop < 0.5 * prev:
            elbow_k = k
            break
    if elbow_k is None:
        elbow_k = 5  # fallback; still use K=3 for rest

    # 3.3 profile K=3
    labels3 = labels_by_k[3]
    cluster_ids = sorted(set(int(c) for c in labels3))
    profiles = []
    for cid in cluster_ids:
        mask = labels3 == cid
        profiles.append(
            {
                "cid": cid,
                "mask": mask,
                "mean_X3": sample_mean(X5[mask, 2]),
                "size": int(np.sum(mask)),
                "mean_X1": sample_mean(X5[mask, 0]),
                "mean_X2": sample_mean(X5[mask, 1]),
                "mean_X4": sample_mean(X5[mask, 3]),
                "mean_X5": sample_mean(X5[mask, 4]),
                "outcome_rate": float(np.mean(y[mask])),
                "mean_value": sample_mean(value[mask]),
            }
        )
    profiles.sort(key=lambda p: p["mean_X3"])
    seg_names = ["A", "B", "C"]
    seg_rows = []
    size_keys = {}
    segment_masks = {}
    for name, p in zip(seg_names, profiles):
        size_keys[name] = p["size"]
        segment_masks[name] = p["mask"]
        seg_rows.append(
            [
                name,
                p["size"],
                p["mean_X1"],
                p["mean_X2"],
                p["mean_X3"],
                p["mean_X4"],
                p["mean_X5"],
                p["outcome_rate"],
                p["mean_value"],
            ]
        )

    # 3.4 hierarchical
    hier_rows: List[List[object]] = []
    cut_counts = {}
    for rule in ("single", "complete", "average"):
        merges, n_at_cut = agglomerative(rule)
        hier_rows.extend(merges)
        cut_counts[rule] = n_at_cut

    # 3.5 PCA
    # already standardised; mean is zero
    S = (Xz.T @ Xz) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # sign convention: flip so entry of largest abs value is positive
    for j in range(eigvecs.shape[1]):
        i_max = int(np.argmax(np.abs(eigvecs[:, j])))
        if eigvecs[i_max, j] < 0:
            eigvecs[:, j] *= -1

    eig_sum = float(np.sum(eigvals))
    explained = [float(v / eig_sum) for v in eigvals]
    cumulative = []
    running = 0.0
    for r in explained:
        running += r
        cumulative.append(running)
    components_for_80pct = next(i + 1 for i, c in enumerate(cumulative) if c >= 0.8)

    W2 = eigvecs[:, :2]
    Z = Xz @ W2
    row_proj = []
    for i in range(3):
        row_proj.extend([float(Z[i, 0]), float(Z[i, 1])])

    mean_pc1 = {
        name: float(np.mean(Z[segment_masks[name], 0])) for name in seg_names
    }

    items = [
        ("wcss_k2", wcss[2]),
        ("wcss_k3", wcss[3]),
        ("wcss_k4", wcss[4]),
        ("wcss_k5", wcss[5]),
        ("wcss_k6", wcss[6]),
        ("elbow_k", int(elbow_k)),
        ("size_segment_A", size_keys["A"]),
        ("size_segment_B", size_keys["B"]),
        ("size_segment_C", size_keys["C"]),
        ("single_clusters_at_cut", cut_counts["single"]),
        ("complete_clusters_at_cut", cut_counts["complete"]),
        ("average_clusters_at_cut", cut_counts["average"]),
        ("eigenvalue_sum", eig_sum),
        ("eigenvalue_1", float(eigvals[0])),
        ("eigenvalue_2", float(eigvals[1])),
        ("eigenvalue_3", float(eigvals[2])),
        ("eigenvalue_4", float(eigvals[3])),
        ("eigenvalue_5", float(eigvals[4])),
        ("explained_ratio_1", explained[0]),
        ("explained_ratio_2", explained[1]),
        ("explained_ratio_3", explained[2]),
        ("explained_ratio_4", explained[3]),
        ("explained_ratio_5", explained[4]),
        ("cumulative_ratio_1", cumulative[0]),
        ("cumulative_ratio_2", cumulative[1]),
        ("cumulative_ratio_3", cumulative[2]),
        ("cumulative_ratio_4", cumulative[3]),
        ("cumulative_ratio_5", cumulative[4]),
        ("components_for_80pct", components_for_80pct),
        ("pc1_loading_X1", float(eigvecs[0, 0])),
        ("pc1_loading_X2", float(eigvecs[1, 0])),
        ("pc1_loading_X3", float(eigvecs[2, 0])),
        ("pc1_loading_X4", float(eigvecs[3, 0])),
        ("pc1_loading_X5", float(eigvecs[4, 0])),
        ("pc2_loading_X1", float(eigvecs[0, 1])),
        ("pc2_loading_X2", float(eigvecs[1, 1])),
        ("pc2_loading_X3", float(eigvecs[2, 1])),
        ("pc2_loading_X4", float(eigvecs[3, 1])),
        ("pc2_loading_X5", float(eigvecs[4, 1])),
        ("row1_pc1", row_proj[0]),
        ("row1_pc2", row_proj[1]),
        ("row2_pc1", row_proj[2]),
        ("row2_pc2", row_proj[3]),
        ("row3_pc1", row_proj[4]),
        ("row3_pc2", row_proj[5]),
        ("mean_pc1_segment_A", mean_pc1["A"]),
        ("mean_pc1_segment_B", mean_pc1["B"]),
        ("mean_pc1_segment_C", mean_pc1["C"]),
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results(os.path.join(OUTPUT_DIR, "part3_results.txt"), items)
    write_csv(
        os.path.join(OUTPUT_DIR, "part3_segments.csv"),
        [
            "segment",
            "size",
            "mean_X1",
            "mean_X2",
            "mean_X3",
            "mean_X4",
            "mean_X5",
            "outcome_rate",
            "mean_value",
        ],
        seg_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "part3_hierarchical.csv"),
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hier_rows,
    )
    print("Wrote part3 outputs")
    print(f"Checkpoint: wcss_k2={wcss[2]:.4f}, eigenvalue_sum={eig_sum:.4f}, elbow_k={elbow_k}")


if __name__ == "__main__":
    main()
