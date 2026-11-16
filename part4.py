"""Part 4 — Model-choice and architecture memo numbers."""
from __future__ import annotations

import math
import os

import numpy as np

from helpers import (
    OUTPUT_DIR,
    confusion_and_metrics,
    knn_predict,
    load_records,
    predict_proba_logistic,
    standardise_train_test,
    stratified_split,
    fit_logistic,
    write_csv,
    write_results,
)


def entropy(p_pos: float) -> float:
    if p_pos <= 0.0 or p_pos >= 1.0:
        return 0.0
    p0 = 1.0 - p_pos
    return float(-(p_pos * math.log2(p_pos) + p0 * math.log2(p0)))


def gini(p_pos: float) -> float:
    p0 = 1.0 - p_pos
    return float(1.0 - (p_pos ** 2 + p0 ** 2))


def split_gains(y: np.ndarray, yes_mask: np.ndarray):
    n = y.size
    n_yes = int(np.sum(yes_mask))
    n_no = n - n_yes
    pos_yes = int(np.sum(y[yes_mask])) if n_yes else 0
    pos_no = int(np.sum(y[~yes_mask])) if n_no else 0
    p_parent = float(np.mean(y))
    e_parent = entropy(p_parent)
    g_parent = gini(p_parent)
    e_yes = entropy(pos_yes / n_yes) if n_yes else 0.0
    e_no = entropy(pos_no / n_no) if n_no else 0.0
    g_yes = gini(pos_yes / n_yes) if n_yes else 0.0
    g_no = gini(pos_no / n_no) if n_no else 0.0
    info_gain = e_parent - (n_yes / n) * e_yes - (n_no / n) * e_no
    gini_gain = g_parent - (n_yes / n) * g_yes - (n_no / n) * g_no
    return n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain


def main() -> None:
    X, y, _ = load_records()
    train_idx, test_idx = stratified_split(y)
    X_train_raw = X[train_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train, X_test, _, _ = standardise_train_test(X, train_idx, test_idx)

    # Refit Part 2 models (same code path)
    w, b, _ = fit_logistic(X_train, y_train, eta=0.1, n_updates=1000, record_at=[1000])
    p_test = predict_proba_logistic(X_test, w, b)

    ks = list(range(1, 22, 2))
    folds = np.arange(X_train.shape[0]) % 5
    mean_f1s = []
    for k in ks:
        fold_f1 = []
        for fold in range(5):
            val_mask = folds == fold
            ref_mask = ~val_mask
            pred, _ = knn_predict(
                X_train[ref_mask], y_train[ref_mask], X_train[val_mask], k
            )
            fold_f1.append(confusion_and_metrics(y_train[val_mask], pred)["f1"])
        mean_f1s.append(float(np.mean(fold_f1)))
    best_k = ks[int(np.argmax(mean_f1s))]
    knn_pred, knn_share = knn_predict(X_train, y_train, X_test, best_k)

    # 4.1 splits on original units
    root_p = float(np.mean(y_train))
    root_entropy = entropy(root_p)
    root_gini = gini(root_p)

    split_defs = {
        "S1": X_train_raw[:, 5] == 1,
        "S2": X_train_raw[:, 6] == 1,
        "S3": X_train_raw[:, 2] > 0.5,
        "S4": X_train_raw[:, 0] <= 24,
    }
    split_rows = []
    info_gains = {}
    gini_gains = {}
    for name in ("S1", "S2", "S3", "S4"):
        n_yes, pos_yes, n_no, pos_no, ig, gg = split_gains(y_train, split_defs[name])
        info_gains[name] = ig
        gini_gains[name] = gg
        split_rows.append([name, n_yes, pos_yes, n_no, pos_no, ig, gg])

    best_split_by_info_gain = max(("S1", "S2", "S3", "S4"), key=lambda s: info_gains[s])
    best_split_by_gini_gain = max(("S1", "S2", "S3", "S4"), key=lambda s: gini_gains[s])

    # 4.2 stump + voting
    best = best_split_by_info_gain
    yes_mask = split_defs[best]
    stump_rate_yes = float(np.mean(y_train[yes_mask])) if np.any(yes_mask) else 0.0
    stump_rate_no = float(np.mean(y_train[~yes_mask])) if np.any(~yes_mask) else 0.0

    # stump on test using same split rule on original test features
    X_test_raw = X[test_idx]
    if best == "S1":
        test_yes = X_test_raw[:, 5] == 1
    elif best == "S2":
        test_yes = X_test_raw[:, 6] == 1
    elif best == "S3":
        test_yes = X_test_raw[:, 2] > 0.5
    else:
        test_yes = X_test_raw[:, 0] <= 24

    stump_prob = np.where(test_yes, stump_rate_yes, stump_rate_no)
    stump_pred = (stump_prob >= 0.3).astype(int)
    logistic_pred = (p_test >= 0.3).astype(int)
    knn_bin = knn_pred  # already majority vote

    hard_vote = ((logistic_pred + knn_bin + stump_pred) >= 2).astype(int)
    soft_prob = (p_test + knn_share + stump_prob) / 3.0
    soft_vote = (soft_prob >= 0.3).astype(int)

    vote_rows = []
    for name, pred in (
        ("logistic", logistic_pred),
        ("knn", knn_bin),
        ("stump", stump_pred),
        ("hard_vote", hard_vote),
        ("soft_vote", soft_vote),
    ):
        m = confusion_and_metrics(y_test, pred)
        vote_rows.append(
            [name, m["tp"], m["fp"], m["tn"], m["fn"], m["precision"], m["recall"], m["f1"]]
        )

    # 4.3 AdaBoost by hand
    y_ab = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=float)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=float)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=float)

    def adaboost_round(w, h, y):
        wrong = h != y
        err = float(np.sum(w[wrong]))
        alpha = 0.5 * math.log((1.0 - err) / err)
        unnorm = w * np.exp(-alpha * y * h)
        after = unnorm / np.sum(unnorm)
        return err, alpha, w.copy(), unnorm, after

    w0 = np.ones(8) / 8.0
    r1_err, r1_alpha, before1, unnorm1, after1 = adaboost_round(w0, h1, y_ab)
    r2_err, r2_alpha, before2, unnorm2, after2 = adaboost_round(after1, h2, y_ab)

    ada_rows = []
    for point in range(8):
        ada_rows.append(
            [1, point + 1, float(before1[point]), float(unnorm1[point]), float(after1[point])]
        )
    for point in range(8):
        ada_rows.append(
            [2, point + 1, float(before2[point]), float(unnorm2[point]), float(after2[point])]
        )

    ensemble = np.sign(r1_alpha * h1 + r2_alpha * h2)
    # sign(0) -> 0; treat as incorrect unless y==0 which never happens
    ensemble_accuracy = float(np.mean(ensemble == y_ab))

    # 4.4 OOB
    oob_fraction = (1.0 - 1.0 / 560.0) ** 560

    # 4.5 architecture counts
    # MLP: 784->16->16->10
    mlp_hidden1 = 784 * 16 + 16
    mlp_hidden2 = 16 * 16 + 16
    mlp_output = 16 * 10 + 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    # CNN
    # conv1: 5x5, 6 filters, in_channels=1 -> params = 5*5*1*6 + 6
    cnn_conv1 = 5 * 5 * 1 * 6 + 6
    # after conv1 pad2 stride1 on 28: (28-5+4)/1+1 = 28; pool2 -> 14
    # conv2: 5x5, 16 filters, in=6 -> 5*5*6*16 + 16
    cnn_conv2 = 5 * 5 * 6 * 16 + 16
    # after conv2 pad0: (14-5)/1+1 = 10; pool2 -> 5; flatten 5*5*16 = 400
    cnn_dense1 = 400 * 32 + 32
    cnn_dense2 = 32 * 10 + 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    # FC vs conv on 128x128x3
    fc_layer_params = 128 * 128 * 3 * 1024 + 1024
    conv_layer_params = 3 * 3 * 3 * 64 + 64

    # RNN: X=64, H=128, Y=16
    # W_xh: 128x64, W_hh: 128x128, b_h: 128, W_hy: 16x128, b_y: 16
    rnn_total = 128 * 64 + 128 * 128 + 128 + 16 * 128 + 16

    arch_rows = [
        ["mlp_hidden1", mlp_hidden1],
        ["mlp_hidden2", mlp_hidden2],
        ["mlp_output", mlp_output],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2],
    ]

    items = [
        ("root_entropy", root_entropy),
        ("root_gini", root_gini),
        ("best_split_by_info_gain", best_split_by_info_gain),
        ("best_split_by_gini_gain", best_split_by_gini_gain),
        ("stump_rate_yes", stump_rate_yes),
        ("stump_rate_no", stump_rate_no),
        ("round1_error", r1_err),
        ("round1_alpha", r1_alpha),
        ("round2_error", r2_err),
        ("round2_alpha", r2_alpha),
        ("ensemble_accuracy", ensemble_accuracy),
        ("oob_fraction", oob_fraction),
        ("mlp_total", mlp_total),
        ("cnn_total", cnn_total),
        ("fc_layer_params", fc_layer_params),
        ("conv_layer_params", conv_layer_params),
        ("rnn_total", rnn_total),
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results(os.path.join(OUTPUT_DIR, "part4_results.txt"), items)
    write_csv(
        os.path.join(OUTPUT_DIR, "part4_splits.csv"),
        ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
        split_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "part4_voting.csv"),
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        vote_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "part4_adaboost.csv"),
        ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        ada_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "part4_architecture.csv"),
        ["layer", "parameters"],
        arch_rows,
    )
    print("Wrote part4 outputs")
    print(
        f"Checkpoint: root_entropy={root_entropy:.4f}, round1_error={r1_err:.4f}, "
        f"best_info={best_split_by_info_gain}, mlp_total={mlp_total}, cnn_total={cnn_total}"
    )


if __name__ == "__main__":
    main()
