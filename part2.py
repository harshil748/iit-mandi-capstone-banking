"""Part 2 — Classification and evaluation."""
from __future__ import annotations

import math
import os

import numpy as np

from helpers import (
    OUTPUT_DIR,
    binary_cross_entropy,
    confusion_and_metrics,
    fit_logistic,
    knn_predict,
    load_records,
    predict_proba_logistic,
    standardise_train_test,
    stratified_split,
    write_csv,
    write_results,
)


def main() -> None:
    X, y, _ = load_records()
    train_idx, test_idx = stratified_split(y)
    X_train, X_test, _, _ = standardise_train_test(X, train_idx, test_idx)
    y_train = y[train_idx]
    y_test = y[test_idx]

    train_rows = int(train_idx.size)
    test_rows = int(test_idx.size)
    train_positives = int(np.sum(y_train))
    test_positives = int(np.sum(y_test))

    # 2.2 logistic
    record_at = [0, 1, 10, 100, 1000]
    w, b, losses = fit_logistic(X_train, y_train, eta=0.1, n_updates=1000, record_at=record_at)
    p_test = predict_proba_logistic(X_test, w, b)
    p_train = predict_proba_logistic(X_train, w, b)
    test_bce = binary_cross_entropy(y_test, p_test)

    # 2.3 learning-rate effect
    lr_losses = {}
    for eta in (0.01, 0.1, 1.0):
        _, _, L = fit_logistic(X_train, y_train, eta=eta, n_updates=200, record_at=[200])
        lr_losses[eta] = L[200]

    # 2.4 odds ratios
    odds = [math.exp(float(wj)) for wj in w]

    # 2.5 evaluate at 0.5 and 0.3
    items_metrics = {}
    for tau in (0.5, 0.3):
        pred = (p_test >= tau).astype(int)
        m = confusion_and_metrics(y_test, pred)
        prefix = f"tau_{tau}"
        items_metrics[f"{prefix}_tp"] = m["tp"]
        items_metrics[f"{prefix}_fp"] = m["fp"]
        items_metrics[f"{prefix}_tn"] = m["tn"]
        items_metrics[f"{prefix}_fn"] = m["fn"]
        items_metrics[f"{prefix}_accuracy"] = m["accuracy"]
        items_metrics[f"{prefix}_precision"] = m["precision"]
        items_metrics[f"{prefix}_recall"] = m["recall"]
        items_metrics[f"{prefix}_f1"] = m["f1"]
        items_metrics[f"{prefix}_specificity"] = m["specificity"]

    blind_accuracy = float(np.mean(y_test == 0))

    # 2.6 threshold sweep
    thresh_rows = []
    for k in range(1, 10):
        tau = k / 10.0
        pred = (p_test >= tau).astype(int)
        m = confusion_and_metrics(y_test, pred)
        thresh_rows.append([tau, m["tp"], m["fp"], m["tn"], m["fn"], m["tpr"], m["fpr"]])

    # 2.7 KNN CV
    ks = list(range(1, 22, 2))
    folds = np.arange(train_rows) % 5
    cv_rows = []
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
        mean_f1 = float(np.mean(fold_f1))
        mean_f1s.append(mean_f1)
        cv_rows.append([k, mean_f1])

    best_i = int(np.argmax(mean_f1s))
    knn_best_k = ks[best_i]
    knn_pred, _ = knn_predict(X_train, y_train, X_test, knn_best_k)
    knn_m = confusion_and_metrics(y_test, knn_pred)

    # 2.8 cost-based threshold on training
    best_tau = None
    best_cost = None
    for k in range(1, 20):
        tau = k / 20.0
        pred = (p_train >= tau).astype(int)
        m = confusion_and_metrics(y_train, pred)
        cost = 50 * m["fp"] + 400 * m["fn"]
        if best_cost is None or cost < best_cost or (cost == best_cost and tau < best_tau):
            best_cost = cost
            best_tau = tau

    def cost_at(probs, labels, tau):
        pred = (probs >= tau).astype(int)
        m = confusion_and_metrics(labels, pred)
        return 50 * m["fp"] + 400 * m["fn"]

    cost_train_at_tau = cost_at(p_train, y_train, best_tau)
    cost_test_at_tau = cost_at(p_test, y_test, best_tau)
    cost_test_at_0_5 = cost_at(p_test, y_test, 0.5)
    cost_test_flag_nobody = 50 * 0 + 400 * int(np.sum(y_test == 1))

    items = [
        ("train_rows", train_rows),
        ("train_positives", train_positives),
        ("test_rows", test_rows),
        ("test_positives", test_positives),
        ("loss_epoch_0", losses[0]),
        ("loss_epoch_1", losses[1]),
        ("loss_epoch_10", losses[10]),
        ("loss_epoch_100", losses[100]),
        ("loss_epoch_1000", losses[1000]),
        ("weight_X1", float(w[0])),
        ("weight_X2", float(w[1])),
        ("weight_X3", float(w[2])),
        ("weight_X4", float(w[3])),
        ("weight_X5", float(w[4])),
        ("weight_X6", float(w[5])),
        ("weight_X7", float(w[6])),
        ("bias", float(b)),
        ("test_bce", test_bce),
        ("loss_epoch_200_eta_0.01", lr_losses[0.01]),
        ("loss_epoch_200_eta_0.1", lr_losses[0.1]),
        ("loss_epoch_200_eta_1.0", lr_losses[1.0]),
        ("odds_ratio_X1", odds[0]),
        ("odds_ratio_X2", odds[1]),
        ("odds_ratio_X3", odds[2]),
        ("odds_ratio_X4", odds[3]),
        ("odds_ratio_X5", odds[4]),
        ("odds_ratio_X6", odds[5]),
        ("odds_ratio_X7", odds[6]),
    ]
    for k, v in items_metrics.items():
        items.append((k, v))
    items.extend(
        [
            ("blind_accuracy", blind_accuracy),
            ("knn_best_k", knn_best_k),
            ("knn_tp", knn_m["tp"]),
            ("knn_fp", knn_m["fp"]),
            ("knn_tn", knn_m["tn"]),
            ("knn_fn", knn_m["fn"]),
            ("knn_accuracy", knn_m["accuracy"]),
            ("knn_precision", knn_m["precision"]),
            ("knn_recall", knn_m["recall"]),
            ("knn_f1", knn_m["f1"]),
            ("cost_tau", best_tau),
            ("cost_train_at_tau", cost_train_at_tau),
            ("cost_test_at_tau", cost_test_at_tau),
            ("cost_test_at_0.5", cost_test_at_0_5),
            ("cost_test_flag_nobody", cost_test_flag_nobody),
        ]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results(os.path.join(OUTPUT_DIR, "part2_results.txt"), items)
    write_csv(
        os.path.join(OUTPUT_DIR, "part2_thresholds.csv"),
        ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"],
        thresh_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "part2_knn_cv.csv"),
        ["K", "mean_cv_f1"],
        cv_rows,
    )
    print("Wrote part2 outputs")
    print(
        f"Checkpoint: train_rows={train_rows}, train_positives={train_positives}, "
        f"test_rows={test_rows}, test_positives={test_positives}, "
        f"loss_epoch_0={losses[0]:.4f}, knn_best_k={knn_best_k}, cost_tau={best_tau}"
    )


if __name__ == "__main__":
    main()
