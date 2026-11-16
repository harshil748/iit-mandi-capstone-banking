"""Shared helpers for Capstone Parts 1–4. Importing this module does not run any Part."""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

DATA_PATH = os.path.join("data", "records.csv")
OUTPUT_DIR = "outputs"


def load_records(path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X (n×7), y outcome (n,), value (n,) from records.csv in file order."""
    rows: List[List[float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    float(row["account_age_months"]),
                    float(row["failed_txn_count_90d"]),
                    float(row["credit_utilisation_ratio"]),
                    float(row["avg_monthly_inflow_inr_thousands"]),
                    float(row["repayment_score"]),
                    float(row["irregular_salary_flag"]),
                    float(row["autopay_flag"]),
                    float(row["loan_default"]),
                    float(row["monthly_spend_inr_thousands"]),
                ]
            )
    arr = np.asarray(rows, dtype=float)
    X = arr[:, :7]
    y = arr[:, 7].astype(int)
    value = arr[:, 8]
    return X, y, value


def sample_mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    return float(np.sum(a) / a.size)


def sample_var(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    m = sample_mean(a)
    return float(np.sum((a - m) ** 2) / (n - 1))


def sample_sd(a: np.ndarray) -> float:
    return float(np.sqrt(sample_var(a)))


def sample_cov(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    return float(np.sum((x - sample_mean(x)) * (y - sample_mean(y))) / (n - 1))


def fmt_float(v: float) -> str:
    return f"{float(v):.4f}"


def write_results(path: str, items: Sequence[Tuple[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    for key, value in items:
        if isinstance(value, bool):
            text = "yes" if value else "no"
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            text = str(int(value))
        elif isinstance(value, (float, np.floating)):
            text = fmt_float(value)
        elif isinstance(value, str):
            if value in ("yes", "no") or value.startswith("S") or value.startswith("E"):
                text = value
            else:
                # numeric-looking strings already formatted, or text labels
                text = value
        else:
            text = str(value)
        lines.append(f"{key} = {text}")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(path: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            out = []
            for cell in row:
                if isinstance(cell, (float, np.floating)):
                    out.append(fmt_float(cell))
                elif isinstance(cell, (int, np.integer)) and not isinstance(cell, (bool, np.bool_)):
                    out.append(int(cell))
                else:
                    out.append(cell)
            w.writerow(out)


def stratified_split(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pinned 70/30 stratified split; returns train_idx, test_idx in file order within each set."""
    y = np.asarray(y).astype(int)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        n_train = (7 * idx.size) // 10
        train_parts.append(idx[:n_train])
        test_parts.append(idx[n_train:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    return train_idx, test_idx


def standardise_train_test(
    X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_raw = X[train_idx]
    means = np.array([sample_mean(X_train_raw[:, j]) for j in range(X.shape[1])])
    sds = np.array([sample_sd(X_train_raw[:, j]) for j in range(X.shape[1])])
    X_train = (X_train_raw - means) / sds
    X_test = (X[test_idx] - means) / sds
    return X_train, X_test, means, sds


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    eta: float = 0.1,
    n_updates: int = 1000,
    record_at: Sequence[int] | None = None,
) -> Tuple[np.ndarray, float, Dict[int, float]]:
    """Full-batch logistic regression from zero init. Returns weights, bias, loss_at."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    record_set = set(record_at or [])
    losses: Dict[int, float] = {}
    p0 = sigmoid(X @ w + b)
    if 0 in record_set:
        losses[0] = binary_cross_entropy(y, p0)
    for t in range(1, n_updates + 1):
        p = sigmoid(X @ w + b)
        err = p - y
        grad_w = (X.T @ err) / n
        grad_b = float(np.sum(err) / n)
        w = w - eta * grad_w
        b = b - eta * grad_b
        if t in record_set:
            losses[t] = binary_cross_entropy(y, sigmoid(X @ w + b))
    return w, b, losses


def predict_proba_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "fpr": fpr,
        "tpr": rec,
        "f1": f1,
    }


def knn_predict(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_query: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return binary predictions and positive-neighbour shares for each query."""
    X_ref = np.asarray(X_ref, dtype=float)
    y_ref = np.asarray(y_ref).astype(int).ravel()
    X_query = np.asarray(X_query, dtype=float)
    preds = np.zeros(X_query.shape[0], dtype=int)
    shares = np.zeros(X_query.shape[0], dtype=float)
    threshold = (k + 1) / 2.0
    for i in range(X_query.shape[0]):
        diff = X_ref - X_query[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        order = np.argsort(dist, kind="stable")
        neigh = order[:k]
        pos = int(np.sum(y_ref[neigh]))
        shares[i] = pos / float(k)
        preds[i] = 1 if pos >= threshold else 0
    return preds, shares


def train_logistic_and_knn():
    """Fit Part 2 models once for reuse by Part 4. Returns a dict of artefacts."""
    X, y, _ = load_records()
    train_idx, test_idx = stratified_split(y)
    X_train, X_test, means, sds = standardise_train_test(X, train_idx, test_idx)
    y_train = y[train_idx]
    y_test = y[test_idx]

    w, b, _ = fit_logistic(
        X_train, y_train, eta=0.1, n_updates=1000, record_at=[0, 1, 10, 100, 1000]
    )
    p_test = predict_proba_logistic(X_test, w, b)
    p_train = predict_proba_logistic(X_train, w, b)

    # KNN CV for best K
    ks = list(range(1, 22, 2))
    mean_f1s = []
    n_train = X_train.shape[0]
    folds = np.arange(n_train) % 5
    for k in ks:
        fold_f1 = []
        for fold in range(5):
            val_mask = folds == fold
            ref_mask = ~val_mask
            pred, _ = knn_predict(X_train[ref_mask], y_train[ref_mask], X_train[val_mask], k)
            m = confusion_and_metrics(y_train[val_mask], pred)
            fold_f1.append(m["f1"])
        mean_f1s.append(float(np.mean(fold_f1)))
    best_i = int(np.argmax(mean_f1s))
    # ties -> smaller K: argmax already prefers first occurrence
    best_k = ks[best_i]
    knn_pred, knn_share = knn_predict(X_train, y_train, X_test, best_k)

    return {
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "means": means,
        "sds": sds,
        "w": w,
        "b": b,
        "p_train": p_train,
        "p_test": p_test,
        "knn_best_k": best_k,
        "knn_mean_f1s": list(zip(ks, mean_f1s)),
        "knn_pred": knn_pred,
        "knn_share": knn_share,
    }
