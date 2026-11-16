"""Part 1 — Estimation, inference and a regression baseline."""
from __future__ import annotations

import math
import os

import numpy as np

from helpers import (
    OUTPUT_DIR,
    load_records,
    sample_cov,
    sample_mean,
    sample_sd,
    sample_var,
    write_results,
)


def closed_form_simple(x: np.ndarray, y: np.ndarray):
    slope = sample_cov(x, y) / sample_var(x)
    intercept = sample_mean(y) - slope * sample_mean(x)
    fitted = intercept + slope * x
    y_bar = sample_mean(y)
    tss = float(np.sum((y - y_bar) ** 2))
    ess = float(np.sum((fitted - y_bar) ** 2))
    rss = float(np.sum((y - fitted) ** 2))
    return slope, intercept, fitted, tss, ess, rss


def main() -> None:
    X, _, value = load_records()
    x4 = X[:, 3]
    n = value.size

    # 1.1
    mean_value = sample_mean(value)
    sd_value = sample_sd(value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # 1.2
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # 1.3
    E = 0.5
    n_required = int(math.ceil((1.96 * sd_value / E) ** 2))

    # 1.4
    reg_slope, reg_intercept, fitted, tss, ess, rss = closed_form_simple(x4, value)
    tss_equals = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(value - fitted)))

    # 1.5 gradient descent on standardised X4
    x4_mean = sample_mean(x4)
    x4_sd = sample_sd(x4)
    z = (x4 - x4_mean) / x4_sd
    m = 0.0
    c = 0.0
    eta = 0.1
    J_at = {}

    def cost(m_, c_):
        e = value - (m_ * z + c_)
        return float(np.mean(e ** 2))

    J_at[0] = cost(m, c)
    for t in range(1, 501):
        e = value - (m * z + c)
        dJ_dm = - (2.0 / n) * float(np.sum(z * e))
        dJ_dc = - (2.0 / n) * float(np.sum(e))
        m = m - eta * dJ_dm
        c = c - eta * dJ_dc
        if t in (1, 10, 100, 500):
            J_at[t] = cost(m, c)

    gd_slope = m / x4_sd
    gd_intercept = c - m * x4_mean / x4_sd
    gd_slope_gap = abs(gd_slope - reg_slope)

    # 1.6 funnel check + log-log
    order_raw = np.argsort(fitted, kind="stable")
    resid = value - fitted
    lower = resid[order_raw[:400]]
    upper = resid[order_raw[400:]]
    resid_sd_lower_raw = sample_sd(lower)
    resid_sd_upper_raw = sample_sd(upper)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    ln_x4 = np.log(x4)
    ln_y = np.log(value)
    loglog_slope, loglog_intercept, fitted_ll, tss_ll, ess_ll, rss_ll = closed_form_simple(
        ln_x4, ln_y
    )
    loglog_r2 = 1.0 - rss_ll / tss_ll
    loglog_rmse = math.sqrt(rss_ll / n)
    order_ll = np.argsort(fitted_ll, kind="stable")
    resid_ll = ln_y - fitted_ll
    resid_sd_lower_log = sample_sd(resid_ll[order_ll[:400]])
    resid_sd_upper_log = sample_sd(resid_ll[order_ll[400:]])
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    # 1.7 multiple regression on ln(value)
    ones = np.ones((n, 1))
    design = np.column_stack(
        [
            ones.ravel(),
            ln_x4,
            X[:, 0],
            X[:, 1],
            X[:, 2],
            X[:, 4],
            X[:, 5],
            X[:, 6],
        ]
    )
    # beta = (X'X)^{-1} X'Y via linear solve
    xtx = design.T @ design
    xty = design.T @ ln_y
    beta = np.linalg.solve(xtx, xty)
    fitted_mr = design @ beta
    y_bar_ln = sample_mean(ln_y)
    tss_mr = float(np.sum((ln_y - y_bar_ln) ** 2))
    rss_mr = float(np.sum((ln_y - fitted_mr) ** 2))
    mr_r2 = 1.0 - rss_mr / tss_mr
    p = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p - 1)

    items = [
        ("n", n),
        ("mean_value", mean_value),
        ("sd_value", sd_value),
        ("se_value", se_value),
        ("ci95_low", ci95_low),
        ("ci95_high", ci95_high),
        ("z_stat", z_stat),
        ("reject_h0", reject_h0),
        ("n_required", n_required),
        ("reg_slope", reg_slope),
        ("reg_intercept", reg_intercept),
        ("tss", tss),
        ("ess", ess),
        ("rss", rss),
        ("tss_equals_ess_plus_rss", tss_equals),
        ("r2", r2),
        ("rmse", rmse),
        ("mae", mae),
        ("gd_J0", J_at[0]),
        ("gd_J1", J_at[1]),
        ("gd_J10", J_at[10]),
        ("gd_J100", J_at[100]),
        ("gd_J500", J_at[500]),
        ("gd_slope", gd_slope),
        ("gd_intercept", gd_intercept),
        ("gd_slope_gap", gd_slope_gap),
        ("resid_sd_lower_raw", resid_sd_lower_raw),
        ("resid_sd_upper_raw", resid_sd_upper_raw),
        ("resid_sd_ratio_raw", resid_sd_ratio_raw),
        ("loglog_intercept", loglog_intercept),
        ("loglog_slope", loglog_slope),
        ("loglog_r2", loglog_r2),
        ("loglog_rmse", loglog_rmse),
        ("resid_sd_lower_log", resid_sd_lower_log),
        ("resid_sd_upper_log", resid_sd_upper_log),
        ("resid_sd_ratio_log", resid_sd_ratio_log),
        ("mr_beta_0", float(beta[0])),
        ("mr_beta_1", float(beta[1])),
        ("mr_beta_2", float(beta[2])),
        ("mr_beta_3", float(beta[3])),
        ("mr_beta_4", float(beta[4])),
        ("mr_beta_5", float(beta[5])),
        ("mr_beta_6", float(beta[6])),
        ("mr_beta_7", float(beta[7])),
        ("mr_r2", mr_r2),
        ("mr_adj_r2", mr_adj_r2),
        ("loglog_adj_r2", loglog_adj_r2),
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_results(os.path.join(OUTPUT_DIR, "part1_results.txt"), items)
    print("Wrote outputs/part1_results.txt")
    print(f"Checkpoint: n={n}, mean_value={mean_value:.4f}, gd_slope_gap={gd_slope_gap:.6f}")


if __name__ == "__main__":
    main()
