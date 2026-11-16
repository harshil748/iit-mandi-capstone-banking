Domain: Banking

## How to run

From a clean clone of this repository (Python 3.10 or newer):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

Each Part script regenerates the matching files under `outputs/`. Shared helpers live in `helpers.py` and are imported without re-running any Part.

## Part 1 findings

R1.1 H0 (mean monthly spend = 8.0 thousand INR) was rejected: `z_stat = 2.8930`, which is above the critical value 1.96. A Type I error here would mean telling the planning team that average monthly spend is not INR 8,000 when it actually is. The gap `mean_value − 8.0 = 0.9382` thousand INR (about INR 938) is statistically detectable with n = 800, and for budgeting that size of miss is large enough to matter in practice.

R1.2 `n_required = 1293` next to the 800 rows available: the team still needs a larger sample if it wants the mean estimated to within ±0.5 column units (INR 500) at 95% confidence with the observed SD.

R1.3 `gd_slope_gap = 0.0000`. Gradient descent reached the closed-form line because the MSE surface for a linear model is convex, so full-batch steps with η = 0.1 converge to the unique least-squares solution.

R1.4 `resid_sd_ratio_raw = 3.7525` and `resid_sd_ratio_log = 1.0396`. The raw model shows a clear funnel (residual spread grows with the fitted value). The log-log model largely removes it (ratio near 1).

R1.5 Predictors in Task 1.7 with |coefficient| above 0.05 (excluding the constant) are ln(X4), X2, X3 and X6 (`mr_beta_1`, `mr_beta_3`, `mr_beta_4`, `mr_beta_6`). `mr_adj_r2 = 0.7403` is well above `loglog_adj_r2 = 0.6561`, so the extra predictors earn their place on adjusted R².

## Part 2 findings

R2.1 Ordered from lowest to highest after 200 updates: η = 1.0 (`0.3703`), η = 0.1 (`0.3776`), η = 0.01 (`0.5091`). A learning rate that is too small crawls; a larger rate (here up to 1.0) reached a lower loss in the same budget without blowing up.

R2.2 The three predictors with the largest absolute weight are X3 (credit utilisation), X1 (account age) and X2 (failed transactions). The largest odds ratio is `odds_ratio_X3 = 2.4629`: raising utilisation by one training SD multiplies the odds of default by about 2.46 when other predictors are held fixed.

R2.3 `blind_accuracy = 0.7875` next to `tau_0.5_accuracy = 0.8042`. Accuracy alone is weak because only 51 of 240 test rows are positives, so a “predict nobody defaults” rule already looks strong. Lowering τ from 0.5 to 0.3 raised recall from `0.4706` to `0.7451` and cut precision from `0.5455` to `0.4634`.

R2.4 Logistic at τ = 0.3 has `f1 = 0.5714` and `recall = 0.7451`; KNN (K = 7) has `f1 = 0.3636` and `recall = 0.3137`. Logistic is clearly stronger on both.

R2.5 Deploy the logistic model at the cost-chosen threshold `cost_tau = 0.15` (`cost_test_at_tau = INR 6050`, versus `cost_test_at_0.5 = INR 11800` and `cost_test_flag_nobody = INR 20400`). The threshold was chosen on training rows so the test set stays an honest check and is not used to tune the alert rule.

## Part 3 findings

R3.1 Successive WCSS drops: WCSS(2)−WCSS(3) = 659.8282, WCSS(3)−WCSS(4) = 108.0739, WCSS(4)−WCSS(5) = 171.0575, WCSS(5)−WCSS(6) = 45.5466. The elbow rule supports K = 3 because the drop from 3 to 4 is less than half the drop from 2 to 3. The later drops do not shrink steadily; the pinned starting rows give a reproducible solution, not necessarily the global optimum.

R3.2 Segment names and actions (from `part3_segments.csv`):
- A — Steady low-utilisation borrowers (size 322, low utilisation, long tenure, high repayment score): keep light monitoring and autopay nudges.
- B — High-inflow active users (size 255, higher spend and failed-txn count): watch failed payments and utilisation before raising limits.
- C — New high-utilisation borrowers (size 223, short tenure, highest utilisation): earlier collections contact and tighter credit checks.
Segment C has the highest `outcome_rate = 0.4664`.

R3.3 At the cut, single linkage yields 4 clusters: {E1}, {E4}, {E5}, {E2+E3+E6}. Complete linkage yields 2 clusters: {E1+E4+E5} and {E2+E3+E6}. Single linkage chains through small nearest-neighbour links and keeps E1, E4 and E5 separate longer, while complete linkage forces every pair inside a cluster to be close and therefore merges the E1–E4–E5 side earlier relative to the final gap.

R3.4 After standardising, the mean vector is the zero vector. PC1 is driven mainly by X1 and X5 positively and by X3 and X2 negatively (loadings `0.5317`, `0.4418`, `-0.5187`, `-0.4100`). Three components keep at least 80% of the variance (`components_for_80pct = 3`). Mean PC1 falls from segment A (`1.7773`) through B (`-0.7831`) to C (`-1.6709`), so PC1 tracks the same risk/tenure axis as the ordered segments.

## Part 4 findings

R4.1 The best split is S4 (`X1 ≤ 24`) because it has the highest information gain (`0.1252`) and the largest Gini gain (`0.0606`): newer accounts concentrate far more defaults on the “yes” side. A tree grown until every leaf is pure would overfit noise in a single 560-row training set. One rule that stops a tree from growing too deep is a minimum leaf size (or a maximum depth).

R4.2 Ship the logistic model at τ = 0.3 (`f1 = 0.5714`, `recall = 0.7451`). Hard vote (`f1 = 0.5385`, `recall = 0.6863`) and soft vote (`f1 = 0.5294`, `recall = 0.7059`) did not beat it; combining with weaker KNN and stump signals did not help on this test set.

R4.3 After round 1 normalisation, the two misclassified points (3 and 7) each carry weight 0.2500, so their total weight is 0.5000. Round 2 trusts h2 more than h1 because h2’s weighted error is lower (`0.1667` vs `0.2500`), which produces a larger α (`0.8047` vs `0.5493`).

R4.4 CNN option 2 shapes and parameters:

| Layer | Output shape | Parameters |
|---|---|---|
| conv1 (5×5, 6, pad 2) | 28 × 28 × 6 | 156 |
| max-pool 2×2 | 14 × 14 × 6 | 0 |
| conv2 (5×5, 16, pad 0) | 10 × 10 × 16 | 2416 |
| max-pool 2×2 | 5 × 5 × 16 | 0 |
| dense1 (32) | 32 | 12832 |
| dense2 (10) | 10 | 330 |

For Model B, W_xh is 128 × 64, W_hh is 128 × 128, and W_hy is 16 × 128. `rnn_total` does not grow with sequence length because the same weight matrices are reused at every time step. For 10 digit classes I would use ReLU in hidden layers, softmax on the output, and categorical cross-entropy. A very deep plain network can stop learning in early layers because gradients shrink as they are multiplied through many layers; a skip connection gives gradients a shorter path and lets the block learn a residual update instead of a full remap.

## Model-choice memo

To: Head of lending-risk analytics, PhonePe (synthetic study)

Recommend deploying the from-scratch logistic regression on the seven standardised predictors, with alert threshold τ = 0.15 chosen by the training cost rule (INR 50 per false alarm, INR 400 per missed default). On the held-out 240 rows this threshold costs INR 6050, versus INR 11800 at τ = 0.5 and INR 20400 if nobody is flagged (`cost_test_at_tau`, `cost_test_at_0.5`, `cost_test_flag_nobody`). At the evaluation threshold τ = 0.3 the same model reaches recall 0.7451 and F1 0.5714, ahead of KNN (K = 7) and of hard/soft votes that mix in weaker stump and neighbour signals. One risk: only 51 positive test cases and a single pinned split, so the cost ranking could shift on another draw of data. Build neither Model A (cheque OCR) nor Model B (remark text) before this structured alert is in production; both are later sizing exercises, not substitutes for the tabular default screen.

## Limitations

1. The data are fully synthetic; nothing here reflects PhonePe’s real borrowers, products or loss rates.
2. Evaluation uses one pinned 560/240 stratified split, so threshold and K choices are tied to that split.
3. Only 51 positive test rows makes precision, recall and cost estimates noisy.
4. Models are linear or local (logistic, KNN, stump); interactions beyond what the features already encode are not explored.
5. Cost weights (INR 50 / INR 400) are scenario assumptions; a different loss ratio would move `cost_tau`.
