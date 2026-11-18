# Results

Protocol: deduplicated 3-core of `interaction_data.csv` — 7,634 users, 1,300 items, 26,522 ratings. Leave-last-out by time: each user's latest rating is test, previous one is validation. Rating metrics use all 7,634 test users; ranking metrics rank the full catalogue (minus already-seen items) for the 6,234 users whose held-out rating is ≥4. Mean ± std over seeds [0, 1, 2] where the model is stochastic.

## Rating prediction

| model | RMSE | MAE |
|---|---|---|
| Global mean | 1.188 | 0.894 |
| Bias baseline | 1.133 | 0.815 |
| Bias baseline — segmented | 1.250 | 0.775 |
| MF (SVD) | 1.127 ± 0.001 | 0.808 ± 0.001 |
| MF (SVD) — segmented | 1.250 ± 0.001 | 0.772 ± 0.000 |
| MF (SVD) — segmented, leaky clusters | 0.900 ± 0.001 | 0.597 ± 0.000 |
| Neural CF | 1.179 ± 0.005 | 0.730 ± 0.007 |
| Neural CF — segmented | 1.230 ± 0.012 | 0.750 ± 0.029 |
| Neural CF — segmented, leaky clusters | 0.904 ± 0.007 | 0.592 ± 0.017 |

## Top-10 ranking

| model | HR@10 | NDCG@10 | Precision@10 |
|---|---|---|---|
| Global mean | 0.000 | 0.000 | 0.0000 |
| Popularity | 0.293 | 0.160 | 0.0293 |
| Bias baseline | 0.171 | 0.113 | 0.0171 |
| Bias baseline — segmented | 0.198 | 0.131 | 0.0198 |
| MF (SVD) | 0.147 ± 0.003 | 0.089 ± 0.003 | 0.0147 ± 0.0003 |
| MF (SVD) — segmented | 0.060 ± 0.003 | 0.030 ± 0.002 | 0.0060 ± 0.0003 |
| MF (SVD) — segmented, leaky clusters | 0.108 ± 0.004 | 0.053 ± 0.002 | 0.0108 ± 0.0004 |
| Neural CF | 0.057 ± 0.062 | 0.037 ± 0.041 | 0.0057 ± 0.0062 |
| Neural CF — segmented | 0.123 ± 0.087 | 0.073 ± 0.046 | 0.0123 ± 0.0087 |
| Neural CF — segmented, leaky clusters | 0.196 ± 0.008 | 0.112 ± 0.014 | 0.0196 ± 0.0008 |
| BPR-MF | 0.305 ± 0.016 | 0.229 ± 0.008 | 0.0305 ± 0.0016 |
| BPR-MF — segmented | 0.289 ± 0.008 | 0.218 ± 0.004 | 0.0289 ± 0.0008 |
| Random (expected) | 0.008 | — | 0.0008 |

## Segmented vs baseline (paired bootstrap, 95% CI, seed 0)

Negative ΔRMSE and positive ΔHR mean the second model is better.

| comparison | metric | delta | ci_low | ci_high | significant |
|---|---|---|---|---|---|
| Bias baseline — segmented vs Bias baseline | ΔRMSE | 0.1168 | 0.0924 | 0.1417 | True |
| Bias baseline — segmented vs Bias baseline | ΔHR@10 | 0.0271 | 0.0225 | 0.0321 | True |
| MF (SVD) — segmented vs MF (SVD) | ΔRMSE | 0.1219 | 0.0979 | 0.1459 | True |
| MF (SVD) — segmented vs MF (SVD) | ΔHR@10 | -0.0858 | -0.0943 | -0.0776 | True |
| MF (SVD) — segmented, leaky clusters vs MF (SVD) | ΔRMSE | -0.2282 | -0.2466 | -0.2096 | True |
| MF (SVD) — segmented, leaky clusters vs MF (SVD) | ΔHR@10 | -0.0387 | -0.0462 | -0.0306 | True |
| Neural CF — segmented vs Neural CF | ΔRMSE | 0.039 | 0.0263 | 0.0516 | True |
| Neural CF — segmented vs Neural CF | ΔHR@10 | 0.0671 | 0.0603 | 0.0735 | True |
| Neural CF — segmented, leaky clusters vs Neural CF | ΔRMSE | -0.2762 | -0.294 | -0.258 | True |
| Neural CF — segmented, leaky clusters vs Neural CF | ΔHR@10 | 0.0725 | 0.0658 | 0.0792 | True |
| BPR-MF — segmented vs BPR-MF | ΔHR@10 | -0.038 | -0.0481 | -0.0278 | True |
| BPR-MF vs Popularity | ΔHR@10 | 0.0287 | 0.0191 | 0.0388 | True |

## Ablation: original random 80/20 split

Same deduplicated data, but the original protocol's random row split.

| model | mean | std |
|---|---|---|
| MF (SVD) | 0.992 | 0.0009 |
| MF (SVD) — segmented, train-only clusters | 1.0763 | 0.0122 |
| MF (SVD) — segmented, leaky clusters (original) | 0.8446 | 0.009 |

## Segments (train-only features)

| cluster | avg_rating | review_count | avg_review_length | avg_votes | users |
|---|---|---|---|---|---|
| 0 | 2.14 | 1.22 | 461.96 | 4.01 | 1114 |
| 1 | 4.79 | 1.39 | 303.27 | 1.37 | 6061 |
| 2 | 4.2 | 3.27 | 2483.84 | 77.01 | 459 |
