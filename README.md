# Does user segmentation help recommenders? An audit and rebuild

A rebuild of a Northwestern MLDS Data Mining group project (Winter 2026), which
tested whether training recommenders separately on K-means user clusters beats
one global model, using Amazon reviews. The original concluded that segmentation
improved rating prediction (SVD RMSE 0.996 → 0.826). This rebuild finds that the
improvement came from **test-set leakage**. Once the leak is removed,
segmentation makes rating prediction worse. It also replaces the ranking metric,
which could not tell a perfect model from random guessing.

> **Credit.** The original project was a group submission by Raghav Chhabra,
> Ben Maisel, Anuraj Nair, Amy Wang and Keyu Yan. This repo is Raghav's separate
> post-course audit and rebuild. `original/` keeps copies of the audited files and
> run logs from the group project, unchanged, for reference.

## What was wrong in the original

| # | Issue | Effect |
|---|---|---|
| 1 | **Precision@10 / Recall@10 only re-sorted each user's own test items** (~1.3 per user, max 7). The "top 10" always contained all of them. | Random scores, a constant and the true ratings all score P@10 = 0.107, R@10 = 0.845. The numbers measured the data, not the model (`tests/test_metrics.py::test_original_metric_is_model_independent`). |
| 2 | `deep_segmented.py` passed `uid=None`, pooling all ~8.7k test rows as one "user". | P@10 = 0.90, R@10 = 0.001, reported as a "precision anomaly". |
| 3 | **Cluster features (avg rating, etc.) were computed on all reviews, including test reviews.** | The cluster label leaks the test rating. This alone produces the reported RMSE gain (see ablation). |
| 4 | 3,981 duplicate (user, item) rows (9%). | The same rating could land in both train and test. |
| 5 | The ≥3-interaction filter ran once, not to convergence. | 4,185 users still had fewer than 3 rows. |
| 6 | Each model used a different split (Surprise 80/20, sklearn 70/15/15, per-cluster splits). | The models weren't compared on the same test rows. |
| 7 | NCF had no bias terms and logged *summed* batch loss. | The epoch-1 loss of 426 was the MLP learning the ~4.4 rating offset, and the logged losses were not comparable between models or clusters. |

## Fixed protocol

* Collapse duplicates, then take a true 3-core: 26,522 ratings, 7,634 users, 1,300 items.
* **Leave-last-out by time.** Each user's latest review is test, the one before is
  validation, the rest is training. Every model and every segmented variant sees
  exactly the same rows.
* Segments are fit on **training data only**. Tiny clusters are merged into the
  nearest large one instead of dropping users from evaluation.
* **Ranking.** Score the full catalogue, drop items the user has already seen, and
  check where the held-out item lands (HR@10, NDCG@10). This is evaluated for the
  6,234 users whose held-out rating is ≥4. Random ranking gets HR@10 ≈ 0.008.
* Three seeds for stochastic models. Paired bootstrap 95% CIs over users for
  segmented-vs-baseline differences.
* Added baselines a reviewer would ask for: global mean, user/item bias,
  popularity, and BPR-MF (a model trained for ranking rather than rating).

## Results (`results/RESULTS.md` has everything)

**Rating prediction (RMSE, lower is better)**

| model | global | segmented (train-only clusters) | segmented (leaky clusters, as original) |
|---|---|---|---|
| Bias baseline | 1.133 | 1.250 | — |
| MF / SVD | **1.127** | 1.250 | 0.900 |
| Neural CF | 1.179 | 1.230 | 0.904 |

**Ablation on the original random 80/20 split:** SVD 0.992, leaky-segmented SVD
0.845 (the original report had 0.996 → 0.826), clean-segmented SVD 1.076.
The leak reproduces the reported gain. Without it, segmentation is worse.

**Why segmentation hurts:** with ~1.5 training ratings per user, a user's average
rating is a noisy estimate. The "critical reviewer" cluster averaged 2.2 stars in
training but 3.5 on the held-out review (regression to the mean). A per-cluster
model locks in the low average. A global model with regularised user biases
shrinks it toward the overall mean. Segmented models do get slightly lower *MAE*
(0.77 vs 0.81), because they predict the dominant 5-star rating more often. In
exchange they make larger misses, which RMSE penalises.

**Top-10 ranking (full catalogue)**

| model | HR@10 | NDCG@10 |
|---|---|---|
| Random | 0.008 | — |
| Popularity | 0.293 | 0.160 |
| MF / SVD (rating loss) | 0.147 | 0.089 |
| **BPR-MF (ranking loss)** | **0.305** | **0.229** |
| BPR-MF, segmented | 0.289 | 0.218 |

Models trained to predict ratings rank poorly. Here they lose to simply
recommending the most popular items. BPR-MF clearly beats popularity on
NDCG@10 (0.223–0.238 on every seed vs 0.160, about +43%), so it puts the right
item higher up. Its HR@10 lead is small and not robust: 0.322 / 0.302 / 0.291
across seeds vs 0.293. The bootstrap CI in `results/` (+0.029, [+0.019, +0.039])
is from seed 0 only. Segmenting BPR-MF hurts (ΔHR@10 −0.038 on seed 0;
segmented is lower on 2 of 3 seeds). Neural CF ranking is unstable across
seeds (HR@10 0.003–0.13), so we draw no ranking conclusion from it.

## Limitations

* The data is the first 1M lines of the 36 GB Amazon v2 dump, not a random
  sample. It covers only 1,300 items after filtering, so results may not
  generalise to the full catalogue.
  The 3-core is sparse (~1.5 training ratings per user), which favours simple
  models.
* Hyperparameters are the original project's for SVD. NCF and BPR use light
  defaults with early stopping on validation. No large tuning sweep.
* One held-out item per user means P@10 is just HR@10 ÷ 10. HR and NDCG are the
  meaningful numbers.

## Run it

The review data is not redistributed here. Download `All_Amazon_Review.json.gz`
(~36 GB) from the [Amazon Reviews v2 page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
and accept its terms. Then:

```bash
pip install -r requirements.txt
python prepare_data.py /path/to/All_Amazon_Review.json.gz   # ~15 s; writes data/interaction_data.csv
python run_experiments.py          # ~2 min on a laptop CPU; writes results/
python -m pytest -q tests          # metric and split sanity tests (no data needed)
```

`prepare_data.py` reproduces the group project's preprocessed sample exactly
(43,834 rows, checked against the original file).

## Layout

```
prepare_data.py         raw dump → data/interaction_data.csv
recsys/data.py          dedupe, iterative k-core, leave-last-out split
recsys/metrics.py       RMSE/MAE, full-catalogue HR/NDCG, paired bootstrap
recsys/segmentation.py  K-means on train-only user features (leaky=True reproduces the bug)
recsys/models.py        GlobalMean, Popularity, BiasBaseline, BiasedMF (SVD), NeuralCF, BPRMF, Segmented
run_experiments.py      every model × seeds → results/
tests/                  sanity tests, incl. one proving the original metric is model-independent
original/               the audited files and original run logs
```
