"""Evaluation metrics.

Why the original Precision@10 / Recall@10 were wrong
----------------------------------------------------
The original `precision_recall_at_k` ranked, for each user, only the items that
user happened to rate in the *test split*. Users had ~1.3 test items on average
(max 7), so the "top 10" was always every test item the user had, in any order:

  * Recall@10 was 1 for every user with a relevant test item and 0 otherwise, so
    it just measured the share of users with a >=4-star test rating (~0.84).
  * Precision@10 was (#relevant test items) / 10, ~0.1 for every model.

Random scores, a constant score and the true ratings all give identical numbers
(see tests/test_metrics.py). In deep_segmented.py the user id was passed as
None, so all ~8.7k test rows were pooled as one "user": P@10=0.9, R@10=0.001.

What this module does instead
-----------------------------
`rank_metrics` scores the *entire catalogue* for each test user, removes items
the user already interacted with in train/val, and checks where the held-out
relevant item lands. With one held-out item per user:
    HR@K   = share of users whose held-out item is in the top K
    NDCG@K = mean of 1/log2(rank+1) for hits (0 otherwise)
    Precision@K = HR@K / K, Recall@K = HR@K
A random ranker scores HR@K ~= K / n_items, which makes the numbers interpretable.
"""
import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def target_ranks(scores: np.ndarray, targets: np.ndarray, seen: list) -> np.ndarray:
    """0-based rank of each user's target item among unseen items.

    scores  : (n_users_eval, n_items) model scores, higher = better
    targets : (n_users_eval,) held-out item index per row
    seen    : list of int arrays, items to exclude per row (train+val history)

    Ties are counted pessimistically (a tied item is ranked above the target),
    so a constant scorer cannot get credit by accident.
    """
    scores = scores.astype(np.float64, copy=True)
    rows = np.arange(len(targets))
    tgt = scores[rows, targets].copy()
    for r, s in enumerate(seen):
        scores[r, s] = -np.inf
    scores[rows, targets] = -np.inf  # don't compare target to itself
    return (scores >= tgt[:, None]).sum(axis=1)


def rank_metrics(ranks: np.ndarray, k: int = 10) -> dict:
    hit = ranks < k
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 2), 0.0)
    return {f"HR@{k}": float(hit.mean()), f"NDCG@{k}": float(ndcg.mean()),
            f"Precision@{k}": float(hit.mean() / k), f"Recall@{k}": float(hit.mean())}


def per_user_rank_arrays(ranks: np.ndarray, k: int = 10):
    """Per-user hit and NDCG contributions, for paired bootstrap tests."""
    hit = (ranks < k).astype(float)
    return hit, np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, stat="mean", n: int = 2000,
                     seed: int = 0) -> tuple:
    """95% CI for stat(b) - stat(a), resampling users with replacement.

    stat="mean" for HR/NDCG/MAE contributions, "rmse" for squared errors.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    if stat == "rmse":
        diffs = np.sqrt(b[idx].mean(1)) - np.sqrt(a[idx].mean(1))
        point = np.sqrt(b.mean()) - np.sqrt(a.mean())
    else:
        diffs = b[idx].mean(1) - a[idx].mean(1)
        point = b.mean() - a.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)
