"""Sanity tests: a metric that can't tell an oracle from noise is broken.

Run:  python -m pytest -q tests
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import dedupe, k_core, leave_last_out
from recsys.metrics import rank_metrics, target_ranks

ROOT = Path(__file__).resolve().parents[1]


def _load_original_metric():
    spec = importlib.util.spec_from_file_location("orig_metrics", ROOT / "original" / "metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.precision_recall_at_k


def _toy(n_users=300, n_items=200, seed=0):
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, n_items, n_users)
    seen = [rng.choice(np.setdiff1d(np.arange(n_items), [t]), 3, replace=False) for t in targets]
    return rng, targets, seen, n_items


def test_oracle_scores_perfect():
    rng, targets, seen, n_items = _toy()
    scores = rng.random((len(targets), n_items))
    scores[np.arange(len(targets)), targets] = 10
    m = rank_metrics(target_ranks(scores, targets, seen), 10)
    assert m["HR@10"] == 1.0 and m["NDCG@10"] == 1.0


def test_random_scores_near_chance():
    rng, targets, seen, n_items = _toy(n_users=5000)
    m = rank_metrics(target_ranks(rng.random((len(targets), n_items)), targets, seen), 10)
    chance = 10 / (n_items - 3)
    assert abs(m["HR@10"] - chance) < 0.015


def test_constant_scores_get_no_credit():
    _, targets, seen, n_items = _toy()
    m = rank_metrics(target_ranks(np.zeros((len(targets), n_items)), targets, seen), 10)
    assert m["HR@10"] == 0.0


def test_seen_items_are_excluded():
    scores = np.array([[9.0, 8.0, 1.0, 0.0]])
    # item 2 is the target; items 0 and 1 were seen in training, so it ranks first
    assert target_ranks(scores, np.array([2]), [np.array([0, 1])])[0] == 0


def test_original_metric_is_model_independent():
    """The bug in one assertion: oracle, random and constant scores score identically."""
    prk = _load_original_metric()
    rng = np.random.default_rng(1)
    uid = rng.integers(0, 2000, 3000)
    true = rng.choice([1, 2, 3, 4, 5], 3000, p=[.06, .04, .08, .16, .66])
    results = set()
    for est in (true.astype(float), rng.random(3000), np.zeros(3000)):
        preds = list(zip(uid, range(3000), true, est, [None] * 3000))
        results.add(tuple(round(x, 10) for x in prk(preds, k=10)))
    assert len(results) == 1


def test_split_is_per_user_and_temporal():
    df = pd.DataFrame({
        "reviewerID": list("aaaabbbccc"), "asin": list("wxyzwxywxz"),
        "overall": [5, 4, 3, 2, 5, 5, 1, 4, 4, 2], "verified": True, "vote": 0,
        "review_len": 10, "unixReviewTime": [1, 2, 3, 4, 1, 2, 3, 1, 2, 3]})
    s = leave_last_out(k_core(dedupe(df), 1))
    assert len(s.test) == len(s.val) == 3
    for u in range(3):
        t_tr = s.train[s.train.user == u].unixReviewTime.max()
        assert t_tr <= s.val[s.val.user == u].unixReviewTime.item() <= s.test[s.test.user == u].unixReviewTime.item()


def test_k_core_is_iterative():
    df = pd.DataFrame({"reviewerID": ["a", "a", "a", "b", "b", "b", "c"],
                       "asin": ["x", "y", "z", "x", "y", "q", "x"]})
    out = k_core(df, 2)
    # item z and q have 1 row -> removed -> a and b drop to 2 rows (still >=2); c removed
    assert set(out.reviewerID) == {"a", "b"} and set(out.asin) == {"x", "y"}
