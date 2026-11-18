"""Data loading, cleaning and a leakage-safe train/val/test split.

Fixes relative to the original pipeline:
  * duplicate (user, item) pairs are collapsed (the original kept ~4k of them,
    so the same user-item rating could sit in both train and test);
  * the >=3 interaction filter is applied iteratively (a true 3-core), because a
    single pass left ~4k users with fewer than 3 rows;
  * the split is leave-last-out per user by timestamp: each user's most recent
    interaction is test, the one before it is validation, the rest is train.
    Every evaluated user therefore has training history, and nothing from the
    future leaks into the past.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    user_ids: pd.Index  # code -> reviewerID
    item_ids: pd.Index  # code -> asin


def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["reviewerID", "asin", "overall", "verified", "vote",
               "review_len", "unixReviewTime"]].copy()


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent rating for each (user, item) pair."""
    df = df.sort_values(["reviewerID", "asin", "unixReviewTime"])
    return df.drop_duplicates(["reviewerID", "asin"], keep="last")


def k_core(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Repeat the user/item count filter until every user and item has >= k rows."""
    while True:
        uc = df["reviewerID"].value_counts()
        ic = df["asin"].value_counts()
        keep = df["reviewerID"].map(uc).ge(k) & df["asin"].map(ic).ge(k)
        if keep.all():
            return df
        df = df[keep]


def leave_last_out(df: pd.DataFrame, seed: int = 0) -> Split:
    df = df.copy()
    df["user"] = df["reviewerID"].astype("category").cat.codes.astype(np.int64)
    df["item"] = df["asin"].astype("category").cat.codes.astype(np.int64)
    user_ids = df["reviewerID"].astype("category").cat.categories
    item_ids = df["asin"].astype("category").cat.categories

    # Ties on day-resolution timestamps are broken randomly but reproducibly.
    rng = np.random.default_rng(seed)
    df["_tiebreak"] = rng.random(len(df))
    df = df.sort_values(["user", "unixReviewTime", "_tiebreak"])
    pos_from_end = df.groupby("user").cumcount(ascending=False)

    test = df[pos_from_end == 0].drop(columns="_tiebreak")
    val = df[pos_from_end == 1].drop(columns="_tiebreak")
    train = df[pos_from_end >= 2].drop(columns="_tiebreak")
    return Split(train.reset_index(drop=True), val.reset_index(drop=True),
                 test.reset_index(drop=True), len(user_ids), len(item_ids),
                 user_ids, item_ids)


def prepare(path: str, k: int = 3, seed: int = 0) -> Split:
    return leave_last_out(k_core(dedupe(load_interactions(path)), k), seed)
