"""K-means user segmentation.

The original segmentation computed each user's average rating (and other
features) over ALL of their reviews, including the ones later used as test
ratings. Because "average rating" is the dominant clustering feature, the cluster
label leaks the test rating: a per-cluster model is partly being told the answer.

`fit_segments(..., leaky=False)` builds features from the training split only.
`leaky=True` reproduces the original behaviour so the effect can be measured.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["avg_rating", "review_count", "avg_review_length", "avg_votes"]


def user_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.groupby("user").agg(avg_rating=("overall", "mean"),
                               review_count=("overall", "size"),
                               avg_review_length=("review_len", "mean"),
                               avg_votes=("vote", "mean"))
    return f.fillna(0)


def fit_segments(train: pd.DataFrame, n_users: int, k: int = 4, min_size: int = 50,
                 leaky: bool = False, all_rows: pd.DataFrame = None,
                 seed: int = 42) -> np.ndarray:
    """Return a cluster label for every user code 0..n_users-1.

    Clusters smaller than `min_size` users are merged into the nearest large
    cluster (the original dropped those users, which changes the test set).
    """
    src = all_rows if leaky else train
    feats = user_features(src).reindex(range(n_users)).fillna(0)
    X = StandardScaler().fit_transform(feats[FEATURES])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    labels = km.labels_.copy()

    sizes = np.bincount(labels, minlength=k)
    big = np.flatnonzero(sizes >= min_size)
    small = np.flatnonzero(sizes < min_size)
    if len(small):
        d = ((X[:, None, :] - km.cluster_centers_[None, big, :]) ** 2).sum(-1)
        nearest_big = big[d.argmin(1)]
        mask = np.isin(labels, small)
        labels[mask] = nearest_big[mask]
    # relabel 0..m-1
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def describe(train: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    f = user_features(train)
    f["cluster"] = labels[f.index]
    out = f.groupby("cluster")[FEATURES].mean()
    out["users"] = f.groupby("cluster").size()
    return out.round(2)
