# metrics.py

from collections import defaultdict


def precision_recall_at_k(predictions, k=10, threshold=4):
    """
    Compute Precision@K and Recall@K for Surprise predictions.

    predictions : list of Surprise predictions
    k           : number of recommendations
    threshold   : rating threshold for relevance
    """

    user_est_true = defaultdict(list)

    # collect predictions per user
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():

        # sort by predicted rating
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # number of relevant items in test set
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # top K recommendations
        top_k = user_ratings[:k]

        # relevant items in top K
        n_rel_and_rec_k = sum(
            (true_r >= threshold) for (_, true_r) in top_k
        )

        # Precision@K
        precisions[uid] = n_rel_and_rec_k / k

        # Recall@K
        recalls[uid] = (
            n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        )

    # average across users
    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)

    return precision, recall