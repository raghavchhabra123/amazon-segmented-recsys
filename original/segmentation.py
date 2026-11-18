# segmentation.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("=================================")
print("USER SEGMENTATION (KMEANS)")
print("=================================")


# -----------------------------
# Load dataset
# -----------------------------

print("\nLoading interaction dataset...")

df = pd.read_csv("interaction_data.csv")

print("Dataset size:", df.shape)


# -----------------------------
# Build user behavior features
# -----------------------------

print("\nCreating user behavior features...")

user_features = df.groupby("reviewerID").agg({

    "overall": ["mean", "count"],
    "review_len": "mean",
    "vote": "mean"

})

user_features.columns = [
    "avg_rating",
    "review_count",
    "avg_review_length",
    "avg_votes"
]

user_features = user_features.fillna(0)

print("User feature table:", user_features.shape)


# -----------------------------
# Normalize features
# -----------------------------

print("\nNormalizing features...")

scaler = StandardScaler()

X = scaler.fit_transform(user_features)


# -----------------------------
# KMeans clustering
# -----------------------------

print("\nRunning KMeans clustering...")

k = 4

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

user_features["cluster"] = clusters


# -----------------------------
# Remove tiny clusters
# -----------------------------

print("\nChecking cluster sizes...")

cluster_counts = user_features["cluster"].value_counts()

print("\nOriginal cluster sizes")
print(cluster_counts)

# remove clusters that are too small to train recommenders
valid_clusters = cluster_counts[cluster_counts > 50].index

user_features = user_features[user_features["cluster"].isin(valid_clusters)]

print("\nFiltered cluster sizes")
print(user_features["cluster"].value_counts())


print("\nValid clusters used for modeling:")
print(sorted(user_features["cluster"].unique()))


# -----------------------------
# Save cluster assignments
# -----------------------------

cluster_map = user_features.reset_index()[["reviewerID", "cluster"]]

cluster_map.to_csv("user_clusters.csv", index=False)

print("\nSaved user_clusters.csv")


# -----------------------------
# Cluster behavior summary
# -----------------------------

print("\nCluster behavior summary")

summary = user_features.groupby("cluster").mean()

print(summary)


print("\n=================================")
print("Segmentation complete.")
print("=================================")