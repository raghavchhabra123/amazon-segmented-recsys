# data_prep.py

from helpers import load_data, basic_clean
import pandas as pd


print("Loading dataset...")

df = load_data("All_Amazon_Review.json.gz", nrows=1000000)
df = basic_clean(df)

print("Original size:", df.shape)

df = df[["reviewerID", "asin", "overall", "verified", "vote", "review_len", "unixReviewTime"]]

# Filter sparse users/items
user_counts = df["reviewerID"].value_counts()
df = df[df["reviewerID"].isin(user_counts[user_counts >= 3].index)]

item_counts = df["asin"].value_counts()
df = df[df["asin"].isin(item_counts[item_counts >= 3].index)]

print("Filtered size:", df.shape)

df.to_csv("interaction_data.csv", index=False)

print("Saved interaction_data.csv")