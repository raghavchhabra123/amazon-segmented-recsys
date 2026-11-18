"""Rebuild data/interaction_data.csv from the raw Amazon Reviews v2 dump.

The data is not committed. Download `All_Amazon_Review.json.gz` from
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ (research use; see the
dataset page for terms), then:

    python prepare_data.py /path/to/All_Amazon_Review.json.gz

This reproduces the original project's preprocessing (first 1M reviews, one-pass
>=3 user/item filter). Deduplication and the iterative 3-core are applied later
in recsys/data.py, so the CSV matches the original project's file.
"""
import sys
from pathlib import Path

import pandas as pd

raw = sys.argv[1] if len(sys.argv) > 1 else "All_Amazon_Review.json.gz"
out = Path(__file__).parent / "data" / "interaction_data.csv"
out.parent.mkdir(exist_ok=True)

df = pd.read_json(raw, lines=True, compression="gzip", nrows=1_000_000)
df["review_len"] = df["reviewText"].str.len()
df["vote"] = pd.to_numeric(df["vote"].fillna(0).astype(str).str.replace(",", ""), errors="coerce")
df = df[["reviewerID", "asin", "overall", "verified", "vote", "review_len", "unixReviewTime"]]

uc = df["reviewerID"].value_counts()
df = df[df["reviewerID"].isin(uc[uc >= 3].index)]
ic = df["asin"].value_counts()
df = df[df["asin"].isin(ic[ic >= 3].index)]

df.to_csv(out, index=False)
print(f"wrote {out} ({len(df):,} rows)")
