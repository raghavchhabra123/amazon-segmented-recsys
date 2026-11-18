"""Run every model under one protocol and write results/ tables.

    python run_experiments.py            # ~2-4 min on a laptop CPU

Outputs:
    results/rating.csv, results/ranking.csv, results/significance.csv,
    results/random_split_ablation.csv, results/segments.csv, results/RESULTS.md
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.data import dedupe, k_core, load_interactions, prepare
from recsys.metrics import (mae, paired_bootstrap, per_user_rank_arrays, rank_metrics,
                            rmse, target_ranks)
from recsys.models import (BiasBaseline, BiasedMF, BPRMF, GlobalMean, NeuralCF,
                           Popularity, Segmented)
from recsys.segmentation import describe, fit_segments

torch.set_num_threads(4)
ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2]


def to_md(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)

K = 10

split = prepare(ROOT / "data" / "interaction_data.csv", k=3, seed=0)
tr, va, te = split.train, split.val, split.test
NU, NI = split.n_users, split.n_items
all_rows = pd.concat([tr, va, te])

# held-out users whose last item is "relevant" (>=4 stars) are the ranking test set
rel = te[te["overall"] >= 4]
hist = pd.concat([tr, va]).groupby("user")["item"].apply(np.asarray)
rel_users, rel_targets = rel["user"].values, rel["item"].values
rel_seen = [hist[u] for u in rel_users]

labels_clean = fit_segments(tr, NU, leaky=False)
labels_leaky = fit_segments(tr, NU, leaky=True, all_rows=all_rows)
describe(tr, labels_clean).to_csv(OUT / "segments.csv")

print(f"users={NU} items={NI} train={len(tr)} val={len(va)} test={len(te)} "
      f"ranking-test users={len(rel)}")


def evaluate(model):
    out, per_user = {}, {}
    if model.rates:
        p = model.predict(te["user"].values, te["item"].values)
        out["RMSE"], out["MAE"] = rmse(te["overall"], p), mae(te["overall"], p)
        per_user["sq"] = (te["overall"].values - p) ** 2
    ranks = target_ranks(model.score_all(rel_users), rel_targets, rel_seen)
    out.update(rank_metrics(ranks, K))
    per_user["hit"], per_user["ndcg"] = per_user_rank_arrays(ranks, K)
    return out, per_user


specs = [  # (name, factory(seed), seeded?)
    ("Global mean", lambda s: GlobalMean(), False),
    ("Popularity", lambda s: Popularity(), False),
    ("Bias baseline", lambda s: BiasBaseline(), False),
    ("Bias baseline — segmented", lambda s: Segmented(BiasBaseline, labels_clean), False),
    ("MF (SVD)", lambda s: BiasedMF(seed=s), True),
    ("MF (SVD) — segmented", lambda s: Segmented(lambda: BiasedMF(seed=s), labels_clean), True),
    ("MF (SVD) — segmented, leaky clusters", lambda s: Segmented(lambda: BiasedMF(seed=s), labels_leaky), True),
    ("Neural CF", lambda s: NeuralCF(seed=s), True),
    ("Neural CF — segmented", lambda s: Segmented(lambda: NeuralCF(seed=s), labels_clean), True),
    ("Neural CF — segmented, leaky clusters", lambda s: Segmented(lambda: NeuralCF(seed=s), labels_leaky), True),
    ("BPR-MF", lambda s: BPRMF(seed=s), True),
    ("BPR-MF — segmented", lambda s: Segmented(lambda: BPRMF(seed=s), labels_clean), True),
]

rows, per_user_seed0 = [], {}
for name, factory, seeded in specs:
    for s in (SEEDS if seeded else [0]):
        t = time.time()
        m = factory(s).fit(tr, va, NU, NI)
        res, pu = evaluate(m)
        rows.append({"model": name, "seed": s, **res})
        if s == 0:
            per_user_seed0[name] = pu
        print(f"{name:42s} seed={s} {time.time() - t:5.1f}s  " +
              "  ".join(f"{k}={v:.4f}" for k, v in res.items() if k in ("RMSE", "HR@10", "NDCG@10")))

raw = pd.DataFrame(rows)
raw.to_csv(OUT / "raw_runs.csv", index=False)
agg = raw.drop(columns="seed").groupby("model", sort=False).agg(["mean", "std"])


def fmt(metric, digits=3):
    s = agg[(metric, "mean")].map(lambda v: f"{v:.{digits}f}" if pd.notna(v) else "—")
    sd = agg[(metric, "std")]
    return s.where(sd.isna(), s + " ± " + sd.map(lambda v: f"{v:.{digits}f}"))


rating = pd.DataFrame({"RMSE": fmt("RMSE"), "MAE": fmt("MAE")})
rating = rating[agg[("RMSE", "mean")].notna()]
ranking = pd.DataFrame({f"HR@{K}": fmt(f"HR@{K}"), f"NDCG@{K}": fmt(f"NDCG@{K}"),
                        f"Precision@{K}": fmt(f"Precision@{K}", 4)})
ranking.loc["Random (expected)"] = [f"{K / NI:.3f}", "—", f"{1 / NI:.4f}"]
rating.to_csv(OUT / "rating.csv")
ranking.to_csv(OUT / "ranking.csv")

# ---------- paired bootstrap: segmented minus baseline (seed 0, same test users)
sig = []
pairs = [("Bias baseline", "Bias baseline — segmented"),
         ("MF (SVD)", "MF (SVD) — segmented"),
         ("MF (SVD)", "MF (SVD) — segmented, leaky clusters"),
         ("Neural CF", "Neural CF — segmented"),
         ("Neural CF", "Neural CF — segmented, leaky clusters"),
         ("BPR-MF", "BPR-MF — segmented"),
         ("Popularity", "BPR-MF")]
for a, b in pairs:
    A, B = per_user_seed0[a], per_user_seed0[b]
    if "sq" in A and "sq" in B:
        sig.append({"comparison": f"{b} vs {a}", "metric": "ΔRMSE",
                    **dict(zip(["delta", "ci_low", "ci_high"], paired_bootstrap(A["sq"], B["sq"], "rmse")))})
    sig.append({"comparison": f"{b} vs {a}", "metric": f"ΔHR@{K}",
                **dict(zip(["delta", "ci_low", "ci_high"], paired_bootstrap(A["hit"], B["hit"])))})
sig = pd.DataFrame(sig)
sig["significant"] = (sig.ci_low > 0) | (sig.ci_high < 0)
sig.round(4).to_csv(OUT / "significance.csv", index=False)

# ---------- ablation: the ORIGINAL random 80/20 row split, with vs without leaky clusters
df = k_core(dedupe(load_interactions(ROOT / "data" / "interaction_data.csv")), 3).copy()
df["user"] = df["reviewerID"].astype("category").cat.codes
df["item"] = df["asin"].astype("category").cat.codes
ab = []
for s in SEEDS:
    rtest = df.sample(frac=0.2, random_state=s)
    rtrain = df.drop(rtest.index)
    nu, ni = df["user"].nunique(), df["item"].nunique()
    lab_clean = fit_segments(rtrain, nu)
    lab_leaky = fit_segments(rtrain, nu, leaky=True, all_rows=df)
    for name, m in [("MF (SVD)", BiasedMF(seed=s)),
                    ("MF (SVD) — segmented, train-only clusters", Segmented(lambda: BiasedMF(seed=s), lab_clean)),
                    ("MF (SVD) — segmented, leaky clusters (original)", Segmented(lambda: BiasedMF(seed=s), lab_leaky))]:
        m.fit(rtrain, rtrain.iloc[:0], nu, ni)
        ab.append({"model": name, "seed": s,
                   "RMSE": rmse(rtest["overall"], m.predict(rtest["user"].values, rtest["item"].values))})
ab = pd.DataFrame(ab).groupby("model", sort=False)["RMSE"].agg(["mean", "std"]).round(4)
ab.to_csv(OUT / "random_split_ablation.csv")

# ---------- markdown summary
md = [
    "# Results",
    "",
    f"Protocol: deduplicated 3-core of `interaction_data.csv` — {NU:,} users, {NI:,} items, "
    f"{len(all_rows):,} ratings. Leave-last-out by time: each user's latest rating is test, "
    f"previous one is validation. Rating metrics use all {len(te):,} test users; ranking metrics "
    f"rank the full catalogue (minus already-seen items) for the {len(rel):,} users whose held-out "
    f"rating is ≥4. Mean ± std over seeds {SEEDS} where the model is stochastic.",
    "",
    "## Rating prediction", "", to_md(rating), "",
    f"## Top-{K} ranking", "", to_md(ranking), "",
    "## Segmented vs baseline (paired bootstrap, 95% CI, seed 0)", "",
    "Negative ΔRMSE and positive ΔHR mean the second model is better.", "",
    to_md(sig.round(4), index=False), "",
    "## Ablation: original random 80/20 split", "",
    "Same deduplicated data, but the original protocol's random row split.", "",
    to_md(ab), "",
    "## Segments (train-only features)", "", to_md(describe(tr, labels_clean)), "",
]
(OUT / "RESULTS.md").write_text("\n".join(md))
print("\n".join(md))
