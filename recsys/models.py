"""Recommender models sharing one interface.

    model.fit(train_df, val_df, n_users, n_items)
    model.predict(users, items) -> predicted ratings (rating models only)
    model.score_all(users)      -> (len(users), n_items) ranking scores

All models use the global user/item index space from `data.Split`, so a
segmented model is just a router over per-cluster models and is evaluated on
exactly the same test rows as its baseline.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .metrics import rmse, target_ranks


# ---------------------------------------------------------------- simple baselines
class GlobalMean:
    name = "Global mean"
    rates = True

    def fit(self, train, val, n_users, n_items):
        self.mu = train["overall"].mean()
        self.n_items = n_items
        return self

    def predict(self, users, items):
        return np.full(len(users), self.mu)

    def score_all(self, users):
        return np.zeros((len(users), self.n_items))


class Popularity:
    """Rank items by training-set interaction count (ranking only)."""
    name = "Popularity"
    rates = False

    def fit(self, train, val, n_users, n_items):
        self.counts = np.bincount(train["item"], minlength=n_items).astype(float)
        return self

    def score_all(self, users):
        return np.tile(self.counts, (len(users), 1))


class BiasBaseline:
    """mu + b_user + b_item, fitted by alternating least squares (as Surprise's
    BaselineOnly). The standard 'do you even need factors?' baseline."""
    name = "Bias baseline"
    rates = True

    def __init__(self, reg_u=15.0, reg_i=10.0, n_iter=10):
        self.reg_u, self.reg_i, self.n_iter = reg_u, reg_i, n_iter

    def fit(self, train, val, n_users, n_items):
        u, i, r = train["user"].values, train["item"].values, train["overall"].values
        self.mu = r.mean()
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        for _ in range(self.n_iter):
            res = r - self.mu - self.bi[i]
            self.bu = np.bincount(u, res, n_users) / (self.reg_u + np.bincount(u, minlength=n_users))
            res = r - self.mu - self.bu[u]
            self.bi = np.bincount(i, res, n_items) / (self.reg_i + np.bincount(i, minlength=n_items))
        return self

    def predict(self, users, items):
        return np.clip(self.mu + self.bu[users] + self.bi[items], 1, 5)

    def score_all(self, users):
        return self.mu + self.bu[users][:, None] + self.bi[None, :]


# ------------------------------------------------------------------ matrix factorisation
class BiasedMF:
    """Funk-SVD with biases, trained by per-rating SGD. Same algorithm and
    defaults as surprise.SVD used in the original project (50 factors, 20 epochs,
    lr 0.005, reg 0.02, init N(0, 0.1))."""
    name = "Matrix factorisation (SVD)"
    rates = True

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, seed=0):
        self.k, self.n_epochs, self.lr, self.reg, self.seed = n_factors, n_epochs, lr, reg, seed

    def fit(self, train, val, n_users, n_items):
        rng = np.random.default_rng(self.seed)
        u = train["user"].values
        i = train["item"].values
        r = train["overall"].values.astype(float)
        self.mu = r.mean()
        bu, bi = np.zeros(n_users), np.zeros(n_items)
        P = rng.normal(0, 0.1, (n_users, self.k))
        Q = rng.normal(0, 0.1, (n_items, self.k))
        lr, reg = self.lr, self.reg
        for _ in range(self.n_epochs):
            for idx in rng.permutation(len(r)):
                uu, ii = u[idx], i[idx]
                pu, qi = P[uu], Q[ii]
                err = r[idx] - (self.mu + bu[uu] + bi[ii] + pu @ qi)
                bu[uu] += lr * (err - reg * bu[uu])
                bi[ii] += lr * (err - reg * bi[ii])
                P[uu] = pu + lr * (err * qi - reg * pu)
                Q[ii] = qi + lr * (err * pu - reg * qi)
        self.bu, self.bi, self.P, self.Q = bu, bi, P, Q
        return self

    def predict(self, users, items):
        est = self.mu + self.bu[users] + self.bi[items] + (self.P[users] * self.Q[items]).sum(1)
        return np.clip(est, 1, 5)

    def score_all(self, users):
        return self.mu + self.bu[users][:, None] + self.bi[None, :] + self.P[users] @ self.Q.T


# ------------------------------------------------------------------------ neural CF
class _NCFNet(nn.Module):
    def __init__(self, n_users, n_items, dim, mu):
        super().__init__()
        self.ue, self.ie = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        self.ub, self.ib = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1)
        for e in (self.ue, self.ie):
            nn.init.normal_(e.weight, 0, 0.05)
        for e in (self.ub, self.ib):
            nn.init.zeros_(e.weight)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.mu = mu

    def forward(self, u, i):
        x = torch.cat([self.ue(u), self.ie(i)], 1)
        return (self.mu + self.ub(u) + self.ib(i) + self.mlp(x)).squeeze(-1)


class NeuralCF:
    """MLP-based neural collaborative filtering for rating prediction.

    Changes vs. the original: global mean + user/item bias terms (the original
    had to learn the ~4.4 offset through the MLP, hence train loss of 426 at
    epoch 1), per-sample MSE, dropout + weight decay, early stopping on val RMSE.
    """
    name = "Neural CF (MLP)"
    rates = True

    def __init__(self, dim=32, lr=1e-3, wd=1e-4, batch=256, max_epochs=40, patience=4, seed=0):
        self.dim, self.lr, self.wd, self.batch = dim, lr, wd, batch
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def fit(self, train, val, n_users, n_items):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        u = torch.as_tensor(train["user"].values)
        i = torch.as_tensor(train["item"].values)
        r = torch.as_tensor(train["overall"].values, dtype=torch.float32)
        self.net = _NCFNet(n_users, n_items, self.dim, float(r.mean()))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            for b in torch.randperm(len(r), generator=g).split(self.batch):
                loss = nn.functional.mse_loss(self.net(u[b], i[b]), r[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = rmse(val["overall"].values, self.predict(val["user"].values, val["item"].values))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val_rmse = best
        return self

    @torch.no_grad()
    def predict(self, users, items):
        self.net.eval()
        p = self.net(torch.as_tensor(users), torch.as_tensor(items)).numpy()
        return np.clip(p, 1, 5)

    @torch.no_grad()
    def score_all(self, users):
        self.net.eval()
        n_items = self.net.ie.num_embeddings
        out = np.empty((len(users), n_items), dtype=np.float32)
        items = torch.arange(n_items)
        for row, uu in enumerate(users):
            out[row] = self.net(torch.full((n_items,), int(uu)), items).numpy()
        return out


# --------------------------------------------------------------- BPR (ranking objective)
class BPRMF:
    """Matrix factorisation trained with Bayesian Personalized Ranking on
    implicit feedback (who reviewed what). Unlike the rating models it is
    optimised for the top-K task, so it is the model to compare on HR/NDCG.
    Early-stops on validation HR@10."""
    name = "BPR-MF (ranking loss)"
    rates = False

    def __init__(self, dim=64, lr=5e-3, reg=1e-4, batch=1024, max_epochs=100,
                 patience=8, seed=0):
        self.dim, self.lr, self.reg, self.batch = dim, lr, reg, batch
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def fit(self, train, val, n_users, n_items):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        u = torch.as_tensor(train["user"].values)
        i = torch.as_tensor(train["item"].values)
        self.U = nn.Embedding(n_users, self.dim)
        self.I = nn.Embedding(n_items, self.dim)
        self.b = nn.Embedding(n_items, 1)
        for e in (self.U, self.I):
            nn.init.normal_(e.weight, 0, 0.01)
        nn.init.zeros_(self.b.weight)
        params = list(self.U.parameters()) + list(self.I.parameters()) + list(self.b.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)

        seen_train = train.groupby("user")["item"].apply(np.asarray)
        vu = val["user"].values
        v_seen = [seen_train.get(x, np.array([], int)) for x in vu]

        best, best_state, bad = -1, None, 0
        for _ in range(self.max_epochs):
            for b in torch.randperm(len(u), generator=g).split(self.batch):
                ub, ip = u[b], i[b]
                ineg = torch.randint(0, n_items, (len(b),), generator=g)
                pu, qp, qn = self.U(ub), self.I(ip), self.I(ineg)
                x = (pu * (qp - qn)).sum(1) + (self.b(ip) - self.b(ineg)).squeeze(-1)
                loss = -nn.functional.logsigmoid(x).mean() + self.reg * (
                    pu.pow(2).sum(1) + qp.pow(2).sum(1) + qn.pow(2).sum(1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            hr = (target_ranks(self.score_all(vu), val["item"].values, v_seen) < 10).mean()
            if hr > best + 1e-4:
                best, bad = hr, 0
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad >= self.patience:
                    break
        with torch.no_grad():
            for p, s in zip(params, best_state):
                p.copy_(s)
        self.best_val_hr = best
        return self

    @torch.no_grad()
    def score_all(self, users):
        U = self.U.weight[torch.as_tensor(users)]
        return (U @ self.I.weight.T + self.b.weight.T).numpy()


# ------------------------------------------------------------------------ segmentation
class Segmented:
    """Train one copy of a model per user cluster and route each user to it."""
    rates = True

    def __init__(self, factory, labels: np.ndarray, tag: str = ""):
        self.factory, self.labels = factory, labels
        proto = factory()
        self.name = f"{proto.name} — segmented{tag}"
        self.rates = proto.rates

    def fit(self, train, val, n_users, n_items):
        self.n_items = n_items
        self.models = {}
        for c in np.unique(self.labels):
            tr = train[self.labels[train["user"].values] == c]
            va = val[self.labels[val["user"].values] == c]
            self.models[c] = self.factory().fit(tr, va, n_users, n_items)
        return self

    def predict(self, users, items):
        users, items = np.asarray(users), np.asarray(items)
        out = np.empty(len(users))
        lab = self.labels[users]
        for c, m in self.models.items():
            mask = lab == c
            if mask.any():
                out[mask] = m.predict(users[mask], items[mask])
        return out

    def score_all(self, users):
        users = np.asarray(users)
        out = np.empty((len(users), self.n_items))
        lab = self.labels[users]
        for c, m in self.models.items():
            mask = lab == c
            if mask.any():
                out[mask] = m.score_all(users[mask])
        return out
