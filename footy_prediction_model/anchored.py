"""Market-anchored model: does each gate add information BEYOND the price?

1. Convert each match's de-margined 1X2 + O/U 2.5 prices into the expected goals
   the market implies (lambda_home, lambda_away).
2. Fit a Poisson regression with log(market lambda) as a fixed offset:
       log(lambda) = log(lambda_market) + b0 + b_home*is_home + sum(beta_g * gate_features)
   A gate whose beta is ~0 (|z| < 2) is already priced in by the bookmaker.
3. Out-of-sample: does the adjusted probability beat the market's own?

For live slips this needs the fixture's 1X2 and O/U 2.5 prices
(columns odds_1, odds_x, odds_2, odds_o25, odds_u25 in the slip CSV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import poisson

from .config import LEARNED_FEATURES, MANUAL_FEATURES, MANUAL_PRIORS
from .markets import devig

_G = np.arange(0.15, 4.5, 0.02)
_K = np.arange(0, 12)


def _grid():
    pm = poisson.pmf(_K[None, :], _G[:, None])          # (n_lambda, goals)
    lh, la = np.meshgrid(_G, _G, indexing="ij")
    i, j = np.indices((len(_K), len(_K)))
    home = (i > j).astype(float)
    away = (i < j).astype(float)
    over = ((i + j) > 2.5).astype(float)
    ph = np.einsum("ak,bl,kl->ab", pm, pm, home)
    pa = np.einsum("ak,bl,kl->ab", pm, pm, away)
    po = np.einsum("ak,bl,kl->ab", pm, pm, over)
    pts = np.stack([ph.ravel(), pa.ravel(), po.ravel()], 1)
    return cKDTree(pts), lh.ravel(), la.ravel()


_TREE = None


def market_lambdas(odds_1, odds_x, odds_2, odds_o, odds_u):
    """Vectorised: implied (lambda_home, lambda_away) from 1X2 and O/U 2.5 prices."""
    global _TREE
    if _TREE is None:
        _TREE = _grid()
    tree, LH, LA = _TREE
    o = np.column_stack([odds_1, odds_x, odds_2, odds_o, odds_u]).astype(float)
    out = np.full((len(o), 2), np.nan)
    ok = np.isfinite(o).all(1) & (o > 1).all(1)
    q = []
    for a, b, c, d, e in o[ok]:
        h, _, aw = devig([a, b, c])
        q.append([h, aw, devig([d, e])[0]])
    if q:
        _, idx = tree.query(np.array(q))
        out[ok, 0], out[ok, 1] = LH[idx], LA[idx]
    return out


def attach_market(rows: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    lam = market_lambdas(matches["odds_h"], matches["odds_d"], matches["odds_a"],
                         matches["odds_o25"], matches["odds_u25"])
    mk = pd.DataFrame({"match_id": matches["match_id"].to_numpy(), "mlh": lam[:, 0], "mla": lam[:, 1]})
    r = rows.merge(mk, on="match_id", how="left")
    r["mkt_lam"] = np.where(r["is_home"] == 1, r["mlh"], r["mla"])
    return r.drop(columns=["mlh", "mla"])


class AnchoredModel:
    def __init__(self, l2: float = 1.0, xi: float = 0.0):
        self.l2, self.xi = l2, xi
        self.cols = ["const", "is_home"] + LEARNED_FEATURES

    def _X(self, r):
        X = r[LEARNED_FEATURES].to_numpy(float)
        return np.column_stack([np.ones(len(r)), r["is_home"].to_numpy(float), X])

    def fit(self, r: pd.DataFrame, ref_date=None):
        r = r[r["goals"].notna() & r["mkt_lam"].notna()]
        if ref_date is not None:
            r = r[r["date"] < pd.Timestamp(ref_date)]
        X, y, off = self._X(r), r["goals"].to_numpy(float), np.log(r["mkt_lam"].to_numpy())
        w = np.ones(len(r))
        if self.xi and ref_date is not None:
            w = np.exp(-self.xi * (pd.Timestamp(ref_date) - r["date"]).dt.days.to_numpy())
        pen = np.ones(X.shape[1]); pen[0] = 0

        def f(b):
            eta = off + X @ b
            mu = np.exp(eta)
            nll = np.sum(w * (mu - y * eta)) + 0.5 * self.l2 * np.sum(pen * b * b)
            g = X.T @ (w * (mu - y)) + self.l2 * pen * b
            return nll, g
        res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        self.beta = res.x
        mu = np.exp(off + X @ self.beta)
        H = (X * (w * mu)[:, None]).T @ X + self.l2 * np.diag(pen)
        self.se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
        self.n = len(r)
        return self

    def table(self):
        t = pd.DataFrame({"feature": self.cols, "beta": self.beta, "se": self.se})
        t["z"] = t["beta"] / t["se"]
        t["goal_effect_per_unit_%"] = 100 * (np.exp(t["beta"]) - 1)
        t["adds_info_beyond_price"] = np.where(t["z"].abs() >= 2, "YES", "no")
        return t

    def lambdas(self, r: pd.DataFrame) -> np.ndarray:
        manual = sum(MANUAL_PRIORS[c] * r[c].to_numpy(float) for c in MANUAL_FEATURES)
        return np.exp(np.log(r["mkt_lam"].to_numpy()) + self._X(r) @ self.beta + manual)


def evaluate_anchored(rows, matches, train_end: str, test_end: str, l2: float = 1.0):
    """Fit on everything before train_end, score [train_end, test_end) vs market."""
    from .markets import market_probs
    r = attach_market(rows, matches)
    am = AnchoredModel(l2=l2).fit(r, train_end)
    te = r[(r["date"] >= pd.Timestamp(train_end)) & (r["date"] < pd.Timestamp(test_end))
           & r["goals"].notna() & r["mkt_lam"].notna()].copy()
    te["adj_lam"] = am.lambdas(te)
    h = te[te.is_home == 1].set_index("match_id")
    a = te[te.is_home == 0].set_index("match_id")
    j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
    g = np.arange(11)
    rec = []
    for mid, x in j.iterrows():
        out = {}
        for tag, lh, la in (("mkt", x["mkt_lam_h"], x["mkt_lam_a"]), ("adj", x["adj_lam_h"], x["adj_lam_a"])):
            M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la)); M /= M.sum()
            p = market_probs(M)
            out.update({f"{tag}_1": p["1"], f"{tag}_X": p["X"], f"{tag}_2": p["2"], f"{tag}_O": p["O2.5"]})
        hg, ag = x["goals_h"], x["goals_a"]
        out["res"] = "1" if hg > ag else "X" if hg == ag else "2"
        out["over"] = int(hg + ag > 2.5)
        rec.append(out)
    d = pd.DataFrame(rec)
    eps = 1e-12
    res = {"n_test_matches": len(d)}
    for tag in ("mkt", "adj"):
        P = d[[f"{tag}_1", f"{tag}_X", f"{tag}_2"]].to_numpy()
        Y = np.stack([(d.res == o).to_numpy() for o in ("1", "X", "2")], 1)
        res[f"{tag}_1x2_logloss"] = float(-np.mean(np.log(np.sum(P * Y, 1) + eps)))
        po, yo = d[f"{tag}_O"].to_numpy(), d["over"].to_numpy()
        res[f"{tag}_ou25_logloss"] = float(-np.mean(yo * np.log(po + eps) + (1 - yo) * np.log(1 - po + eps)))
    return am, res
