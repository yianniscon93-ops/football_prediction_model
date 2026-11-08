"""Time-decayed Poisson regression with Dixon-Coles low-score correction.

log(lambda_team) = intercept + league + home_adv[league]*is_home
                   + attack[team] - defence_strength[opp]      (team ratings)
                   + sum(beta_g * gate_features)                (learned gate weights)
                   + sum(prior_m * manual_gate3_inputs)         (fixed priors)

Each training match is weighted by exp(-xi * days_ago), so recent form counts
more. xi is tuned by backtest (see tune.py).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from .config import FEATURE_GROUPS, LEARNED_FEATURES, MANUAL_FEATURES, ModelConfig


class FootyModel:
    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or ModelConfig()
        self.glm = None
        self.columns_: list[str] = []
        self.rho_ = 0.0
        self.teams_: set[str] = set()
        self.ref_date_ = None

    # ------------------------------------------------------------------ design
    def _design(self, rows: pd.DataFrame) -> np.ndarray:
        n = len(rows)
        col_idx = {c: i for i, c in enumerate(self.columns_)}
        X = np.zeros((n, len(self.columns_)))
        lg = rows["league"].astype(str).to_numpy()
        ih = rows["is_home"].to_numpy()
        team = rows["team"].astype(str).to_numpy()
        opp = rows["opp"].astype(str).to_numpy()
        for i in range(n):
            j = col_idx.get(f"lg_{lg[i]}")
            if j is not None:
                X[i, j] = 1
            if ih[i]:
                j = col_idx.get(f"home_{lg[i]}")
                if j is not None:
                    X[i, j] = 1
            if self.cfg.use_team_ratings:
                j = col_idx.get(f"att_{team[i]}")
                if j is not None:
                    X[i, j] = 1
                j = col_idx.get(f"def_{opp[i]}")
                if j is not None:
                    X[i, j] = 1
        for f in LEARNED_FEATURES:
            X[:, col_idx[f]] = rows[f].to_numpy(float)
        return X

    # --------------------------------------------------------------------- fit
    def fit(self, rows: pd.DataFrame, ref_date) -> "FootyModel":
        ref_date = pd.Timestamp(ref_date)
        tr = rows[rows["goals"].notna() & (rows["date"] < ref_date)
                  & (rows["date"] >= ref_date - pd.Timedelta(days=self.cfg.train_days))]
        if len(tr) < 200:
            raise ValueError(f"Not enough training rows before {ref_date.date()}: {len(tr)}")
        leagues = sorted(tr["league"].astype(str).unique())
        cols = [f"lg_{l}" for l in leagues[1:]] + [f"home_{l}" for l in leagues]
        if self.cfg.use_team_ratings:
            self.teams_ = set(tr["team"].astype(str))
            cols += [f"att_{t}" for t in sorted(self.teams_)]
            cols += [f"def_{t}" for t in sorted(self.teams_)]
        self.columns_ = cols + LEARNED_FEATURES
        X = self._design(tr)
        age = (ref_date - tr["date"]).dt.days.to_numpy()
        w = np.exp(-self.cfg.xi * age)
        off = self._manual_offset(tr)
        y = tr["goals"].to_numpy(float)
        # sklearn has no offset arg; manual features are 0 in history anyway.
        self.glm = PoissonRegressor(alpha=self.cfg.alpha, max_iter=3000, tol=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.glm.fit(X, y, sample_weight=w / w.mean())
        self.ref_date_ = ref_date
        self._fit_rho(tr, X, w, off)
        return self

    def _manual_offset(self, rows):
        pri = self.cfg.manual_priors
        return sum(pri[c] * rows[c].to_numpy(float) for c in MANUAL_FEATURES)

    def _fit_rho(self, tr, X, w, off):
        lam = np.exp(self.glm.intercept_ + X @ self.glm.coef_ + off)
        df = tr[["match_id", "is_home", "goals"]].copy()
        df["lam"], df["w"] = lam, w
        h = df[df.is_home == 1].set_index("match_id")
        a = df[df.is_home == 0].set_index("match_id")
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
        x, y = j["goals_h"].to_numpy(), j["goals_a"].to_numpy()
        lh, la, ww = j["lam_h"].to_numpy(), j["lam_a"].to_numpy(), j["w_h"].to_numpy()

        def nll(rho):
            t = np.ones_like(lh)
            t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
            t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
            t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
            t = np.where((x == 1) & (y == 1), 1 - rho, t)
            if np.any(t <= 0):
                return 1e12
            return -np.sum(ww * np.log(t))
        self.rho_ = float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)

    # ----------------------------------------------------------------- predict
    def predict_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Expected goals for each scoring row + per-gate contributions (log scale)."""
        X = self._design(rows)
        coef = dict(zip(self.columns_, self.glm.coef_))
        out = rows[["match_id", "team", "opp", "is_home"]].copy()
        lin = self.glm.intercept_ + X @ self.glm.coef_
        pri = self.cfg.manual_priors
        manual = sum(pri[c] * rows[c].to_numpy(float) for c in MANUAL_FEATURES)
        out["lam"] = np.exp(lin + manual)
        # contributions
        for g, feats in FEATURE_GROUPS.items():
            out[g] = sum(coef[f] * rows[f].to_numpy(float) for f in feats)
        out["G3_manual"] = manual
        rat = np.zeros(len(rows))
        if self.cfg.use_team_ratings:
            rat = np.array([coef.get(f"att_{t}", 0.0) + coef.get(f"def_{o}", 0.0)
                            for t, o in zip(rows["team"].astype(str), rows["opp"].astype(str))])
        out["ratings"] = rat
        out["known_team"] = rows["team"].astype(str).isin(self.teams_).to_numpy() if self.cfg.use_team_ratings else True
        out["known_opp"] = rows["opp"].astype(str).isin(self.teams_).to_numpy() if self.cfg.use_team_ratings else True
        return out

    def predict_matches(self, rows: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_rows(rows)
        h = p[p.is_home == 1].set_index("match_id")
        a = p[p.is_home == 0].set_index("match_id")
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
        j["known"] = j["known_team_h"] & j["known_team_a"]
        return j

    def score_matrix(self, lh: float, la: float) -> np.ndarray:
        g = np.arange(self.cfg.max_goals + 1)
        M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        r = self.rho_
        M[0, 0] *= 1 - lh * la * r
        M[0, 1] *= 1 + lh * r
        M[1, 0] *= 1 + la * r
        M[1, 1] *= 1 - r
        M = np.clip(M, 0, None)
        return M / M.sum()

    def coefficients(self) -> pd.DataFrame:
        coef = dict(zip(self.columns_, self.glm.coef_))
        recs = []
        for g, feats in FEATURE_GROUPS.items():
            for f in feats:
                recs.append({"gate": g, "feature": f, "beta": coef[f],
                             "goal_effect_per_unit_%": 100 * (np.exp(coef[f]) - 1)})
        for f, b in self.cfg.manual_priors.items():
            recs.append({"gate": "G3_manual (prior)", "feature": f, "beta": b,
                         "goal_effect_per_unit_%": 100 * (np.exp(b) - 1)})
        return pd.DataFrame(recs)
