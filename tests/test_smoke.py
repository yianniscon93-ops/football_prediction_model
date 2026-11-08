"""End-to-end smoke test on a synthetic league (no network needed).
Run: python -m pytest tests -q"""
import numpy as np
import pandas as pd

from footy_prediction_model import FootyModel, build_features, score_slip
from footy_prediction_model.markets import devig, market_probs, parse_selection


def _synthetic(seed=0, seasons=3, keep_rates=False):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(12)]
    att = dict(zip(teams, rng.normal(0, 0.3, 12)))
    dfn = dict(zip(teams, rng.normal(0, 0.3, 12)))
    recs, d = [], pd.Timestamp("2022-08-01")
    for s in range(seasons):
        for rnd in range(22):
            perm = rng.permutation(teams)
            for h, a in zip(perm[::2], perm[1::2]):
                lh = np.exp(0.15 + 0.25 + att[h] - dfn[a])
                la = np.exp(0.15 + att[a] - dfn[h])
                hg, ag = rng.poisson(lh), rng.poisson(la)
                recs.append(dict(league="ZZ", date=d, home=h, away=a, hg=hg, ag=ag,
                                 hs=hg * 4 + 6, as_=ag * 4 + 5, hst=hg * 2 + 2, ast=ag * 2 + 2,
                                 _lh=lh, _la=la))
            d += pd.Timedelta(days=7)
        d += pd.Timedelta(days=60)
    m = pd.DataFrame(recs)
    if not keep_rates:
        m = m.drop(columns=["_lh", "_la"])
    for c in ("home_elo", "away_elo", "odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25",
              "home_xg", "away_xg"):
        m[c] = np.nan
    m["season"] = np.where(m.date.dt.month >= 7, m.date.dt.year, m.date.dt.year - 1)
    m["match_id"] = np.arange(len(m))
    return m, att


def test_pipeline_recovers_strength():
    m, att = _synthetic()
    rows = build_features(m)
    assert len(rows) == 2 * len(m)
    model = FootyModel().fit(rows, m.date.max() + pd.Timedelta(days=1))
    coef = dict(zip(model.columns_, model.glm.coef_))
    est = np.array([coef[f"att_{t}"] for t in sorted(att)])
    true = np.array([att[t] for t in sorted(att)])
    assert np.corrcoef(est, true)[0, 1] > 0.7


def test_markets_and_parsing():
    M = np.outer([0.3, 0.4, 0.3], [0.5, 0.3, 0.2])
    p = market_probs(M / M.sum())
    assert abs(p["1"] + p["X"] + p["2"] - 1) < 1e-9
    assert abs(p["1X"] - (p["1"] + p["X"])) < 1e-9
    assert parse_selection("Over 2.5", "A", "B") == "O2.5"
    assert parse_selection("x2", "A", "B") == "X2"
    assert parse_selection("B", "A", "B") == "2"
    assert abs(sum(devig([1.9, 3.6, 4.2])) - 1) < 1e-6


def test_score_slip():
    m, _ = _synthetic()
    slip = pd.DataFrame([dict(league="ZZ", date=m.date.max() + pd.Timedelta(days=3),
                              home="T0", away="T1", selection="1X", odds=1.30),
                         dict(league="ZZ", date=m.date.max() + pd.Timedelta(days=3),
                              home="T2", away="T3", selection="Under 3.5", odds=1.25)])
    res, summary = score_slip(m, slip)
    assert res["model_p"].between(0, 1).all()
    assert summary["legs"] == 2
