"""Model-level tests: leakage, score matrix, xG target. No network."""
import numpy as np
import pandas as pd

from footy_prediction_model import FootyModel, ModelConfig, build_features
from footy_prediction_model.config import LEARNED_FEATURES

from test_smoke import _synthetic


def _with_true_xg(m, att, noise=0.15, seed=1):
    """Attach 'real xG' = the true scoring rate (+ noise), like a perfect xG model."""
    rng = np.random.default_rng(seed)
    m = m.copy()
    lh = m["_lh"].to_numpy()
    la = m["_la"].to_numpy()
    m["home_xg"] = np.clip(lh * np.exp(rng.normal(0, noise, len(m))), 0.05, None)
    m["away_xg"] = np.clip(la * np.exp(rng.normal(0, noise, len(m))), 0.05, None)
    return m


def test_features_are_leak_free():
    """Changing a later result must not change features of earlier matches."""
    m, _ = _synthetic()
    cut = m["date"].iloc[len(m) // 2]
    base = build_features(m)
    m2 = m.copy()
    late = m2["date"] > cut
    m2.loc[late, "hg"] = m2.loc[late, "hg"] + 3
    m2.loc[late, "hst"] = m2.loc[late, "hst"] + 5
    alt = build_features(m2)
    early = base["date"] <= cut
    cols = LEARNED_FEATURES + ["goals"]
    pd.testing.assert_frame_equal(base.loc[early, cols].reset_index(drop=True),
                                  alt.loc[early, cols].reset_index(drop=True))


def test_fit_ignores_matches_on_or_after_ref_date():
    m, _ = _synthetic()
    rows = build_features(m)
    ref = m["date"].iloc[len(m) // 2]
    a = FootyModel().fit(rows, ref)
    rows2 = rows.copy()
    rows2.loc[rows2["date"] >= ref, "goals"] = 9
    b = FootyModel().fit(rows2, ref)
    assert np.allclose(a.glm.coef_, b.glm.coef_)


def test_score_matrix_is_a_distribution():
    m, _ = _synthetic()
    model = FootyModel().fit(build_features(m), m.date.max() + pd.Timedelta(days=1))
    for lh, la in ((0.3, 0.3), (1.5, 1.1), (3.5, 0.4)):
        M = model.score_matrix(lh, la)
        assert (M >= 0).all() and abs(M.sum() - 1) < 1e-9
        # marginal means stay close to the requested lambdas (DC only touches 0/1 cells)
        g = np.arange(M.shape[0])
        assert abs((M.sum(1) * g).sum() - lh) < 0.1
        assert abs((M.sum(0) * g).sum() - la) < 0.1


def test_xg_target_recovers_strength_better_than_goals():
    """The point of xG: with one season of data, ratings fitted on xG should track
    true team strength more closely than ratings fitted on goals."""
    m, att = _synthetic(seed=3, seasons=1, keep_rates=True)
    m = _with_true_xg(m, att)
    rows = build_features(m)
    ref = m.date.max() + pd.Timedelta(days=1)
    true = np.array([att[t] for t in sorted(att)])

    def corr(w):
        mod = FootyModel(ModelConfig(xi=0.0, xg_weight=w)).fit(rows, ref)
        coef = dict(zip(mod.columns_, mod.glm.coef_))
        return np.corrcoef([coef[f"att_{t}"] for t in sorted(att)], true)[0, 1], mod

    c_goals, _ = corr(0.0)
    c_xg, mod = corr(1.0)
    assert c_xg > c_goals
    assert c_xg > 0.85
    assert 0.8 < mod.xg_scale_ < 1.25 and mod.xg_share_ == 1.0


def test_xg_weight_zero_matches_goals_only_model():
    m, att = _synthetic(keep_rates=True)
    m = _with_true_xg(m, att)
    rows = build_features(m)
    ref = m.date.max() + pd.Timedelta(days=1)
    a = FootyModel(ModelConfig(xg_weight=0.0)).fit(rows, ref)
    b = FootyModel(ModelConfig(xg_weight=0.0)).fit(rows.assign(xg_for=np.nan), ref)
    assert np.allclose(a.glm.coef_, b.glm.coef_)
