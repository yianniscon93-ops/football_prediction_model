"""Parlay builder / verdict logic (pure functions, no model fit)."""
import numpy as np
import pandas as pd

from footy_prediction_model.scan import _verdict, build_parlays, pick_legs, tier_names


def _legs(n_fixtures):
    recs = []
    for m in range(n_fixtures):
        for leg, p, ok in (("Under 4.5", 0.86, False), ("1", 0.78 - m * 0.005, True),
                           ("Over 1.5", 0.75, True)):
            recs.append({"match_id": m, "fixture": f"F{m}", "leg": leg, "p": p, "fair_p": p,
                         "edge": 0.0, "playable": ok})
    return pd.DataFrame(recs)


def test_one_leg_per_fixture_most_likely_playable():
    picks = pick_legs(_legs(5))
    assert len(picks) == 5 and picks["match_id"].is_unique
    assert (picks["leg"] == "1").all()          # Under 4.5 not playable (outside band)


def test_parlays_are_disjoint_and_6_to_7_legs():
    for n in (6, 7, 13, 14, 20, 27):
        parts = build_parlays(pick_legs(_legs(n)))
        sizes = [len(p) for p in parts]
        assert sum(sizes) == n and all(6 <= s <= 7 for s in sizes), (n, sizes)
        ids = np.concatenate([p["match_id"].to_numpy() for p in parts])
        assert len(set(ids)) == len(ids)
        if len(parts) > 1:   # safest first
            assert parts[0]["p"].min() >= parts[-1]["p"].max()
    assert len(build_parlays(pick_legs(_legs(4)))) == 1   # fewer than 6: one short ticket
    assert tier_names(3) == ["Conservative", "Balanced", "Aggressive"]


def test_verdict():
    assert _verdict(0.82, 0.81, 0.80, 1.25) == "GREEN"
    assert _verdict(0.69, 0.79, 0.80, 1.25) == "RED"       # model -11 pts, anchored agrees
    assert _verdict(0.77, 0.79, 0.80, 1.25) == "YELLOW"
    assert _verdict(0.74, 0.79, 0.80, 1.25) == "ORANGE"
    assert _verdict(0.70, 0.81, 0.80, 1.25) == "YELLOW"    # anchored disagrees: not red
    assert _verdict(0.70, 0.70, 0.65, 1.54) == "YELLOW"    # >1.50 never green
    assert _verdict(0.80, np.nan, np.nan, np.nan) == "NODATA"


def test_watch_settle_and_log(tmp_path):
    from footy_prediction_model.watch import grade, log_watch, settle
    assert settle("1", 2, 1) == 1 and settle("X2", 2, 1) == 0 and settle("12", 1, 1) == 0
    assert settle("O2.5", 2, 1) == 1 and settle("U2.5", 2, 1) == 0 and settle("BTTS_Y", 2, 0) == 0
    assert settle("A_SCORE", 0, 1) == 1 and np.isnan(settle("1", np.nan, 0))
    w = pd.DataFrame({"date": pd.to_datetime(["2026-11-08"]), "league": ["E0"], "home": ["A"], "away": ["B"],
                      "key": ["1"], "leg": ["1"], "fair_p": [0.6], "model_p": [0.45], "fair_odds": [1.67],
                      "min_odds": [1.59], "logged": [pd.Timestamp("2026-11-08 12:00")]})
    path = str(tmp_path / "w.csv")
    log_watch(w, path)
    log_watch(w.assign(fair_p=0.7, logged=pd.Timestamp("2026-11-08 15:00")), path)   # later re-scan
    m = pd.DataFrame({"league": ["E0"], "date": pd.to_datetime(["2026-11-08"]), "home": ["A"], "away": ["B"],
                      "hg": [2.0], "ag": [0.0]})
    g, s = grade(path, m)
    assert len(g) == 1 and g["fair_p"].iloc[0] == 0.6        # first (pre-match) price kept
    assert s["n"] == 1 and s["won"] == 1.0 and abs(s["roi_min"] - 0.59) < 1e-9


def test_review_settles_scan_against_results(tmp_path):
    from footy_prediction_model.review import review
    rows = []
    for mid, (fx, p1, px, p2) in enumerate([("A v B", .60, .25, .15), ("C v D", .30, .30, .40)]):
        for key, fp, mp in (("1", p1, p1 - .05), ("X", px, px), ("2", p2, p2 + .05), ("O2.5", .55, .5),
                            ("1X", p1 + px, p1 + px - .05)):
            rows.append(dict(match_id=mid, kickoff="2026-11-08 16:00", league="E0", fixture=fx, leg=key,
                             key=key, fair_p=fp, fair_odds=1 / fp, min_odds=round(1 / fp / 1.05, 2),
                             model_p=mp, reason=""))
    legs = pd.DataFrame(rows)
    m = pd.DataFrame({"league": "E0", "date": pd.to_datetime(["2026-11-08"] * 2), "home": ["A", "C"],
                      "away": ["B", "D"], "hg": [2.0, 0.0], "ag": [0.0, 0.0]})
    rep = review(pd.Timestamp("2026-11-08"), legs, m, str(tmp_path))
    assert "2 of 2 games have results" in rep
    assert "Tuning verdict: not yet" in rep
    log = pd.read_csv(tmp_path / "review_log.csv")
    assert len(log) == 2 and set(log["hg"]) == {2.0, 0.0}
    review(pd.Timestamp("2026-11-08"), legs, m, str(tmp_path))      # re-run same day: no duplicates
    assert len(pd.read_csv(tmp_path / "review_log.csv")) == 2
