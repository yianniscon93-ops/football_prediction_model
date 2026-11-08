"""xG ingestion tests (FotMob parsing, caching, name mapping, loader merge). No network."""
import numpy as np
import pandas as pd

from footy_prediction_model.data import load_matches
from footy_prediction_model.xg import map_to_reference, parse_match_xg, update_raw


def _details(h, a):
    return {"content": {"stats": {"Periods": {"All": {"stats": [
        {"title": "Top stats", "stats": [{"key": "ball_possession", "stats": [55, 45]},
                                         {"key": "expected_goals", "stats": [h, a]}]}]}}}}}


def test_parse_match_xg():
    assert parse_match_xg(_details("1.05", "0.98")) == (1.05, 0.98)
    h, a = parse_match_xg(_details(None, None))
    assert np.isnan(h) and np.isnan(a)
    assert np.isnan(parse_match_xg({"content": {"stats": None}})[0])


class FakeClient:
    def __init__(self, fixtures, xg):
        self.fixtures, self.xg, self.detail_calls = fixtures, xg, 0

    def finished_matches(self, league_id, season):
        return self.fixtures.get(season, [])

    def match_xg(self, mid):
        self.detail_calls += 1
        return self.xg.get(mid, (np.nan, np.nan))


def _fx(i, h, a, hg, ag, day):
    return {"fm_id": i, "utc": f"2025-09-{day:02d}T19:00:00Z", "home_id": h, "away_id": a,
            "home_fm": f"FM{h}", "away_fm": f"FM{a}", "hg": hg, "ag": ag}


def test_update_raw_caches_and_skips_uncovered_seasons(tmp_path):
    path = str(tmp_path / "raw.csv")
    covered = [_fx(i, 1, 2, 1, 0, 1 + i) for i in range(12)]
    uncovered = [_fx(100 + i, 1, 2, 0, 0, 1 + i) for i in range(12)]
    cl = FakeClient({"2025/2026": covered, "2024/2025": uncovered},
                    {i: (1.2, 0.4) for i in range(12)})
    raw = update_raw(["E0"], [2024, 2025], path, client=cl, workers=2, verbose=False)
    assert len(raw) == 12 and raw["home_xg"].notna().all()
    assert cl.detail_calls == 12 + 3           # probe results reused; 3 to rule out 2024/25
    cl.detail_calls = 0
    update_raw(["E0"], [2025], path, client=cl, workers=2, verbose=False)
    assert cl.detail_calls == 0                 # second run: all cached


def test_map_to_reference_learns_names_from_fixture_overlap():
    ref = pd.DataFrame({
        "league": "E0",
        "date": pd.to_datetime(["2025-09-01", "2025-09-08", "2025-09-15", "2025-09-22"]),
        "home": ["Man City", "Wolves", "Man City", "Nott'm Forest"],
        "away": ["Wolves", "Nott'm Forest", "Nott'm Forest", "Man City"],
        "hg": [2, 1, 3, 3], "ag": [0, 1, 1, 1]})
    raw = pd.DataFrame({
        "fm_id": [1, 2, 3, 4], "league": "E0", "season": 2025,
        # one kick-off is late UTC on the previous day: the +-1 day window must catch it
        "utc": ["2025-09-01T14:00:00Z", "2025-09-07T23:30:00Z", "2025-09-15T19:00:00Z",
                "2025-09-22T15:00:00Z"],
        "home_id": [10, 20, 10, 30], "away_id": [20, 30, 30, 10],
        "home_fm": ["Manchester City", "Wolverhampton", "Manchester City", "Nottingham Forest"],
        "away_fm": ["Wolverhampton", "Nottingham Forest", "Nottingham Forest", "Manchester City"],
        "hg": [2, 1, 3, 3], "ag": [0, 1, 1, 1],
        "home_xg": [2.1, 0.9, 2.5, 1.4], "away_xg": [0.3, 1.0, 0.8, 1.9]})
    xg, names = map_to_reference(raw, ref)
    assert names == {"Manchester City": "Man City", "Wolverhampton": "Wolves",
                     "Nottingham Forest": "Nott'm Forest"}
    assert len(xg) == 4
    # our local date is kept, not FotMob's UTC date
    row = xg[(xg.home == "Wolves")].iloc[0]
    assert row["date"] == pd.Timestamp("2025-09-08") and row["home_xg"] == 0.9


def test_load_matches_merges_xg_by_league(tmp_path):
    fd = tmp_path / "E0.csv"
    pd.DataFrame({"Div": ["E0", "E0"], "Date": ["01/09/2025", "08/09/2025"],
                  "HomeTeam": ["Man City", "Wolves"], "AwayTeam": ["Wolves", "Man City"],
                  "FTHG": [2, 0], "FTAG": [0, 1]}).to_csv(fd, index=False)
    xg = tmp_path / "xg.csv"
    pd.DataFrame({"date": ["2025-09-01"], "league": ["E0"], "home": ["Man City"],
                  "away": ["Wolves"], "home_xg": [2.4], "away_xg": [0.3]}).to_csv(xg, index=False)
    m = load_matches([str(fd)], leagues=["E0"], since=None, xg_path=str(xg))
    assert m["home_xg"].notna().sum() == 1
    assert m.loc[m.home == "Man City", "home_xg"].iloc[0] == 2.4


def test_unify_spellings_merges_punctuation_variants():
    from footy_prediction_model.data import unify_spellings
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2025-01-01", "2025-02-01"]),
                       "home": ["Nottm Forest", "Nott'm Forest", "Arsenal"],
                       "away": ["Arsenal", "Arsenal", "Nott'm Forest"]})
    out, mp = unify_spellings(df)
    assert mp == {"Nottm Forest": "Nott'm Forest"}
    assert set(out["home"]) | set(out["away"]) == {"Nott'm Forest", "Arsenal"}
