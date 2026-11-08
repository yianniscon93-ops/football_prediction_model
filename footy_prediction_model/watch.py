"""Watch list: a no-stakes experiment.

Hypothesis (from the 2024-26 backtest): when the model rates a bet 10+ points BELOW the
sharp market, the market's side wins more often than its price implies. Held-out
2025/26: 72% won vs 63% implied (+9% at min odds); 2024/25: about break-even. So it
is tracked live before anyone stakes money on it.

Each scan logs qualifying bets (pre-match prices) to <data>/watchlist.csv; the
`watch` command grades them against results.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

WATCH_GAP = 0.10
WATCH_ODDS = (1.19, 2.60)
COLS = ["date", "league", "home", "away", "key", "leg", "fair_p", "model_p", "fair_odds", "min_odds",
        "logged"]


def select_watch(legs: pd.DataFrame, views: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    if legs is None or legs.empty:
        return pd.DataFrame(columns=COLS + ["kickoff", "match_id"])
    L = legs.dropna(subset=["fair_p"]).copy()
    thin = views.set_index("match_id")["thin"].fillna(False) if "thin" in views else pd.Series(dtype=bool)
    L = L[~L["match_id"].map(thin).fillna(False).astype(bool)]
    L = L[~L["reason"].fillna("").str.contains("v3.1")]
    L = L[((L["model_p"] - L["fair_p"]) <= -WATCH_GAP) & L["fair_odds"].between(*WATCH_ODDS)]
    # one bet per match (the biggest doubt), so the live record counts each game once
    L = L.assign(_gap=L["model_p"] - L["fair_p"]).sort_values("_gap").groupby("match_id").head(1)
    meta = views.set_index("match_id")[["home", "away"]]
    L = L.join(meta, on="match_id")
    L["date"] = date.normalize()
    L["logged"] = pd.Timestamp.now().floor("min")
    return L[COLS + ["kickoff", "match_id"]].sort_values(["kickoff", "match_id"])


def log_watch(w: pd.DataFrame, path: str) -> pd.DataFrame:
    """Append; the first logged (earliest, pre-match) price for a bet is kept."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = w[COLS].copy()
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["date", "logged"], dtype={"key": str, "leg": str})
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=["date", "league", "home", "away", "key"], keep="first")
    new.to_csv(path, index=False)
    return new


def settle(key, hg: float, ag: float) -> float:
    key = str(key)
    if pd.isna(hg) or pd.isna(ag):
        return np.nan
    t = hg + ag
    rules = {"1": hg > ag, "X": hg == ag, "2": hg < ag, "1X": hg >= ag, "X2": hg <= ag, "12": hg != ag,
             "BTTS_Y": hg > 0 and ag > 0, "BTTS_N": not (hg > 0 and ag > 0),
             "H_SCORE": hg > 0, "A_SCORE": ag > 0}
    if key in rules:
        return float(rules[key])
    if key[0] in "OU":
        line = float(key[1:])
        return float(t > line if key[0] == "O" else t < line)
    return np.nan


def grade(path: str, matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not os.path.exists(path):
        return pd.DataFrame(), {"n": 0}
    w = pd.read_csv(path, parse_dates=["date", "logged"], dtype={"key": str, "leg": str})
    res = matches[matches["hg"].notna()][["league", "date", "home", "away", "hg", "ag"]]
    w = w.merge(res, on=["league", "date", "home", "away"], how="left")
    w["won"] = [settle(k, h, a) for k, h, a in zip(w["key"], w["hg"], w["ag"])]
    done = w[w["won"].notna()]
    s = {"n": int(len(done)), "pending": int(w["won"].isna().sum())}
    if len(done):
        s.update(won=float(done["won"].mean()), implied=float(done["fair_p"].mean()),
                 roi_min=float((done["won"] * done["min_odds"]).mean() - 1),
                 roi_fair=float((done["won"] * done["fair_odds"]).mean() - 1))
    return w, s
