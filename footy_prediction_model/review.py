"""Post-matchday review: what happened vs what the scan said, and a running live log.

  python -m footy_prediction_model fetch --dest data --no-history   # results first
  python -m footy_prediction_model review --data data --date 2026-11-08

Reads scan_<date>_legs.csv (what we said before kick-off), settles every market with
the results, and reports:
  1. results with how surprising each one was (market and model chance of what happened)
  2. shortlist bets: won/lost, and hits vs expected
  3. watch list record
  4. model vs market scores for the day (log-loss, 1X2 and O/U 2.5)
  5. running live record across all reviewed days (data/review_log.csv), with a
     tuning verdict that only fires on enough matches
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .watch import grade, settle

LOG_COLS = ["date", "league", "home", "away", "hg", "ag", "q1", "qx", "q2", "qo25",
            "p1", "px", "p2", "po25"]
MIN_LIVE_MATCHES = 300     # below this, no tuning verdict (one season of backtest noise ~ +-0.01)


def _wide(legs: pd.DataFrame) -> pd.DataFrame:
    """One row per match with market (q) and model (p) for 1/X/2/O2.5."""
    keep = legs[legs["key"].isin(["1", "X", "2", "O2.5"])]
    q = keep.pivot_table(index="match_id", columns="key", values="fair_p", aggfunc="first")
    p = keep.pivot_table(index="match_id", columns="key", values="model_p", aggfunc="first")
    out = pd.DataFrame({"q1": q.get("1"), "qx": q.get("X"), "q2": q.get("2"), "qo25": q.get("O2.5"),
                        "p1": p.get("1"), "px": p.get("X"), "p2": p.get("2"), "po25": p.get("O2.5")})
    meta = legs.drop_duplicates("match_id").set_index("match_id")[["league", "fixture", "kickoff"]]
    return meta.join(out)


def _scores(df: pd.DataFrame) -> dict:
    """Log-loss for model and market on settled matches with prices."""
    d = df.dropna(subset=["q1", "hg"])
    if d.empty:
        return {"n": 0}
    res = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    Q, P = d[["q1", "qx", "q2"]].to_numpy(), d[["p1", "px", "p2"]].to_numpy()
    r = np.arange(len(d))
    over = (d.hg + d.ag > 2.5).to_numpy()
    ll = lambda pr, y: -np.mean(np.where(y, np.log(pr), np.log(1 - pr)))  # noqa: E731
    out = {"n": len(d), "market_1x2": float(-np.mean(np.log(Q[r, res]))),
           "model_1x2": float(-np.mean(np.log(P[r, res]))),
           "market_ou": float(ll(d.qo25.to_numpy(), over)), "model_ou": float(ll(d.po25.to_numpy(), over))}
    diff = -np.log(P[r, res]) + np.log(Q[r, res])          # per-match model minus market
    out["diff_se"] = float(diff.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return out


def review(date: pd.Timestamp, legs: pd.DataFrame, matches: pd.DataFrame, data_dir: str) -> str:
    from .report import angles
    date = pd.Timestamp(date).normalize()
    res = matches[(matches["date"] == date) & matches["hg"].notna()]
    legs = legs.copy()
    legs["kickoff"] = pd.to_datetime(legs["kickoff"])
    # attach results by fixture name "Home v Away"
    res = res.assign(fixture=res["home"] + " v " + res["away"])
    legs = legs.merge(res[["fixture", "league", "hg", "ag", "home", "away"]], on=["fixture", "league"], how="left")
    legs["won"] = [settle(k, h, a) for k, h, a in zip(legs["key"].astype(str), legs["hg"], legs["ag"])]
    W = _wide(legs).join(legs.drop_duplicates("match_id").set_index("match_id")[["hg", "ag", "home", "away"]])
    settled = W["hg"].notna()
    L = [f"# Matchday review - {date:%a %d %b %Y}\n",
         f"{int(settled.sum())} of {len(W)} games have results."
         + ("" if settled.all() else " Run `fetch --no-history` later for the rest.") + "\n"]

    # 1. results and surprise
    L.append("## 1. Results\n")
    L.append("| League | Game | Score | Market had it | Model had it | |")
    L.append("|---|---|---|---|---|---|")
    for _, r in W[settled].sort_values(["league", "kickoff"]).iterrows():
        k = 0 if r.hg > r.ag else 1 if r.hg == r.ag else 2
        q = [r.q1, r.qx, r.q2][k]
        p = [r.p1, r.px, r.p2][k]
        tag = "" if pd.isna(q) else ("🔥 upset" if q < 0.25 else "surprise" if q < 0.40 else "")
        qs = "-" if pd.isna(q) else f"{q:.0%}"
        L.append(f"| {r.league} | {r.fixture} | {int(r.hg)}-{int(r.ag)} | {qs} | {p:.0%} | {tag} |")

    # 2. shortlist
    L.append("\n## 2. Shortlist bets (most likely bet in range per game)\n")
    picks = []
    for mid, g in legs.groupby("match_id"):
        like, _ = angles(g)
        if like:
            b = like[0]
            picks.append({**b, "won": g.loc[g["key"] == b["key"], "won"].iloc[0]})
    P = pd.DataFrame(picks)
    if len(P):
        done = P[P["won"].notna()]
        for _, b in P.sort_values("fair_p", ascending=False).iterrows():
            mark = "⏳" if pd.isna(b["won"]) else ("✅" if b["won"] else "❌")
            L.append(f"- {mark} {b['fixture']}: {b['leg']} (chance {b['fair_p']:.0%}, min odds {b['min_odds']:.2f})")
        if len(done):
            exp = done["fair_p"].sum()
            L.append(f"\nWon {int(done['won'].sum())} of {len(done)}; the market expected {exp:.1f}. "
                     f"Return at min odds: {(done['won'] * done['min_odds']).mean() - 1:+.1%} per bet.")
            miss = done[done["won"] == 0]
            if len(miss):
                L.append("Misses: " + "; ".join(f"{m.fixture} {m.leg} ({m.fair_p:.0%})" for m in miss.itertuples())
                         + ". A loss at 80% is expected about 1 game in 5.")

    # 3. watch list
    _, ws = grade(os.path.join(data_dir, "watchlist.csv"), matches)
    L.append("\n## 3. Watch list (no stakes)\n")
    if ws.get("n"):
        L.append(f"All time: {ws['n']} settled, won {ws['won']:.0%} vs {ws['implied']:.0%} implied, "
                 f"return at min odds {ws['roi_min']:+.1%}. Pending {ws['pending']}. "
                 "Needs about 200 settled before it means anything.")
    else:
        L.append("Nothing settled yet.")

    # 4. today's scores + 5. live log
    sc = _scores(W)
    L.append("\n## 4. Model vs market today\n")
    if sc["n"]:
        L.append(f"{sc['n']} priced games. 1X2 log-loss: model {sc['model_1x2']:.3f}, market {sc['market_1x2']:.3f}. "
                 f"O/U 2.5: model {sc['model_ou']:.3f}, market {sc['market_ou']:.3f} (lower is better). "
                 f"With {sc['n']} games the uncertainty on the gap is about ±{2 * sc['diff_se']:.2f}, "
                 "so one day decides nothing.")
    log_path = os.path.join(data_dir, "review_log.csv")
    day = W[settled].assign(date=date)[LOG_COLS]
    if os.path.exists(log_path):
        old = pd.read_csv(log_path, parse_dates=["date"])
        day = pd.concat([old[old["date"] != date], day], ignore_index=True)
    day.to_csv(log_path, index=False)
    live = _scores(day)
    L.append("\n## 5. Live record (all reviewed days)\n")
    if live["n"]:
        gap = live["model_1x2"] - live["market_1x2"]
        L.append(f"{live['n']} priced games over {day['date'].nunique()} day(s). 1X2 log-loss model "
                 f"{live['model_1x2']:.4f} vs market {live['market_1x2']:.4f} (gap {gap:+.4f}, "
                 f"±{2 * live['diff_se']:.4f}). Backtest gap was +0.018.")
        if live["n"] < MIN_LIVE_MATCHES:
            L.append(f"\n**Tuning verdict: not yet.** {live['n']} of the {MIN_LIVE_MATCHES} live games needed "
                     "before a change can be told apart from luck.")
        elif gap - 2 * live["diff_se"] > 0.018:
            L.append("\n**Tuning verdict: the model is doing worse live than in the backtest.** Re-run the "
                     "backtest with fresh data and check xG coverage and team-name mapping.")
        else:
            L.append("\n**Tuning verdict: live performance matches the backtest.** No retune needed.")
    return "\n".join(L)
