"""Scan a matchday: every fixture x every market -> model vs sharp fair price ->
playable legs -> disjoint 6-7 leg parlays.

  python -m footy_prediction_model scan --data data --leagues E0 SP1 I1 D1 F1 G1 CY1

Prices: football-data.co.uk fixtures.csv (free, refreshed before each round): Betfair
Exchange (near margin-free, used as "fair") + market average/max for 1X2 and O/U 2.5.
Other goals markets (O1.5, U3.5, BTTS, team to score) are priced from the expected
goals those fair prices imply. Leagues missing from fixtures.csv (CY1, Europe) come
from open-matchday fixtures without prices -> model only, low confidence.

Stoiximan itself can't be read automatically (bot protection), so every leg gets a
"min odds": don't take the leg on Stoiximan below that price.
"""
from __future__ import annotations

import difflib
import os

import numpy as np
import pandas as pd

from .config import FLAG_EMOJI, ModelConfig
from .data import EURO_CODES
from .features import build_features
from .markets import devig, market_probs
from .model import FootyModel

FD_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
UK_TO_CY = pd.Timedelta(hours=2)   # UK and Cyprus change clocks on the same days

# (label shown, market key)
LEG_MARKETS = [("1", "1"), ("2", "2"), ("1X", "1X"), ("X2", "X2"), ("12", "12"),
               ("Over 1.5", "O1.5"), ("Over 2.5", "O2.5"), ("Under 2.5", "U2.5"),
               ("Under 3.5", "U3.5"), ("Under 4.5", "U4.5"),
               ("BTTS Yes", "BTTS_Y"), ("BTTS No", "BTTS_N"),
               ("Home to score", "H_SCORE"), ("Away to score", "A_SCORE")]
UNDER_LIKE = {"U2.5", "U3.5", "U4.5", "BTTS_N"}
MIN_FAIR_ODDS = 1.19            # odds floor: shorter legs add little to a parlay
MAX_FAIR_ODDS = 1.43            # fair prob 70%: the long end of the backtested "safe leg" band
MAX_MARGIN = 1.05               # min odds = fair odds / 1.05 (accept up to ~5% bookie margin)
PARLAY_SIZE = (6, 7)


# ------------------------------------------------------------------ fixtures + prices
def fetch_fd_fixtures(dest: str = "data") -> str:
    import requests
    out = os.path.join(dest, "fixtures", "fd_fixtures.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = requests.get(FD_FIXTURES_URL, timeout=60)
    r.raise_for_status()
    with open(out, "wb") as f:
        f.write(r.content)
    return out


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)


def load_fd_fixtures(path: str, date: pd.Timestamp, leagues: list[str]) -> pd.DataFrame:
    f = pd.read_csv(path, encoding_errors="replace")
    f.columns = [c.lstrip("﻿") for c in f.columns]
    f["date"] = pd.to_datetime(f["Date"], dayfirst=True, errors="coerce")
    f = f[(f["date"] == date) & f["Div"].isin(leagues)].copy()
    ko = pd.to_datetime(f["Date"] + " " + f["Time"].fillna("12:00"), dayfirst=True, errors="coerce")
    out = pd.DataFrame({"league": f["Div"], "date": f["date"], "kickoff": ko + UK_TO_CY,
                        "home": f["HomeTeam"].str.strip(), "away": f["AwayTeam"].str.strip()})
    for k, src in (("bfe", "BFE"), ("avg", "Avg"), ("max", "Max")):
        out[f"{k}_1"], out[f"{k}_x"], out[f"{k}_2"] = (_num(f, f"{src}H"), _num(f, f"{src}D"),
                                                       _num(f, f"{src}A"))
        out[f"{k}_o"], out[f"{k}_u"] = _num(f, f"{src}>2.5"), _num(f, f"{src}<2.5")
    return out.reset_index(drop=True)


def load_om_fixtures(data_dir: str, leagues: list[str], date: pd.Timestamp,
                     name_map: dict) -> pd.DataFrame:
    """Unpriced fixtures from open-matchday (fetched with `fetch`)."""
    from .fetch import CODE_TO_FOLDER, OM_BASE, _get
    recs = []
    for lg in leagues:
        folder = CODE_TO_FOLDER.get(lg)
        if not folder:
            continue
        p = os.path.join(data_dir, folder, "fixtures_2627.csv")
        try:
            _get(f"{OM_BASE}/{folder}/fixtures_2627.csv", p)
        except Exception:  # noqa: BLE001
            if not os.path.exists(p):
                continue
        fx = pd.read_csv(p)
        fx = fx[pd.to_datetime(fx["date"]) == date]
        for _, r in fx.iterrows():
            pre = "[EU] " if lg in EURO_CODES else ""
            h = name_map.get(pre + r["home"], name_map.get(r["home"], r["home"]))
            a = name_map.get(pre + r["away"], name_map.get(r["away"], r["away"]))
            t = r["time"] if isinstance(r.get("time"), str) else "00:00"
            recs.append({"league": lg, "date": date, "home": h, "away": a,
                         "kickoff": pd.Timestamp(f"{date.date()} {t}") if t != "00:00" else pd.NaT})
    return pd.DataFrame(recs)


def _match_names(fx: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map fixture names to our team names (exact, then fuzzy within the league)."""
    notes = []
    fx = fx.copy()
    for lg, g in fx.groupby("league"):
        cur = matches[(matches["league"] == lg)]
        cur = cur[cur["season"] >= cur["season"].max() - 1]
        known = sorted(set(cur["home"]) | set(cur["away"]))
        for col in ("home", "away"):
            for i in g.index:
                n = fx.at[i, col]
                if n in known or f"{n} ({lg})" in known:
                    fx.at[i, col] = n if n in known else f"{n} ({lg})"
                    continue
                m = difflib.get_close_matches(n, known, n=1, cutoff=0.6)
                if m:
                    notes.append(f"name '{n}' -> '{m[0]}' ({lg})")
                    fx.at[i, col] = m[0]
                else:
                    notes.append(f"unknown team '{n}' ({lg}) - no data")
    return fx, notes


# ------------------------------------------------------------------------- pricing
def _fair_markets(row, model: FootyModel) -> tuple[dict | None, str]:
    """Fair probabilities for every market from the sharpest available 1X2 + O/U 2.5."""
    from .anchored import market_lambdas

    def ok(xs):
        return all(pd.notna(x) and x > 1 for x in xs)

    chosen, srcs = [], []
    for keys in (("1", "x", "2"), ("o", "u")):
        bfe = [row.get(f"bfe_{k}") for k in keys]
        avg = [row.get(f"avg_{k}") for k in keys]
        # Exchange is sharpest, but thin markets can show stale quotes (e.g. Under 2.5
        # at 1.13 when every bookie has 1.74): fall back to the average if they disagree.
        if ok(bfe) and (not ok(avg) or abs(devig(bfe)[0] - devig(avg)[0]) <= 0.05):
            chosen += bfe
            srcs.append("Betfair Exch")
        elif ok(avg):
            chosen += avg
            srcs.append("market avg")
        else:
            return None, "no prices"
    lh, la = market_lambdas(*[np.array([x]) for x in chosen])[0]
    if not np.isfinite(lh):
        return None, "no prices"
    P = market_probs(model.score_matrix(lh, la))
    p1, px, p2 = devig(chosen[:3])
    po = devig(chosen[3:])[0]
    P.update({"1": p1, "X": px, "2": p2, "1X": p1 + px, "X2": px + p2, "12": p1 + p2,
              "O2.5": po, "U2.5": 1 - po, "_lh": lh, "_la": la})
    return P, srcs[0] if srcs[0] == srcs[1] else "mixed"


def _verdict(model_p, anch_p, fair_p, fair_odds):
    if fair_p is None or np.isnan(fair_p):
        return "NODATA"
    em = model_p - fair_p
    ea = (anch_p - fair_p) if anch_p is not None and not np.isnan(anch_p) else em
    if em <= -0.10 and ea < 0:
        v = "RED"
    elif em <= -0.05 and ea < 0:
        v = "ORANGE"
    elif em >= 0 and ea >= 0:
        v = "GREEN"
    else:
        v = "YELLOW"
    if fair_odds > 1.50 and v == "GREEN":
        v = "YELLOW"
    return v


def _conceded_last10(matches: pd.DataFrame) -> dict[str, float]:
    played = matches[matches["hg"].notna()].sort_values("date")
    long = pd.concat([
        pd.DataFrame({"team": played["home"], "date": played["date"], "ga": played["ag"]}),
        pd.DataFrame({"team": played["away"], "date": played["date"], "ga": played["hg"]})])
    return long.sort_values("date").groupby("team")["ga"].apply(lambda s: s.tail(10).mean()).to_dict()


def scan(matches: pd.DataFrame, fixtures: pd.DataFrame, cfg: ModelConfig | None = None,
         ref_date=None, min_fair_odds: float = MIN_FAIR_ODDS,
         max_fair_odds: float = MAX_FAIR_ODDS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (fixture view, all legs, summary). Playable legs have fair odds in
    [min_fair_odds, max_fair_odds]."""
    band = (1 / max_fair_odds, 1 / min_fair_odds)
    from .anchored import AnchoredModel, attach_market
    cfg = cfg or ModelConfig()
    fx = fixtures.reset_index(drop=True).copy()
    fx["match_id"] = np.arange(len(fx)) + int(matches["match_id"].max()) + 1
    fx["season"] = np.where(fx["date"].dt.month >= 7, fx["date"].dt.year, fx["date"].dt.year - 1)
    add = fx[["match_id", "season", "league", "date", "home", "away"]].copy()
    for c in matches.columns:
        if c not in add.columns:
            add[c] = np.nan
    allm = pd.concat([matches, add[matches.columns]], ignore_index=True)
    rows = build_features(allm)
    ref = pd.Timestamp(ref_date) if ref_date else fx["date"].min().normalize()
    model = FootyModel(cfg).fit(rows, ref)
    pred = model.predict_matches(rows[rows["match_id"].isin(fx["match_id"])])
    am = AnchoredModel().fit(attach_market(rows, allm), ref)
    ga10 = _conceded_last10(matches)
    hist_n = matches[matches["hg"].notna()].groupby("league").size()

    views, legs = [], []
    for _, f in fx.iterrows():
        mid = f["match_id"]
        name = f"{f['home']} v {f['away']}"
        known = mid in pred.index and bool(pred.loc[mid, "known"])
        if not known:
            views.append({"kickoff": f["kickoff"], "league": f["league"], "fixture": name,
                          "note": "team(s) not in data"})
            continue
        r = pred.loc[mid]
        M = model.score_matrix(r["lam_h"], r["lam_a"])
        PM = market_probs(M)
        fair, src = _fair_markets(f, model)
        PA = None
        if fair is not None:
            rr = rows[rows["match_id"] == mid].sort_values("is_home", ascending=False).copy()
            rr["mkt_lam"] = [fair["_lh"], fair["_la"]]
            lh, la = am.lambdas(rr)
            PA = market_probs(model.score_matrix(lh, la))
        top = np.dstack(np.unravel_index(np.argsort(M.ravel())[::-1][:3], M.shape))[0]
        thin = hist_n.get(f["league"], 0) < 600
        views.append({
            "kickoff": f["kickoff"], "league": f["league"], "fixture": name,
            "xg": f"{PM['xg_h']:.1f}-{PM['xg_a']:.1f}",
            "scores": ", ".join(f"{i}-{j} {M[i, j]:.0%}" for i, j in top),
            "p1": PM["1"], "px": PM["X"], "p2": PM["2"], "po25": PM["O2.5"], "pbtts": PM["BTTS_Y"],
            "f1": fair["1"] if fair else np.nan, "fx": fair["X"] if fair else np.nan,
            "f2": fair["2"] if fair else np.nan, "fo25": fair["O2.5"] if fair else np.nan,
            "price_src": src, "thin": thin,
            "ga10_h": ga10.get(f["home"], np.nan), "ga10_a": ga10.get(f["away"], np.nan),
        })
        for label, key in LEG_MARKETS:
            fp = fair[key] if fair else np.nan
            fo = 1 / fp if fair else np.nan
            mp = PM[key]
            ap = PA[key] if PA else np.nan
            # Risk = the sharp market's probability. The 2024-26 backtest showed model
            # disagreement does not predict which short legs lose, so it's an alert only.
            use = fp if fair else (ap if PA else mp)
            v = _verdict(mp, ap, fp, fo)
            reasons = []
            if fair is None:
                reasons.append("no prices")
            elif not (band[0] <= fp <= band[1]):
                reasons.append(f"outside {min_fair_odds:.2f}-{max_fair_odds:.2f} band")
            if key in UNDER_LIKE:
                bad = [t for t in (f["home"], f["away"]) if ga10.get(t, 0) >= 2.0]
                if bad:
                    reasons.append(f"v3.1: {', '.join(bad)} concede 2+/game (last 10)")
            if thin:
                reasons.append("thin data")
            avg_key = {"1": "avg_1", "2": "avg_2", "O2.5": "avg_o", "U2.5": "avg_u"}.get(key)
            legs.append({
                "match_id": mid, "kickoff": f["kickoff"], "league": f["league"], "fixture": name,
                "leg": label if label not in ("Home to score", "Away to score")
                else f"{f['home'] if label.startswith('Home') else f['away']} to score",
                "key": key, "fair_p": fp, "fair_odds": fo,
                "min_odds": np.floor(100 * fo / MAX_MARGIN) / 100 if fair else np.nan,
                "avg_odds": f.get(avg_key, np.nan) if avg_key else np.nan,
                "model_p": mp, "anch_p": ap, "p": use, "edge": mp - fp if fair else np.nan,
                "model_edge": mp - fp if fair else np.nan,
                "verdict": v, "playable": not reasons, "reason": "; ".join(reasons),
            })
    return pd.DataFrame(views), pd.DataFrame(legs), {"ref": ref, "rho": model.rho_, "band": band,
                                                     "xg_share": getattr(model, "xg_share_", 0.0)}


# ------------------------------------------------------------------------ parlays
def pick_legs(legs: pd.DataFrame) -> pd.DataFrame:
    """One leg per fixture: the most likely playable leg (ties -> bigger edge)."""
    ok = legs[legs["playable"]].sort_values(["p", "edge"], ascending=False)
    return ok.groupby("match_id", sort=False).head(1).sort_values("p", ascending=False)


def build_parlays(picks: pd.DataFrame, size=PARLAY_SIZE) -> list[pd.DataFrame]:
    """Disjoint parlays of 6-7 legs, safest first (Conservative, Balanced..., Aggressive)."""
    n = len(picks)
    if n == 0:
        return []
    if n < size[0]:
        return [picks]
    k = max(1, n // size[0])
    while n / k > size[1]:
        k += 1
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, i = [], 0
    for s in sizes:
        out.append(picks.iloc[i:i + s])
        i += s
    return out


def tier_names(k: int) -> list[str]:
    if k == 1:
        return ["Conservative"]
    if k == 2:
        return ["Conservative", "Aggressive"]
    return ["Conservative"] + [f"Balanced {i}" if k > 3 else "Balanced" for i in range(1, k - 1)] + ["Aggressive"]


# ------------------------------------------------------------------------- report
def _pct(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0%}"


def format_scan(views: pd.DataFrame, legs: pd.DataFrame, summary: dict, notes: list[str]) -> str:
    L = []
    day = summary["ref"].strftime("%a %d %b %Y")
    L.append(f"# Matchday scan - {day}\n")
    L.append("_Fair = Betfair Exchange price with the margin removed: the sharpest price we can "
             "read, and the probability used for every leg and parlay. Model = our xG-trained goals "
             "model, shown as a second opinion. Kick-offs in Cyprus time._\n")

    # 1. most likely scenario
    L.append("## 1. Most likely scenario\n")
    L.append("| KO | Lg | Fixture | Model xG | Top scores | 1 / X / 2 model (fair) | O2.5 model (fair) | BTTS |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, v in views.sort_values("kickoff").iterrows():
        ko = v["kickoff"].strftime("%H:%M") if pd.notna(v["kickoff"]) else "?"
        if "xg" not in v or pd.isna(v.get("xg")):
            L.append(f"| {ko} | {v['league']} | {v['fixture']} | - | - | {v.get('note', '')} | - | - |")
            continue
        fair = (f" ({_pct(v['f1'])}/{_pct(v['fx'])}/{_pct(v['f2'])})" if pd.notna(v["f1"]) else " (no prices)")
        L.append(f"| {ko} | {v['league']} | {v['fixture']} | {v['xg']} | {v['scores']} | "
                 f"{_pct(v['p1'])}/{_pct(v['px'])}/{_pct(v['p2'])}{fair} | "
                 f"{_pct(v['po25'])} ({_pct(v['fo25'])}) | {_pct(v['pbtts'])} |")

    # 2. legs
    picks = pick_legs(legs)
    L.append("\n## 2. Best leg per fixture\n")
    lo, hi = 1 / summary["band"][1], 1 / summary["band"][0]
    L.append(f"Legs with fair odds {lo:.2f}-{hi:.2f} that pass v3.1; the most likely one per fixture. "
             "**Min odds** = don't take it on Stoiximan below this (fair odds less a 5% margin).\n")
    L.append("| KO | Fixture | Leg | Fair odds | Min odds | Fair | Model | Anchored | Flag |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, x in picks.sort_values("kickoff").iterrows():
        ko = x["kickoff"].strftime("%H:%M") if pd.notna(x["kickoff"]) else "?"
        L.append(f"| {ko} | {x['fixture']} | {x['leg']} | {x['fair_odds']:.2f} | **{x['min_odds']:.2f}** | "
                 f"{_pct(x['fair_p'])} | {_pct(x['model_p'])} | {_pct(x['anch_p'])} | "
                 f"{FLAG_EMOJI[x['verdict']]} |")

    # model disagrees: an alert to check team news, not a reason to drop the leg
    risky = legs[(legs["verdict"].isin(["ORANGE", "RED"])) & legs["fair_p"].between(*summary["band"])]
    if len(risky):
        L.append("\n**Model disagrees - check team news before playing** (in the backtest these legs "
                 "still hit at the market rate, so this is an alert, not a veto):\n")
        for _, x in risky.sort_values("model_edge").head(12).iterrows():
            L.append(f"- {FLAG_EMOJI[x['verdict']]} {x['fixture']} - {x['leg']} @ fair {x['fair_odds']:.2f}: "
                     f"model {_pct(x['model_p'])} vs fair {_pct(x['fair_p'])} "
                     f"(model's own odds {1 / x['model_p']:.2f})")

    # 3. parlays
    parlays = build_parlays(picks)
    L.append("\n## 3. Parlays (disjoint: each fixture on one ticket only)\n")
    if not parlays:
        L.append("No playable legs.")
    for name, p in zip(tier_names(len(parlays)), parlays):
        p_all, f_all = float(np.prod(p["p"])), float(np.prod(p["fair_p"]))
        odds_min, odds_fair = float(np.prod(p["min_odds"])), float(np.prod(p["fair_odds"]))
        weak = p.sort_values("p").iloc[0]
        L.append(f"**{name}** - {len(p)} legs | odds {odds_min:.2f} (at min) to {odds_fair:.2f} (fair) | "
                 f"P(all hit) {p_all:.0%} (fair {f_all:.0%}) | weakest: {weak['fixture']} {weak['leg']} "
                 f"({_pct(weak['p'])})")
        for _, x in p.iterrows():
            L.append(f"  - {x['fixture']} - **{x['leg']}** (min {x['min_odds']:.2f})")
        L.append("")

    # 4. exclusions
    L.append("## 4. Excluded fixtures\n")
    chosen = set(picks["match_id"])
    for mid, g in legs.groupby("match_id"):
        if mid in chosen:
            continue
        inb = g[g["fair_p"].between(*summary["band"])]
        why = sorted(set("; ".join(inb["reason"]).split("; ")) - {""}) if len(inb) else \
            [g["reason"].iloc[0] or f"no leg priced {lo:.2f}-{hi:.2f}"]
        L.append(f"- {g['fixture'].iloc[0]}: {', '.join(why) or 'no playable leg'}")
    for _, v in views[views.get("note", pd.Series(dtype=str)).notna()].iterrows() if "note" in views else []:
        L.append(f"- {v['fixture']}: {v['note']}")

    if notes:
        L.append("\n## Notes\n")
        L.extend(f"- {n}" for n in notes)
    L.append("\n_Backtest (2024-26, 3,491 matches): legs at fair 1.19-1.43 hit 75-79% whatever the "
             "model said, so P(all hit) uses the fair price. Where you add value: take legs only at or "
             "above Min odds, and check line-ups before placing._")
    return "\n".join(L)
