"""Score a betting slip.

Slip CSV columns (required): league,date,home,away,selection,odds
Optional Gate 3 columns (0 if blank):
  home_absence_att, away_absence_att   share 0-1 of attacking output missing
  home_absence_def, away_absence_def   share 0-1 of first-choice defence missing
  home_gk_out, away_gk_out             1 if first-choice GK out
  home_euro_midweek, away_euro_midweek 1 if played in Europe 2-3 days before
Optional market columns for exact de-margining:
  market_odds  e.g. "1.30|5.50|9.00" (all outcomes of the same market, any order
               containing the leg's own price)
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

from .config import FLAG_EMOJI, FEATURE_GROUPS, FlagConfig, ModelConfig
from .features import build_features
from .markets import devig, fair_single, flag, market_probs, parse_selection
from .model import FootyModel

GATE_LABELS = {"ratings": "team ratings+home", **{g: g for g in FEATURE_GROUPS}, "G3_manual": "G3_manual"}


def _suggest(name, known):
    m = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    return f" (did you mean: {', '.join(m)}?)" if m else ""


def score_slip(matches: pd.DataFrame, slip: pd.DataFrame, cfg: ModelConfig | None = None,
               fc: FlagConfig | None = None, managers: dict | None = None,
               ref_date=None) -> tuple[pd.DataFrame, dict]:
    cfg, fc = cfg or ModelConfig(), fc or FlagConfig()
    slip = slip.copy()
    slip["date"] = pd.to_datetime(slip["date"], dayfirst=False)
    known_names = set(matches["home"]) | set(matches["away"])
    warnings = []
    for c in ("home", "away"):
        slip[c] = slip[c].astype(str).str.strip()
        fixed = []
        for n, lg in zip(slip[c], slip["league"]):
            if n not in known_names and f"{n} ({lg})" in known_names:
                n = f"{n} ({lg})"
            elif n not in known_names:
                warnings.append(f"Team not in data: '{n}'{_suggest(n, list(known_names))}")
            fixed.append(n)
        slip[c] = fixed

    # append fixtures as unplayed matches
    fx = pd.DataFrame({
        "match_id": np.arange(len(slip)) + int(matches["match_id"].max()) + 1,
        "season": np.where(slip["date"].dt.month >= 7, slip["date"].dt.year, slip["date"].dt.year - 1),
        "league": slip["league"], "date": slip["date"], "home": slip["home"], "away": slip["away"],
    })
    for c in matches.columns:
        if c not in fx.columns:
            fx[c] = np.nan
    slip["match_id"] = fx["match_id"].to_numpy()
    allm = pd.concat([matches, fx[matches.columns]], ignore_index=True)
    manual = slip.set_index("match_id")
    rows = build_features(allm, managers=managers, manual=manual)

    ref = pd.Timestamp(ref_date) if ref_date else slip["date"].min().normalize()
    last_data = matches.loc[matches["hg"].notna(), "date"].max()
    if (ref - last_data).days > 10:
        warnings.append(f"Latest result in data is {last_data.date()} - {(ref - last_data).days} days "
                        "before the slip. Form features are stale: load current-season CSVs.")
    hist_n = matches[matches["hg"].notna()].groupby("league").size()
    for lg in sorted(set(slip["league"])):
        if hist_n.get(lg, 0) < 600:
            warnings.append(f"Thin data for {lg}: only {hist_n.get(lg, 0)} matches - treat its legs "
                            "as low confidence.")
    model = FootyModel(cfg).fit(rows, ref)
    pred = model.predict_matches(rows[rows["match_id"].isin(slip["match_id"])])

    # Market-anchored probabilities when the fixture's 1X2 + O/U2.5 prices are supplied
    anch = {}
    acols = ["odds_1", "odds_x", "odds_2", "odds_o25", "odds_u25"]
    if all(c in slip.columns for c in acols) and slip[acols].notna().all(1).any():
        from .anchored import AnchoredModel, attach_market, market_lambdas
        hist = attach_market(rows, allm)
        am = AnchoredModel().fit(hist, ref)
        lam = market_lambdas(*[slip[c] for c in acols])
        for (_, leg), (lh, la) in zip(slip.iterrows(), lam):
            if np.isfinite(lh):
                rr = rows[rows["match_id"] == leg["match_id"]].sort_values("is_home", ascending=False).copy()
                rr["mkt_lam"] = [lh, la]
                anch[leg["match_id"]] = am.lambdas(rr)

    out = []
    for _, leg in slip.iterrows():
        mid = leg["match_id"]
        rec = {"fixture": f"{leg['home']} v {leg['away']}", "selection": leg["selection"],
               "odds": float(leg["odds"])}
        if "market_odds" in leg and isinstance(leg.get("market_odds"), str) and leg["market_odds"]:
            mk = [float(x) for x in leg["market_odds"].split("|")]
            fair = devig(mk)[int(np.argmin([abs(x - rec["odds"]) for x in mk]))]
        else:
            fair = fair_single(rec["odds"], fc.default_overround)
        rec["fair_p"] = fair
        r = pred.loc[mid] if mid in pred.index else None
        if r is None or not bool(r["known"]):
            rec.update(model_p=np.nan, edge=np.nan, flag="NODATA",
                       why="team(s) not in training data - judge manually")
            out.append(rec)
            continue
        M = model.score_matrix(r["lam_h"], r["lam_a"])
        P = market_probs(M)
        key = parse_selection(str(leg["selection"]), leg["home"], leg["away"])
        p = P[key]
        rec.update(market=key, model_p=p, edge=p - fair,
                   flag=flag(p, fair, rec["odds"], fc), xg_h=P["xg_h"], xg_a=P["xg_a"])
        if mid in anch:
            lh, la = anch[mid]
            pa = market_probs(model.score_matrix(lh, la))[key]
            rec.update(anchored_p=pa, anchored_edge=pa - fair)
        # explanation: gate contributions as % change in each side's expected goals
        parts = []
        for g in ["ratings"] + list(FEATURE_GROUPS) + ["G3_manual"]:
            dh, da = r[f"{g}_h"], r[f"{g}_a"]
            if abs(dh) > 0.03 or abs(da) > 0.03:
                parts.append(f"{g}: H {100*(np.exp(dh)-1):+.0f}% / A {100*(np.exp(da)-1):+.0f}%")
        rec["why"] = "; ".join(parts) if parts else "ratings only"
        out.append(rec)

    res = pd.DataFrame(out)
    ok = res["model_p"].notna()
    summary = {
        "legs": len(res),
        "combined_odds": float(np.prod(res["odds"])),
        "model_p_all_hit": float(np.prod(res.loc[ok, "model_p"])) if ok.any() else np.nan,
        "fair_p_all_hit": float(np.prod(res["fair_p"])),
        "legs_without_model": int((~ok).sum()),
        "warnings": warnings,
    }
    # legs without a model prob use the fair price
    p_all = float(np.prod(np.where(ok, res["model_p"], res["fair_p"])))
    summary["p_all_hit_used"] = p_all
    summary["ev_per_unit"] = p_all * summary["combined_odds"] - 1
    if ok.any():
        wl = res.loc[ok].sort_values(["edge", "model_p"]).iloc[0]
        summary["weakest_link"] = f"{wl['fixture']} {wl['selection']} (model {wl['model_p']:.0%}, edge {wl['edge']:+.1%})"
    if len(res) > fc.max_parlay_legs:
        summary["warnings"].append(f"{len(res)} legs > {fc.max_parlay_legs}: P(all hit) is "
                                   f"{p_all:.1%}. Consider splitting.")
    return res, summary


def format_report(res: pd.DataFrame, summary: dict) -> str:
    has_a = "anchored_p" in res.columns
    hdr = "| # | Fixture | Pick | Odds | Fair | Model | Edge | Flag |" + (" Anchored |" if has_a else "")
    lines = [hdr, "|---" * (8 + has_a) + "|"]
    for i, r in res.reset_index(drop=True).iterrows():
        mp = "-" if pd.isna(r["model_p"]) else f"{r['model_p']:.0%}"
        ed = "-" if pd.isna(r["edge"]) else f"{r['edge']:+.1%}"
        line = (f"| {i+1} | {r['fixture']} | {r['selection']} | {r['odds']:.2f} | "
                f"{r['fair_p']:.0%} | {mp} | {ed} | {FLAG_EMOJI[r['flag']]} |")
        if has_a:
            line += " - |" if pd.isna(r.get("anchored_p")) else f" {r['anchored_p']:.0%} |"
        lines.append(line)
    lines.append("")
    lines.append("**Why (gate contributions to expected goals):**")
    for i, r in res.reset_index(drop=True).iterrows():
        lines.append(f"{i+1}. {r['fixture']}: {r['why']}")
    lines.append("")
    lines.append(f"Legs {summary['legs']} | combined odds {summary['combined_odds']:.2f} | "
                 f"P(all hit) model {summary['p_all_hit_used']:.1%} vs bookie fair "
                 f"{summary['fair_p_all_hit']:.1%} | EV {summary['ev_per_unit']:+.1%}")
    if "weakest_link" in summary:
        lines.append(f"Weakest link: {summary['weakest_link']}")
    for w in summary["warnings"]:
        lines.append(f"⚠️ {w}")
    lines.append("")
    lines.append("_v0.1 caveat: in the 2024-26 backtest, legs priced 1.15-1.40 hit 77-79% whatever "
                 "the flag. Treat flags as 'model vs market disagreement - check team news', not as "
                 "a reason to drop a leg. The parlay maths (P(all hit), EV) is reliable._")
    return "\n".join(lines)
