"""Leak-free feature builder.

Walks matches in date order. For every match it first computes features from
the state BEFORE kick-off, then (if the match has a result) updates the state.
Unplayed fixtures (goals NaN) therefore get features as of "now", which is how
live slips are scored.

Output is a long frame with two rows per match: one for the home side scoring,
one for the away side scoring.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .config import (DEFAULT_EUROPE_LINE, DEFAULT_RELEGATION_SPOTS, EUROPE_LINE,
                     LEARNED_FEATURES, MANUAL_FEATURES, RELEGATION_SPOTS)
from .data import EURO_CODES, xg_proxy

A_SHORT = 2 / (5 + 1)    # EWMA over ~5 matches
A_LONG = 2 / (30 + 1)    # EWMA over ~30 matches
ELO_K, ELO_HFA = 20.0, 65.0


class _Team:
    __slots__ = ("last_date", "n", "xgf_s", "xgf_l", "xga_s", "xga_l", "gf_l", "streak", "elo")

    def __init__(self):
        self.last_date = None
        self.n = 0
        self.xgf_s = self.xgf_l = self.xga_s = self.xga_l = self.gf_l = None
        self.streak = 0
        self.elo = None

    @staticmethod
    def _ewm(old, x, a):
        return x if old is None else old + a * (x - old)

    def update(self, date, gf, ga, xgf, xga):
        self.xgf_s = self._ewm(self.xgf_s, xgf, A_SHORT)
        self.xgf_l = self._ewm(self.xgf_l, xgf, A_LONG)
        self.xga_s = self._ewm(self.xga_s, xga, A_SHORT)
        self.xga_l = self._ewm(self.xga_l, xga, A_LONG)
        self.gf_l = self._ewm(self.gf_l, gf, A_LONG)
        if gf > ga:
            self.streak = self.streak + 1 if self.streak > 0 else 1
        elif gf < ga:
            self.streak = self.streak - 1 if self.streak < 0 else -1
        else:
            self.streak = 0
        self.last_date = date
        self.n += 1


def _form(short, long, n):
    if n < 3 or short is None or long is None:
        return 0.0
    return math.log((short + 0.3) / (long + 0.3))


def _stakes(team, table, played, n_teams, league):
    """Motivation: how close the team is to a live table line (title, Europe,
    relegation), scaled by season progress. Returns (stakes, dead_rubber)."""
    total = 2 * (n_teams - 1)
    gp = played.get(team, 0)
    if n_teams < 6 or gp == 0:
        return 0.0, 0.0
    gl = max(total - gp, 0)
    ranked = sorted(table.items(), key=lambda kv: -kv[1])
    pts_by_pos = [p for _, p in ranked]
    pos = {t: i + 1 for i, (t, _) in enumerate(ranked)}
    p, r = table.get(team, 0), pos.get(team, n_teams)
    lines = [1, EUROPE_LINE.get(league, DEFAULT_EUROPE_LINE),
             n_teams - RELEGATION_SPOTS.get(league, DEFAULT_RELEGATION_SPOTS)]
    best, any_live = 0.0, False
    for k in lines:
        if k >= len(pts_by_pos):
            continue
        gap = p - pts_by_pos[k] if r <= k else pts_by_pos[k - 1] - p
        gap = max(gap, 0)
        if gap <= 3 * gl:
            any_live = True
            best = max(best, math.exp(-gap / max(gl, 1)))
    progress = gp / total
    dead = 1.0 if (gl <= 10 and not any_live) else 0.0
    return best * progress, dead


def build_features(matches: pd.DataFrame, managers: dict | None = None,
                   manual: pd.DataFrame | None = None) -> pd.DataFrame:
    """matches: output of data.load_matches (optionally with unplayed fixtures
    appended, goals NaN). manual: optional frame indexed by match_id with Gate 3
    columns home_absence_att, away_absence_att, home_absence_def, away_absence_def,
    home_gk_out, away_gk_out, home_euro_midweek, away_euro_midweek."""
    managers = managers or {}
    m = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    hx = np.where(m["home_xg"].notna(), m["home_xg"], xg_proxy(m["hg"], m["hs"], m["hst"]))
    ax = np.where(m["away_xg"].notna(), m["away_xg"], xg_proxy(m["ag"], m["as_"], m["ast"]))

    # league-season membership for promoted flag and table sizes
    members = m.groupby(["league", "season"])["home"].agg(lambda s: set(s)).to_dict()
    for (lg, se), aw in m.groupby(["league", "season"])["away"]:
        members[(lg, se)] |= set(aw)

    teams: dict[str, _Team] = defaultdict(_Team)
    tables = defaultdict(lambda: defaultdict(int))
    played = defaultdict(lambda: defaultdict(int))
    h2h = defaultdict(lambda: deque(maxlen=6))
    last_euro: dict = {}
    rows = []

    for i, r in enumerate(m.itertuples(index=False)):
        lg, se, d = r.league, r.season, r.date
        H, A = teams[r.home], teams[r.away]
        # Elo: seed / overwrite from data when present, otherwise internal
        if not pd.isna(r.home_elo):
            H.elo = float(r.home_elo)
        if not pd.isna(r.away_elo):
            A.elo = float(r.away_elo)
        if H.elo is None:   # leagues without ClubElo: start at 1500 (within-league scale)
            H.elo = 1500.0
        if A.elo is None:
            A.elo = 1500.0
        is_euro = lg in EURO_CODES
        n_teams = len(members.get((lg, se), ()))
        prev = members.get((lg, se - 1))

        side_feats = {}
        for side, T, O, tname, oname in (("h", H, A, r.home, r.away), ("a", A, H, r.away, r.home)):
            if is_euro:
                st, dead = 0.0, 0.0
            else:
                st, dead = _stakes(tname, tables[(lg, se)], played[(lg, se)], n_teams, lg)
            le = last_euro.get(tname)
            euro_recent = 1.0 if (not is_euro and le is not None and 0 < (d - le).days <= 4) else 0.0
            mg = managers.get(tname, [])
            past = [x for x in mg if x <= d]
            nm = 0.0
            if past:
                days = (d - past[-1]).days
                nm = math.exp(-days / 30) if days <= 90 else 0.0
            rest = 0.0
            if T.last_date is not None:
                rest = (min(max((d - T.last_date).days, 3), 14) - 7) / 7
            hist = [x for x in h2h[(tname, oname)] if (d - x[0]).days <= 6 * 365]
            h2h_res = sum(x[1] for x in hist) / (len(hist) + 4) if hist else 0.0
            side_feats[side] = dict(
                xg_form_att=_form(T.xgf_s, T.xgf_l, T.n),
                xg_form_def=_form(T.xga_s, T.xga_l, T.n),
                streak=max(min(T.streak, 5), -5) / 5,
                promoted=1.0 if (prev is not None and tname not in prev) else 0.0,
                h2h=h2h_res, rest=rest, stakes=st, dead=dead, new_mgr=nm,
                elo=T.elo, euro=euro_recent,
            )

        for side, oside, is_home, tname, oname, goals in (
                ("h", "a", 1, r.home, r.away, r.hg), ("a", "h", 0, r.away, r.home, r.ag)):
            f, o = side_feats[side], side_feats[oside]
            elo_diff = 0.0
            if f["elo"] is not None and o["elo"] is not None:
                elo_diff = (f["elo"] - o["elo"]) / 100
            rows.append({
                "match_id": r.match_id, "date": d, "league": lg, "season": se,
                "team": tname, "opp": oname, "is_home": is_home, "goals": goals,
                "att_xg_form": f["xg_form_att"], "opp_def_xg_form": o["xg_form_def"],
                "streak": f["streak"], "opp_streak": o["streak"],
                "promoted": f["promoted"], "opp_promoted": o["promoted"],
                "h2h_resid": f["h2h"], "rest": f["rest"], "opp_rest": o["rest"],
                "stakes": f["stakes"], "opp_stakes": o["stakes"],
                "dead_rubber": f["dead"], "opp_dead_rubber": o["dead"],
                "new_mgr": f["new_mgr"], "opp_new_mgr": o["new_mgr"],
                "elo_diff": elo_diff,
                "has_history": int(teams[tname].n > 0 and teams[oname].n > 0),
                "_euro_recent": f["euro"], "_opp_euro_recent": o["euro"],
            })

        # ---- update state with the result (only if played) ----
        if pd.isna(r.hg) or pd.isna(r.ag):
            continue
        hg, ag = float(r.hg), float(r.ag)
        hres = hg - (H.gf_l if H.gf_l is not None else hg)
        ares = ag - (A.gf_l if A.gf_l is not None else ag)
        h2h[(r.home, r.away)].append((d, hres))
        h2h[(r.away, r.home)].append((d, ares))
        if H.elo is not None and A.elo is not None:
            exp_h = 1 / (1 + 10 ** (-(H.elo + ELO_HFA - A.elo) / 400))
            res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            mult = math.log(abs(hg - ag) + 1) + 1 if hg != ag else 1.0
            delta = ELO_K * mult * (res - exp_h)
            H.elo += delta
            A.elo -= delta
        if is_euro:
            last_euro[r.home] = d
            last_euro[r.away] = d
        H.update(d, hg, ag, float(hx[i]), float(ax[i]))
        A.update(d, ag, hg, float(ax[i]), float(hx[i]))
        if is_euro:
            continue
        t = tables[(lg, se)]
        t[r.home] += 3 if hg > ag else 1 if hg == ag else 0
        t[r.away] += 3 if ag > hg else 1 if hg == ag else 0
        played[(lg, se)][r.home] += 1
        played[(lg, se)][r.away] += 1

    out = pd.DataFrame(rows)
    for c in MANUAL_FEATURES:
        out[c] = 0.0
    if manual is not None and len(manual):
        _apply_manual(out, manual)
    # European midweek detected automatically from UCL/UEL/UECL results
    out["euro_midweek"] = np.maximum(out["euro_midweek"], out.pop("_euro_recent"))
    out["opp_euro_midweek"] = np.maximum(out["opp_euro_midweek"], out.pop("_opp_euro_recent"))
    assert all(c in out.columns for c in LEARNED_FEATURES)
    return out


def _apply_manual(rows: pd.DataFrame, manual: pd.DataFrame) -> None:
    """Map per-fixture Gate 3 inputs onto the two scoring rows."""
    for mid, g in manual.iterrows():
        def val(col):
            v = g.get(col, 0)
            return 0.0 if pd.isna(v) else float(v)
        for is_home, me, them in ((1, "home", "away"), (0, "away", "home")):
            idx = (rows["match_id"] == mid) & (rows["is_home"] == is_home)
            rows.loc[idx, "absence_att"] = val(f"{me}_absence_att")
            rows.loc[idx, "opp_absence_def"] = val(f"{them}_absence_def")
            rows.loc[idx, "opp_gk_out"] = val(f"{them}_gk_out")
            rows.loc[idx, "euro_midweek"] = val(f"{me}_euro_midweek")
            rows.loc[idx, "opp_euro_midweek"] = val(f"{them}_euro_midweek")
