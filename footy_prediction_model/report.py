"""Per-match HTML analysis for a matchday scan: only what matters per game.

Each card: scoreboard (model xG), one-line read, model vs market on the main markets,
"bets that make sense" (where the model and the sharp market differ most), what to
avoid, and one line of form per team. Matches grouped by league. Light theme.
"""
from __future__ import annotations

import html
import math
import re

import numpy as np
import pandas as pd

from .data import EURO_CODES, xg_proxy

LEAGUE_NAMES = {"E0": "Premier League", "SP1": "LaLiga", "I1": "Serie A", "D1": "Bundesliga",
                "F1": "Ligue 1", "G1": "Super League Greece", "CY1": "Cyprus First Division",
                "N1": "Eredivisie", "P1": "Liga Portugal", "B1": "Pro League", "SC0": "Premiership",
                "T1": "Süper Lig", "UCL": "Champions League", "UEL": "Europa League",
                "UECL": "Conference League"}
# Market-led suggestions. The 2024-26 backtest showed model-vs-market gaps work
# backwards (model +10 pts above market -> 45% hit vs 48% implied), so the model is
# shown as a second opinion, never as the reason to bet.
PICK_ODDS = (1.19, 1.60)  # fair-odds window
DIFF_NOTE = 0.10          # flag "model disagrees" at 10+ points
DOUBT_GAP = DIFF_NOTE
AVOID_MAX_ODDS = 1.70
FAMILY = {"1": "res", "X": "res", "2": "res", "1X": "res", "X2": "res", "12": "res",
          "O1.5": "goals", "O2.5": "goals", "O3.5": "goals", "U2.5": "goals", "U3.5": "goals",
          "U4.5": "goals", "BTTS_Y": "btts", "BTTS_N": "btts", "H_SCORE": "btts", "A_SCORE": "btts"}
e = html.escape


# ------------------------------------------------------------------ data helpers
def _played(matches):
    m = matches[matches["hg"].notna()].copy()
    m["hx"] = np.where(m["home_xg"].notna(), m["home_xg"], xg_proxy(m["hg"], m["hs"], m["hst"]))
    m["ax"] = np.where(m["away_xg"].notna(), m["away_xg"], xg_proxy(m["ag"], m["as_"], m["ast"]))
    return m


def _team_games(pl, team, before):
    h = pl[(pl["home"] == team) & (pl["date"] < before)]
    a = pl[(pl["away"] == team) & (pl["date"] < before)]
    g = pd.concat([
        pd.DataFrame({"date": h["date"], "opp": h["away"], "venue": "H", "gf": h["hg"], "ga": h["ag"],
                      "xf": h["hx"], "xa": h["ax"]}),
        pd.DataFrame({"date": a["date"], "opp": a["home"], "venue": "A", "gf": a["ag"], "ga": a["hg"],
                      "xf": a["ax"], "xa": a["hx"]})])
    return g.sort_values("date")


def _positions(pl, league, season, before):
    m = pl[(pl["league"] == league) & (pl["season"] == season) & (pl["date"] < before)]
    if m.empty or league in EURO_CODES:
        return {}
    pts: dict[str, list] = {}
    for r in m.itertuples():
        for t, gf, ga in ((r.home, r.hg, r.ag), (r.away, r.ag, r.hg)):
            p = pts.setdefault(t, [0, 0, 0])
            p[0] += 3 if gf > ga else 1 if gf == ga else 0
            p[1] += gf - ga
            p[2] += gf
    order = sorted(pts, key=lambda t: pts[t], reverse=True)
    return {t: (i + 1, int(pts[t][0])) for i, t in enumerate(order)}


def _pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def _ko(ts):
    return ts.strftime("%H:%M") if pd.notna(ts) else "TBC"


def _ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _bet_name(leg, key, home, away):
    return {"1": f"{home} to win", "X": "Draw", "2": f"{away} to win", "1X": f"{home} or draw",
            "X2": f"{away} or draw", "12": "Either team to win", "BTTS_Y": "Both teams to score",
            "BTTS_N": "Not both to score"}.get(key, leg)


# ------------------------------------------------------------------ bet angles
def angles(L: pd.DataFrame | None, thin: bool = False) -> tuple[list[dict], list[dict]]:
    """(picks, avoid) for one match. Picks: the most likely bets by the sharp market
    with fair odds in PICK_ODDS that pass v3.1, best per market family, top 2.
    Avoid: short prices blocked by the v3.1 rule."""
    if L is None or L.empty or L["fair_p"].isna().all() or thin:
        return [], []
    L = L.dropna(subset=["fair_p"]).copy()
    L["gap"] = L["model_p"] - L["fair_p"]
    L["v31"] = L["reason"].fillna("").str.contains("v3.1")
    pick = L[L["fair_odds"].between(*PICK_ODDS) & ~L["v31"]].copy()
    pick["fam"] = pick["key"].map(FAMILY)
    pick = pick.sort_values("fair_p", ascending=False).groupby("fam").head(1).head(2)
    avoid = L[L["v31"] & (L["fair_odds"] <= AVOID_MAX_ODDS)].copy()
    avoid = avoid.sort_values("fair_p", ascending=False).head(2)
    return pick.to_dict("records"), avoid.to_dict("records")


# ------------------------------------------------------------------ card
def _read(v, PM, PF, home, away):
    src = PF if PF else PM
    names = {"1": home, "X": "A draw", "2": away}
    fav = max(("1", "X", "2"), key=lambda k: src[k])
    bi, bj = np.unravel_index(np.argmax(v["M"]), v["M"].shape)
    if fav == "X" or src[fav] < 0.40:
        s = "Close game, no clear favourite."
    elif src[fav] >= 0.65:
        s = f"{names[fav]} clear favourites ({_pct(src[fav])})."
    else:
        s = f"{names[fav]} favourites ({_pct(src[fav])})."
    tot = PM["xg_h"] + PM["xg_a"]
    s += (" Low-scoring profile" if tot < 2.3 else " High-scoring profile" if tot > 3.0
          else " Average goal profile") + f" ({tot:.1f} xG). Likeliest score {bi}–{bj}."
    return s


def _mvm(PM, PF):
    cols = [("1", "Home"), ("X", "Draw"), ("2", "Away"), ("O2.5", "Over 2.5"), ("BTTS_Y", "Both score")]
    head = "".join(f"<th>{lab}</th>" for _, lab in cols)
    mk = "".join(f"<td>{_pct(PF[k]) if PF else '–'}</td>" for k, _ in cols)
    md = []
    for k, _ in cols:
        cls = ""
        if PF:
            d = PM[k] - PF[k]
            cls = ' class="hi up"' if d >= DIFF_NOTE else ' class="hi down"' if d <= -DIFF_NOTE else ""
        md.append(f"<td{cls}>{_pct(PM[k])}</td>")
    return (f'<table class="mvm"><thead><tr><th></th>{head}</tr></thead><tbody>'
            f'<tr><th>Market</th>{mk}</tr><tr><th>Model</th>{"".join(md)}</tr></tbody></table>')


def _bet_rows(items, home, away, kind):
    rows = []
    for b in items:
        name = _bet_name(b["leg"], b["key"], home, away)
        if kind == "avoid":
            why = "v3.1: a team concedes 2+ a game"
        else:
            why = f"{100 * b['fair_p']:.0f}% likely"
            if b["gap"] <= -DIFF_NOTE or b["gap"] >= DIFF_NOTE:
                why += f" · model says {100 * b['model_p']:.0f}%, check team news"
        odds = (f'<span class="odds"><b>{b["min_odds"]:.2f}</b><small>min odds</small></span>'
                if kind == "like" else f'<span class="odds dim">{b["fair_odds"]:.2f}<small>fair</small></span>')
        rows.append(f'<li><span class="bname">{e(name)}</span><span class="why">{e(why)}</span>{odds}</li>')
    return "".join(rows)


def _team_line(name, g, pos, flags):
    chips = []
    for r in g.tail(5).itertuples():
        res = "W" if r.gf > r.ga else "D" if r.gf == r.ga else "L"
        tip = f"{r.date:%d %b} {'v' if r.venue == 'H' else '@'} {r.opp} {int(r.gf)}-{int(r.ga)}"
        chips.append(f'<i class="{res.lower()}" title="{e(tip)}">{res}</i>')
    t10 = g.tail(10)
    xg = f"xG {t10.xf.mean():.1f}–{t10.xa.mean():.1f} last 10" if len(t10) else ""
    p = f"{_ordinal(pos[0])}, {pos[1]} pts" if pos else ""
    fl = "".join(f'<span class="flag">{e(f)}</span>' for f in flags)
    return (f'<div class="tl"><span class="tln">{e(name)}</span><span class="tlp">{p}</span>'
            f'<span class="form">{"".join(chips)}</span><span class="tlx">{xg}</span>{fl}</div>')


def _card(v, pl, date, L=None):
    home, away, lg = v["home"], v["away"], v["league"]
    mid = int(v["match_id"])
    head = f'<div class="meta"><span class="ko">{_ko(v["kickoff"])}</span></div>'
    if not isinstance(v.get("M"), np.ndarray):
        return (f'<article class="match" id="m{mid}"><header class="board">{head}<div class="score">'
                f'<span class="team home">{e(home)}</span><span class="led">–</span>'
                f'<span class="team away">{e(away)}</span></div></header>'
                f'<div class="body"><p class="read">No data for one of these teams.</p></div></article>')
    PM, PF = v["model_P"], v["fair_P"]
    fh, fa = v["feats"]["h"], v["feats"]["a"]
    like, avoid = angles(L, bool(v.get("thin")))
    gh, ga = _team_games(pl, home, date), _team_games(pl, away, date)
    season = date.year if date.month >= 7 else date.year - 1
    pos = _positions(pl, lg, season, date)

    def flags(f, g):
        out = []
        if f.get("euro_midweek", 0) >= 1:
            out.append("Europe midweek")
        t10 = g.tail(10)
        if len(t10) and t10.ga.mean() >= 2.0:
            out.append(f"concedes {t10.ga.mean():.1f} a game")
        return out

    if not PF:
        bets = ('<div class="bets none"><h3>Bets in range</h3><p>No bookmaker prices for this game, '
                'so there is nothing to compare the model with.</p></div>')
    else:
        like_html = (f'<ul class="blist like">{_bet_rows(like, home, away, "like")}</ul>' if like else
                     f'<p class="quiet">Nothing priced {PICK_ODDS[0]:.2f}–{PICK_ODDS[1]:.2f} that passes the checks.</p>')
        avoid_html = (f'<h4>Avoid</h4><ul class="blist avoid">{_bet_rows(avoid, home, away, "avoid")}</ul>'
                      if avoid else "")
        bets = f'<div class="bets"><h3>Bets in range</h3>{like_html}{avoid_html}</div>'
    return f'''
<article class="match" id="m{mid}">
  <header class="board">
    {head}
    <div class="score">
      <span class="team home">{e(home)}</span>
      <span class="led" title="Model expected goals">{PM["xg_h"]:.1f}<i>xG</i>{PM["xg_a"]:.1f}</span>
      <span class="team away">{e(away)}</span>
    </div>
  </header>
  <div class="body">
    <div class="left">
      <p class="read">{e(_read(v, PM, PF, home, away))}</p>
      {_mvm(PM, PF)}
      <div class="teams">
        {_team_line(home, gh, pos.get(home), flags(fh, gh))}
        {_team_line(away, ga, pos.get(away), flags(fa, ga))}
      </div>
    </div>
    {bets}
  </div>
</article>'''


# ------------------------------------------------------------------ page
def _league_order(lg):
    keys = list(LEAGUE_NAMES)
    return keys.index(lg) if lg in keys else 99


def _top_gaps(views, by_mid):
    rows = []
    for _, v in views.iterrows():
        like, _ = angles(by_mid.get(v["match_id"]), bool(v.get("thin")))
        if like:
            b = like[0]
            rows.append((b["fair_p"], v, b))
    rows.sort(key=lambda x: -x[0])
    if not rows:
        return ('<section class="top"><h2>Shortlist</h2>'
                '<p>No bets in range today.</p></section>')
    items = "".join(
        f'<tr><td class="tk">{_ko(v["kickoff"])}</td><td><a href="#m{int(v["match_id"])}">{e(v["home"])} v {e(v["away"])}</a>'
        f'<small>{e(LEAGUE_NAMES.get(v["league"], v["league"]))}</small></td>'
        f'<td class="tb">{e(_bet_name(b["leg"], b["key"], v["home"], v["away"]))}</td>'
        f'<td class="n">{_pct(b["fair_p"])}</td><td class="n strong">{_pct(b["model_p"])}</td>'
        f'<td class="n">{b["fair_odds"]:.2f}</td><td class="n"><b>{b["min_odds"]:.2f}</b></td></tr>'
        for g, v, b in rows[:12])
    return (f'<section class="top"><h2>Shortlist</h2><p>The most likely bet in range for each game, '
            f'most likely first.</p><div class="tbl"><table><thead><tr>'
            f'<th>KO</th><th>Match</th><th>Bet</th><th class="n">Chance</th><th class="n">Model</th>'
            f'<th class="n">Fair odds</th><th class="n">Min odds</th></tr></thead><tbody>{items}</tbody></table></div></section>')


def _watch(watch, rec):
    if watch is None:
        return ""
    if rec and rec.get("n"):
        record = (f"Record so far: {rec['n']} settled, won {_pct(rec['won'])} against {_pct(rec['implied'])} "
                  f"implied; {100 * rec['roi_min']:+.0f}% return at min odds.")
    else:
        record = f"Record so far: nothing settled yet ({rec.get('pending', 0) if rec else 0} pending)."
    if watch.empty:
        rows = '<tr><td colspan="6">No bets qualify today.</td></tr>'
    else:
        w = watch.sort_values(["kickoff", "model_p"])
        w = w.assign(gap=w["model_p"] - w["fair_p"]).sort_values("gap").groupby("match_id").head(1)
        w = w.sort_values(["kickoff", "match_id"])
        rows = "".join(
            f'<tr><td class="tk">{_ko(r.kickoff)}</td><td><b>{e(r.home)} v {e(r.away)}</b></td>'
            f'<td class="tb">{e(_bet_name(r.leg, r.key, r.home, r.away))}</td><td class="n">{_pct(r.fair_p)}</td>'
            f'<td class="n">{_pct(r.model_p)}</td><td class="n">{r.fair_odds:.2f}</td></tr>' for r in w.itertuples())
    return (f'<section class="top watch"><h2>Watch list</h2><p>No stakes. These are bets the market rates '
            f'highly and the model doubts by 10+ points. In 2025/26 such bets won 72% of the time against '
            f'63% implied, but in 2024/25 they only broke even, so they are tracked before any money goes on '
            f'them. {e(record)}</p><div class="tbl"><table><thead><tr><th>KO</th><th>Match</th><th>Bet</th>'
            f'<th class="n">Market</th><th class="n">Model</th><th class="n">Fair odds</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></section>')


CSS = r"""
/* Layout: crisp matchday programme. Summary of the day's biggest model-vs-market gaps,
   then league sections; each match = scoreboard band + two columns (read | bets). Light only. */
:root {
  --paper: #f3f5f2; --card: #ffffff; --ink: #0d1a12; --muted: #56645b; --line: #dde3de;
  --pitch: #136b3a; --pitch-2: #0f5c31; --chalk: rgba(255,255,255,.25); --on-pitch: #ffffff;
  --board: #08110c; --led: #ffc548; --note-bg: #fff8e6; --note-fg: #5c4300;
  --up: #12733d; --up-bg: #e3f3e8; --down: #b42318; --down-bg: #fdeceb; --draw: #8e9a93;
  --f-display: "Barlow Condensed", "Arial Narrow", sans-serif;
  --f-body: "Barlow", "Helvetica Neue", Arial, sans-serif;
  color-scheme: light;
}
* { box-sizing: border-box }
body { background: var(--paper); color: var(--ink); font-family: var(--f-body); font-size: 17px; line-height: 1.5 }
.wrap { max-width: 1040px; margin: 0 auto; padding-block: 36px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 40px }
h1, h2, h3, h4 { font-family: var(--f-display); margin: 0; text-wrap: balance }
.masthead { display: grid; gap: 12px }
.kicker { font-family: var(--f-display); font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: var(--pitch); font-size: 16px }
.masthead h1 { font-size: clamp(44px, 8vw, 72px); line-height: .9; text-transform: uppercase }
.masthead p { margin: 0; color: var(--muted); max-width: 68ch }
.masthead p.caveat { border-left: 4px solid var(--led); background: var(--note-bg); padding: 12px 16px; font-size: 15px; color: var(--note-fg); max-width: 80ch }
.top h2, .lg-title { font-size: clamp(28px, 4.5vw, 38px); text-transform: uppercase; letter-spacing: .02em }
.top { display: grid; gap: 14px }
.top > p { margin: 0; color: var(--muted); max-width: 80ch }
.watch .tbl { border-style: dashed }
.tbl { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 10px }
.top table { width: 100%; border-collapse: collapse; min-width: 640px; font-size: 16px }
.top th { font-size: 12px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); text-align: left; padding: 12px 14px; border-bottom: 1px solid var(--line) }
.top td { padding: 12px 14px; border-bottom: 1px solid var(--line); vertical-align: middle }
.top tr:last-child td { border-bottom: 0 }
.top a { color: var(--ink); font-weight: 700; text-decoration: none }
.top a:hover, .top a:focus-visible { color: var(--pitch); text-decoration: underline }
.top small { display: block; color: var(--muted); font-size: 13px }
.tk { font-family: var(--f-display); font-weight: 700; font-size: 20px }
.tb { font-weight: 600 }
.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap }
.top th.n { text-align: right }
.strong { font-weight: 700 } .gap { color: var(--up); font-weight: 700 }
.league { display: grid; gap: 22px }
.lg-title { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding-bottom: 8px; border-bottom: 3px solid var(--ink) }
.lg-title span { font-family: var(--f-body); font-size: 15px; font-weight: 600; letter-spacing: 0; text-transform: none; color: var(--muted) }
.match { background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; scroll-margin-top: 16px }
.match.has-like { border-color: var(--pitch); box-shadow: 0 0 0 1px var(--pitch) }
.board { color: var(--on-pitch); padding: 16px 22px 20px; display: grid; gap: 10px;
  background:
    linear-gradient(90deg, transparent calc(50% - 1px), var(--chalk) calc(50% - 1px), var(--chalk) calc(50% + 1px), transparent calc(50% + 1px)),
    repeating-linear-gradient(90deg, var(--pitch) 0 80px, var(--pitch-2) 80px 160px) }
.meta { display: flex; justify-content: center }
.ko { font-family: var(--f-display); font-weight: 700; font-size: 18px; letter-spacing: .12em; background: rgba(0,0,0,.25); padding: 2px 12px; border-radius: 4px }
.score { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 18px }
.team { font-family: var(--f-display); font-weight: 700; font-size: clamp(24px, 4.4vw, 38px); line-height: 1; text-transform: uppercase; min-width: 0; overflow-wrap: anywhere }
.team.home { text-align: right }
.led { display: inline-flex; align-items: center; gap: 10px; background: var(--board); color: var(--led); padding: 8px 16px; border-radius: 6px;
  font-family: var(--f-display); font-weight: 700; font-size: clamp(30px, 5.5vw, 44px); line-height: 1; font-variant-numeric: tabular-nums }
.led i { font-style: normal; font-size: 13px; letter-spacing: .16em; opacity: .7 }
.body { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 28px; padding: 22px 24px 24px }
.left { display: grid; gap: 18px; align-content: start; min-width: 0 }
.read { margin: 0; font-size: 19px; line-height: 1.45; font-weight: 500 }
.mvm { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 17px }
.mvm th { font-size: 12px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 700; text-align: center; padding: 4px 6px }
.mvm tbody th { text-align: left; padding-left: 0 }
.mvm td { text-align: center; padding: 8px 6px; border-top: 1px solid var(--line); font-weight: 600 }
.mvm td.hi { font-weight: 800 } .mvm td.up { color: var(--up); background: var(--up-bg) } .mvm td.down { color: var(--down); background: var(--down-bg) }
.teams { display: grid; gap: 8px }
.tl { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; font-size: 15px }
.tln { font-weight: 700; min-width: 0 }
.tlp, .tlx { color: var(--muted); font-variant-numeric: tabular-nums }
.form { display: inline-flex; gap: 3px }
.form i { font-style: normal; width: 22px; height: 22px; display: grid; place-items: center; border-radius: 4px; color: var(--on-pitch);
  font-family: var(--f-display); font-weight: 700; font-size: 14px }
.form i.w { background: var(--up) } .form i.d { background: var(--draw) } .form i.l { background: var(--down) }
.flag { font-size: 13px; font-weight: 700; color: var(--down); background: var(--down-bg); padding: 1px 8px; border-radius: 10px }
.bets { display: grid; gap: 12px; align-content: start; min-width: 0 }
.bets h3 { font-size: 22px; text-transform: uppercase; letter-spacing: .04em }
.bets h4 { font-size: 16px; text-transform: uppercase; letter-spacing: .1em; color: var(--down); margin-top: 6px }
.blist { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px }
.blist li { display: grid; grid-template-columns: 1fr auto; grid-template-areas: "n o" "w o"; gap: 2px 14px; align-items: center;
  padding: 12px 14px; border-radius: 8px }
.blist.like li { background: var(--up-bg); border: 1px solid color-mix(in oklab, var(--up) 30%, transparent) }
.blist.avoid li { background: var(--down-bg) }
.bname { grid-area: n; font-weight: 700; font-size: 18px; line-height: 1.25 }
.why { grid-area: w; font-size: 14px; color: var(--muted); font-variant-numeric: tabular-nums }
.odds { grid-area: o; display: grid; justify-items: end; font-family: var(--f-display); font-size: 28px; line-height: 1; font-variant-numeric: tabular-nums }
.odds small { font-family: var(--f-body); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-top: 3px }
.odds.dim { font-size: 22px; color: var(--muted) }
.quiet, .bets.none p { margin: 0; color: var(--muted); font-size: 16px }
.foot { color: var(--muted); font-size: 14px; display: grid; gap: 6px; max-width: 80ch }
.foot p { margin: 0 }
a:focus-visible { outline: 2px solid var(--pitch); outline-offset: 2px }
@media (max-width: 760px) {
  .body { grid-template-columns: minmax(0, 1fr); gap: 22px; padding: 18px }
  .score { gap: 10px } .board { padding: 14px 16px 16px }
  .mvm { font-size: 15px } .mvm th { font-size: 10px; letter-spacing: .04em }
}
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth } }
"""


def render_html(views: pd.DataFrame, matches: pd.DataFrame, date: pd.Timestamp,
                notes: list[str] | None = None, standalone: bool = True,
                legs: pd.DataFrame | None = None, watch: pd.DataFrame | None = None,
                watch_record: dict | None = None) -> str:
    pl = _played(matches)
    views = views.copy()
    views["_ko"] = views["kickoff"].fillna(pd.Timestamp.max)
    views["_lo"] = views["league"].map(_league_order)
    views = views.sort_values(["_lo", "_ko", "home"])
    by_mid = {m: g for m, g in legs.groupby("match_id")} if legs is not None and len(legs) else {}
    secs = []
    for lg, gv in views.groupby("league", sort=False):
        cards = "\n".join(_card(v, pl, date, by_mid.get(v["match_id"])) for _, v in gv.iterrows())
        secs.append(f'<section class="league" id="lg-{e(lg)}"><h2 class="lg-title">{e(LEAGUE_NAMES.get(lg, lg))}'
                    f'<span>{len(gv)} {"match" if len(gv) == 1 else "matches"}</span></h2>{cards}</section>')
    note_html = "".join(f"<p>{e(n)}</p>" for n in (notes or []))
    body = f"""<title>Matchday {date:%-d %b %Y}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600;700;800&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header class="masthead">
    <span class="kicker">Matchday analysis</span>
    <h1>{date:%A %-d %B}</h1>
    <p>{len(views)} games. The scoreboard shows the model's expected goals. "Market" is the Betfair Exchange
    price with the bookmaker margin removed, which is the sharpest estimate of each chance. Bets listed are the
    most likely ones priced {PICK_ODDS[0]:.2f}–{PICK_ODDS[1]:.2f} that pass the v3.1 rule. Kick-offs in Cyprus time.</p>
    <p class="caveat">Why not the model's favourite angles: in the 2024–26 backtest, bets the model rated 10–15 points
    above the market won 45% of the time against 48% implied. The model is under-confident on favourites, so its
    gaps mostly point at underdogs. It stays on the page as a second opinion.</p>
  </header>
  {_top_gaps(views, by_mid)}
  {_watch(watch, watch_record)}
  {"".join(secs)}
  <footer class="foot">
    <p>Model: Poisson / Dixon-Coles trained on 70% Opta xG and 30% goals. Min odds = fair odds less a 5% margin:
    don't take a bet on Stoiximan below it. Avoid = Unders blocked by the v3.1 rule (a team conceding 2+ a game
    over its last 10). Model figures that differ from the market by 10+ points are highlighted.</p>
    {note_html}
  </footer>
</div>"""
    # internal names like "Aris (CY1)" keep same-named clubs apart; hide the tag on the page
    body = re.sub(r" \((?:" + "|".join(LEAGUE_NAMES) + r")\)", "", body)
    if not standalone:
        return body
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            f'</head><body>\n{body}\n</body></html>')
