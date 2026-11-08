# CLAUDE.md — footy_prediction_model

Context carried over from the claude.ai "Football betting" project (Nov 2026).
Read this first; README.md has the model design and backtest numbers.

## Who / what
- Yiannis: senior data engineer in Cyprus (Databricks, PySpark, BigQuery). Casual,
  direct, often Greeglish. Keep answers short and concrete.
- Purpose: structured pre-match research for multi-leg parlays on **Stoiximan**
  (decimal odds), mainly European football. Goal: consistent value through rigorous
  research, not longshot stacking. Every leg must be analytically justified.
- This repo turns the manual v3 Protocol into a statistical model
  (`footy_prediction_model`). Slips are often shared as Stoiximan screenshots (Greek
  team names) → convert to `examples/slip_*.csv` and run `predict`.

## v3 Protocol (finalised 13 Mar 2026, v3.1 added 21 Mar 2026)
Five gates, applied to every fixture:
1. Stats / xG and underlying metrics
2. Head-to-head (subject to v3.1 override)
3. Team news, injuries, rotation risk
4. Motivation and seasonal context
5. League quality differential (tiered ranking; added after the Samsunspor loss)

**v3.1:** current form overrides historical H2H when a team's defensive profile has
changed. A team conceding 2+ goals/game over a 10-match sample compromises any Under
bet involving it, whatever the H2H. No blanket rules by league or fixture type — stats
decide each fixture.

**Structural rules**
- Any leg above 1.50 is auto-flagged 🟡 minimum and stress-tested harder.
- The highest-odds "safe" leg is structurally the weakest link.
- No double exposure: one fixture appears once per parlay, across all markets.
- Handicap markets are banned in all contexts.
- Standing exclusions: fixtures with no data edge; fixtures where motivation adds
  structural uncertainty without compensating value.
- Stoiximan applies correlation discounts on same-game bet builders — verify live odds.

**European knockout first legs (absolute):** no handicaps; BTTS downgraded/avoid;
avoid result markets (draws more common than league averages; lower-tier home vs
top-5-league away = trap); approved markets: Under goals, double chance, draw.
Crisis-narrative trap: domestic struggle ≠ European collapse — never back against a
top-league home team on result markets; use goals markets. March 12 sample (16
fixtures): stronger-league away teams won or drew every time, BTTS ~40%, Under 2.5
dominant, most games decided by 1 goal or drawn.

## Ways of working
- Research every fixture on the slip, not a curated shortlist. Yiannis makes final
  adjustments; Claude provides structured research.
- Output: tables with probability breakdowns (1X2, O/U, BTTS), flags
  🟢 safe / 🟡 odds >1.50 or elevated uncertainty / 🟠 risky / 🔴 skip,
  parlay tiers Conservative / Balanced / Aggressive, explicit weakest link, and an
  exclusion list with reasons.

## Loss log (core maxim: Conservative = profitable. Clever = death.)
- **12 Mar 2026:** 4/8 legs failed — Samsunspor 1X (league gap ignored → Gate 5),
  Palace −2 handicap (elite defence, 0-0), Rakow X2 (crisis-narrative trap), BTTS
  Forest (0-1). All 4 conservative legs hit.
- **21 Mar 2026:** Espanyol v Getafe Under 2.5 lost (2-1). Overweighted old H2H
  (80% unders) while Espanyol were conceding 2.3/game → v3.1. All 6 conservative legs hit.
- **7 Nov 2026:** €10 14-fold @35.3, 13/14 hit. Lost on Nott'm Forest v Man City X2
  @1.22 (Forest 3-1; City's 3rd straight league loss, Maresca era; 115-charges guilty
  verdict leaked 25 Sep but City kept winning after it, so not the cause). Lessons:
  (a) 14 legs at ~80% each ≈ 1–3% to land — split into several 6–7 leg parlays;
  (b) model scored City X2 78% vs fair 77% — market had it right, it was the 1-in-5;
  (c) edge-test shows bookies OVER-react to results streaks and UNDER-react to
  underlying shots/xG form — weight underlying performance, not W/L runs.

## Parlay sizing (8 Nov 2026, supersedes lesson (a) cap of 4–6)
When many legs are playable, build several **disjoint** 6–7 leg parlays (each fixture on
one ticket only), safest first: Conservative, Balanced 1..n, Aggressive. Default leg
floor: fair odds ≥ 1.19 (Stoiximan shows ~5% less).

## Model state (v0.2, 8 Nov 2026)
- Time-decayed Poisson GLM + Dixon-Coles; target = 0.7·xG + 0.3·goals where real xG
  exists. xi=0.002/day, alpha=3e-3. Gate 3 injuries/GK/Europe-midweek = fixed priors.
- Backtest 2024-26 (3,491 top-5 matches), xG chosen on 24/25, held out 25/26:
  1X2 log-loss 0.986 vs market 0.971; O/U 0.677 vs 0.673 (xG halved the O/U gap,
  z −2.2 held out). Still behind the market; value bets ROI −10%.
- Model disagreement does NOT predict which short legs lose: legs at fair 1.19–1.43
  that the model rated 10+ pts riskier hit 79%. **Scans use the sharp fair price
  (Betfair Exchange, de-margined) as the probability**; the model is a "check team
  news" alert. Edges = Min odds (price shopping), line-ups, parlay sizing.
- Edge test with real xG: attacking AND defensive xG form (+, z 2.7 / 2.4), results
  streak (−, z −2.8), dead rubber, Elo add info beyond price, but out of sample the
  adjustments still don't beat the market.
- Model is under-confident on favourites (predicted 75% → actual 80%).
- Data: history = xgabora (to Sep 2026, odds + ClubElo); current = open-matchday
  (daily, incl. CY1, G1, UCL/UEL/UECL); xG = FotMob/Opta via `xg` command (top 5 from
  2020/21, G1 from 2025/26, Europe current season, none for CY1) → data/xg/ (gitignored,
  don't republish: Opta data); prices = football-data.co.uk fixtures.csv.
- Understat is down (maintenance page since ≥ Nov 2026); FBref lost Opta stats (Jan 2026).
- Stoiximan can't be read automatically (Cloudflare); don't try to bypass it.
- Env: use `.venv/bin/python` (python.org Python here lacks sklearn and CA certs).

## Matchday workflow
"Scan today, leagues X" → `fetch --no-history` (fresh results), `xg` (new matches
only), then `scan --date ... --leagues ...` → scan_<date>.md. Yiannis checks Stoiximan
prices against **Min odds** for the chosen legs only. A slip from a screenshot still
works via `predict`.

## Next tasks (priority order)
1. CLV tracking: store each scan's fair prices, compare with closing prices.
2. Lineup/injury data to learn Gate 3 weights.
3. Correlated-leg handling for same-day parlays.
4. Prices for CY1 + European ties (The Odds API free key) — currently model-only.
5. Calibration fix for favourite under-confidence.
6. Daily xG GitHub Action: on hold. FotMob data is Opta's, so don't commit it to the
   public repo; if claude.ai chat needs it, use a private repo or a summary.
