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
  (a) 14 legs at ~80% each ≈ 1–3% to land — cap parlays at 4–6 legs or split;
  (b) model scored City X2 78% vs fair 77% — market had it right, it was the 1-in-5;
  (c) edge-test shows bookies OVER-react to results streaks and UNDER-react to
  underlying shots/xG form — weight underlying performance, not W/L runs.

## Model state (v0.1, 7 Nov 2026)
- Time-decayed Poisson GLM + Dixon-Coles; each gate = feature with learned β;
  Gate 3 injuries/GK/Europe-midweek = fixed priors (no historical data).
  Tuned: xi=0.001/day, alpha=3e-3.
- Backtest 2024-26 (3,487 top-5 matches): well calibrated, but 1X2 log-loss 0.990 vs
  market 0.971; value bets ROI −10%; flags do NOT separate winning from losing
  1.15–1.40 legs (all ~77–79%). Flags = "model disagrees, check news", not "drop".
- Edge test (market-anchored): beyond price only xG/shots form (+, z 2.7), results
  streak (−, z −2.6), Elo (z −2.1) matter; H2H/motivation/rest are priced in.
  Out-of-sample the adjustments didn't beat the market.
- Data: history = github.com/xgabora/Club-Football-Match-Data-2000-2025 (to Sep 2026,
  odds + ClubElo); current season = github.com/oladeeayo/open-matchday (daily,
  75 comps incl. CY1, G1, UCL/UEL/UECL; non-commercial w/ attribution).
  `python -m footy_prediction_model fetch --dest data`.
- FBref lost Opta advanced stats (Jan 2026) — don't rely on it for xG.

## Next tasks (priority order)
1. **Real xG:** Understat scraper (top 5 leagues, 2014→now) → CSV
   `date,home,away,home_xg,away_xg` (team names mapped to football-data spelling),
   plus a daily GitHub Action committing it to this public repo so claude.ai chat can
   read it. Feed via `--xg`. Re-run `backtest` and `edge-test` to see if xG form
   beats the shot proxy.
2. Pinnacle closing odds as benchmark; CLV tracking.
3. Lineup/injury data to learn Gate 3 weights.
4. Correlated-leg handling for same-day parlays; leg-count cap in report.
5. Greek/Cypriot leagues: thin data (CY1 only 2026/27) — keep low-confidence warnings.
