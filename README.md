# footy_prediction_model

The v3 Protocol as a statistical model. Every gate is a feature with a **learned
weight** in a time-decayed Poisson / Dixon-Coles goals model, trained on a blend of
real xG and goals. It prices every market (1X2, double chance, O/U, BTTS, team to
score) from one scoreline matrix, then compares each leg against the sharp
(Betfair Exchange) price.

```
target   = 0.7·xG + 0.3·goals   (where real xG exists, goals elsewhere)
then the home/away strength gap is stretched ×1.11 (log λ_h/λ_a; total kept)
log(λ_team) = league + home_adv + attack[team] − defence[opp]       (Gate 1 ratings)
            + β·xG-form + β·results-streak + β·promoted             (Gate 1 form)
            + β·H2H residual                                         (Gate 2)
            + β·rest days + prior·injuries/GK/Europe midweek         (Gate 3)
            + β·table stakes + β·dead rubber + β·new manager         (Gate 4)
            + β·Elo difference                                       (Gate 5)
past matches weighted by exp(−ξ · days_ago), ξ = 0.002/day (half-life ~350 days)
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PY=.venv/bin/python

# 1. data: history (2000-Sep 2026, odds + ClubElo) + current season (refreshed daily)
$PY -m footy_prediction_model fetch --dest data --leagues E0 SP1 I1 D1 F1 G1 CY1

# 2. real xG (FotMob/Opta) -> data/xg/xg.csv; cached, so re-runs only fetch new matches
$PY -m footy_prediction_model xg --data data --leagues E0 SP1 I1 D1 F1 G1 CY1

# 3a. scan a matchday: every fixture, every market -> legs + parlays (no slip needed)
$PY -m footy_prediction_model scan --data data --leagues E0 SP1 I1 D1 F1 G1 CY1 [--date 2026-11-08]
#    -> scan_<date>.html (per-match page), scan_<date>.md, scan_<date>_legs.csv, data/watchlist.csv
$PY -m footy_prediction_model watch --data data      # grade the watch-list experiment

# 3b. or score a slip you already have
$PY -m footy_prediction_model predict --data data --leagues E0 SP1 I1 D1 F1 G1 CY1 \
    --slip examples/slip_2026-11-07.csv --out slip_report.md

# research
$PY -m footy_prediction_model backtest  --data data/history/Matches.csv --xg data/xg/xg.csv --start 2024-08-01 --end 2026-06-30
$PY -m footy_prediction_model edge-test --data data/history/Matches.csv --xg data/xg/xg.csv
$PY -m footy_prediction_model coefs     --data data
$PY -m footy_prediction_model teams     --data data --league E0
$PY -m pytest tests -q
```

`--data <dir>` picks up `<dir>/xg/xg.csv` automatically; `--no-xg` switches it off,
`--xg-weight 0` trains on goals only.

## Matchday page (`scan_<date>.html`)

Light-theme page, grouped by league:

- **Shortlist:** the most likely bet in range (fair odds 1.19–1.60, passes v3.1) for
  each game, with **Min odds** (don't take it on Stoiximan below this).
- **Watch list:** a no-stakes experiment, see below.
- **One card per game:** model xG scoreboard, a one-line read, a model-vs-market strip
  (gaps of 10+ points highlighted), bets in range, Avoid (v3.1 Unders), and one line of
  form per team (position, last 5, xG last 10, Europe midweek, conceding 2+).

Bets come from the sharp market, not from model gaps: see "Model gaps work backwards".

## Scan markdown (`scan_<date>.md`)

1. **Most likely scenario** per fixture: model xG, top 3 scorelines, 1/X/2, O2.5 and
   BTTS (model, with the fair price in brackets).
2. **Best leg per fixture**: the most likely leg with fair odds 1.19–1.43
   (`--min-fair-odds`, `--max-fair-odds`) that passes v3.1. **Min odds** = fair odds
   less a 5% margin: don't take the leg on Stoiximan below it.
3. **Model disagrees, check team news**: legs where the model is 5+ points below the
   fair price. An alert, not a veto (see backtest below).
4. **Parlays**: disjoint (each fixture on one ticket), 6–7 legs, safest first:
   Conservative, Balanced 1..n, Aggressive. Odds range, P(all hit) and weakest link.
5. **Excluded fixtures** with reasons, plus notes (thin data, unmatched names, stale data).

Prices come from football-data.co.uk `fixtures.csv` (refreshed before each round):
Betfair Exchange with the margin removed is the "fair" price, falling back to the
market average where the exchange is missing or stale. O1.5, U3.5, BTTS and team to
score are priced from the expected goals those fair prices imply. Stoiximan can't be
read automatically (bot protection), hence the Min odds column. Leagues not in that
file (CY1, UCL/UEL/UECL) come from open-matchday fixtures: model only, kept out of parlays.

## Slip CSV (`predict`)

Required: `league,date,home,away,selection,odds`

- **league codes:** E0 SP1 I1 D1 F1 G1 CY1 N1 P1 SC0 B1 T1, plus UCL UEL UECL
- **selection:** `1`, `X`, `2`, `1X`, `X2`, `12`, `Over 2.5`, `Under 3.5`, `BTTS Yes`,
  `BTTS No`, `Home to score`, `Away Over 1.5`, or a team name
- **Gate 3, optional (0 if blank):** `home_absence_att`, `away_absence_att` (share
  0–1 of attacking output missing), `home_absence_def`, `away_absence_def`,
  `home_gk_out`, `away_gk_out`, `home_euro_midweek`, `away_euro_midweek`.
  Euro midweek is also detected automatically from UCL/UEL/UECL results.
- **Market-anchored mode, optional:** `odds_1,odds_x,odds_2,odds_o25,odds_u25`.

Team names follow football-data.co.uk spelling (`Man City`, `Nott'm Forest`,
`Ath Madrid`, `Espanol`). Use `teams` to list them; fix mistakes with `--aliases aliases.csv`.

## Data sources

| What | Source | Notes |
|---|---|---|
| History 2000 to Sep 2026 | github.com/xgabora/Club-Football-Match-Data-2000-2025 | results, shots, SoT, average odds, ClubElo |
| Current season | github.com/oladeeayo/open-matchday | daily results + fixtures, 75 comps incl. CY1, G1, UCL/UEL/UECL. Non-commercial, with attribution |
| Real xG | FotMob (Opta xG), `xg` command | top 5 from 2020/21, G1 from 2025/26, UCL/UEL/UECL; **none for CY1**. Unofficial endpoint: personal use, low request rate, don't republish |
| Pre-match prices | football-data.co.uk `fixtures.csv` | Betfair Exchange, Bet365, average, max: 1X2 + O/U 2.5 |
| Manager changes | your CSV `team,date` | `--managers` |

Understat has served a maintenance page since at least Nov 2026, and FBref lost Opta
stats in Jan 2026, hence FotMob. Team identity is learned by fixture overlap (same
league, date ±1 day, same score), so no hand-kept name table is needed.

## What the backtest says (v0.2, 8 Nov 2026)

Rolling weekly refits, top 5 leagues, 2024/25 + 2025/26: 3,491 matches, no leakage.
Settings chosen on 2024/25 only; 2025/26 is the held-out check.

| | Goals-only (v0.1*) | **xG model (v0.2)** | Market |
|---|---|---|---|
| 1X2 log-loss 2024-26 | 0.9893 | **0.9863** | 0.9708 |
| O/U 2.5 log-loss 2024-26 | 0.6807 | **0.6772** | 0.6725 |
| O/U 2.5, held-out 2025/26 | 0.6855 | **0.6814** (z −2.2) | 0.6764 |

\* v0.1 re-run with the team-name fix below.

- **xG as the training target is what helps.** Weight 0.7 is best (0.5–1.0 all
  similar). It closes about half the O/U gap to the market. 1X2 gains are small
  (held out: −0.0012, z −0.6).
- **xG as a form feature adds nothing** over the shots proxy.
- **Still behind the market.** The 50/50 model-market blend (0.9761) is worse than the
  market alone, and value bets ROI is −10%.
- **Calibration:** good up to 70%. Above that the model is under-confident on
  favourites (predicted 75%, actual 80%).

**Can the model spot the risky short leg?** Legs at fair 1.19–1.43, by model edge:

| Model vs fair | Legs | Hit |
|---|---|---|
| ≤ −10 pts | 209 | 79% |
| −10 to −5 | 649 | 75% |
| −5 to +5 | 1,527 | 77–78% |
| > +5 | 56 | 57% |

No. When the model disagrees, the market is right. So scans use the fair price for
every probability, and the model is a "check team news" alert. Real edges: Min odds
(price shopping), line-ups, and parlay sizing.

**Model gaps work backwards.** Across all bets at fair odds 1.19–2.60, held-out
2025/26, after the ×1.11 stretch:

| Model vs market | Bets | Market | Won | Return at Min odds |
|---|---|---|---|---|
| 10+ pts higher | 129 | 49% | 38% | −26% |
| 6–10 higher | 453 | 52% | 51% | −6% |
| within ±6 | 5,678 | 56% | 57% | −4% |
| 10+ pts lower | 265 | 63% | 72% | +9% |

A big gap usually means the model is missing information the market has (injuries,
transfers, real level), so the model's gaps are never used to pick bets. The last row
looks like an edge, but in 2024/25 it only broke even. That's why it's the **watch
list**: each scan logs one such bet per game (pre-match price) to `data/watchlist.csv`,
and `watch` grades the record. No stakes until it holds up live.

**Calibration:** the raw model is under-confident on favourites. Weaker ridge shrinkage
didn't help. A ×1.11 stretch of the strength gap, fitted on 2024/25, improved held-out
1X2 log-loss from 0.9965 to 0.9955 (`strength_stretch` in config).

**Edge test with real xG** (market-anchored, trained 2016–24, tested 2024–26). These
add information beyond the price: attacking xG form (+9% goals per unit, z 2.7),
**opponent defensive xG form (+8%, z 2.4, new with real xG)**, results streak (−4%,
z −2.8, the market over-reacts to runs), dead rubber (+6%, z 2.2), Elo (z −2.0).
Out of sample the adjusted price still doesn't beat the market (1X2 0.9724 vs 0.9719).

**Fixed in v0.2:** the history file spelled two clubs two ways ("Nottm Forest" until
mid-2024, then "Nott'm Forest"; "MGladbach"/"M'gladbach"), splitting each into two
teams. The loader now merges names that only differ by punctuation/case.

## Roadmap

1. Closing-line odds and CLV tracking: log each scan's prices, compare with kick-off prices.
2. Lineup/injury data to learn the Gate 3 weights.
3. Correlated-leg handling for same-day parlays (same league + kick-off).
4. Prices for CY1 and European ties (The Odds API, free key).
5. Grade the watch list after ~200 settled games; promote it only if it beats Min odds.

## Layout

```
footy_prediction_model/
  config.py    gates -> features, priors, flag thresholds, defaults (xi, xg_weight)
  data.py      loaders (football-data, xgabora, open-matchday), name harmonisation
  fetch.py     downloads history + daily current-season data
  xg.py        FotMob xG scraper (cached) + fixture-overlap team mapping
  features.py  leak-free chronological feature builder (all gates)
  model.py     time-decayed Poisson GLM (xG/goals target) + Dixon-Coles rho
  markets.py   scoreline matrix -> markets, de-margining, flags
  anchored.py  market-anchored model / edge test
  backtest.py  rolling-origin backtest, calibration, safe-leg test, tuning
  scan.py      matchday scan: fixtures + fair prices -> legs -> disjoint parlays
  report.py    per-match HTML page (light theme, by league)
  watch.py     watch-list experiment: log, settle, grade
  predict.py   slip scoring + report
  cli.py       commands
tests/         leakage, score matrix, xG target, stretch, xG ingestion, parlay builder, watch list
```
