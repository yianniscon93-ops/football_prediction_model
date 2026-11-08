# footy_prediction_model

The v3 Protocol as a statistical model. Every gate is a feature with a **learned
weight** in a time-decayed Poisson / Dixon-Coles goals model. It prices every market
(1X2, double chance, O/U, BTTS, team to score) from one scoreline matrix, then
compares each Stoiximan leg against the de-margined bookmaker price.

```
log(xG_team) = league + home_adv + attack[team] − defence[opp]       (Gate 1 ratings)
             + β·xG-form + β·results-streak + β·promoted             (Gate 1 form)
             + β·H2H residual                                         (Gate 2)
             + β·rest days + prior·injuries/GK/Europe midweek         (Gate 3)
             + β·table stakes + β·dead rubber + β·new manager         (Gate 4)
             + β·Elo difference                                       (Gate 5)
past matches weighted by exp(−ξ · days_ago)   ← the "form weight", tuned by backtest
```

## Quick start

```bash
pip install -r requirements.txt

# 1. data: history (2000-Sep 2026, odds + ClubElo) + current season (refreshed daily)
python -m footy_prediction_model fetch --dest data --leagues E0 SP1 I1 D1 F1 G1 CY1

# 2. score a slip
python -m footy_prediction_model predict --data data --leagues E0 SP1 I1 D1 F1 G1 CY1 \
    --slip examples/slip_2026-11-07.csv --out slip_report.md

# research commands
python -m footy_prediction_model backtest  --data data/history/Matches.csv --start 2024-08-01 --end 2026-06-30
python -m footy_prediction_model edge-test --data data/history/Matches.csv     # gates vs market
python -m footy_prediction_model tune      --data data/history/Matches.csv --start 2024-08-01 --end 2025-06-30
python -m footy_prediction_model coefs     --data data                          # learned gate weights
python -m footy_prediction_model teams     --data data --league E0              # exact team names
```

## Slip CSV

Required: `league,date,home,away,selection,odds`

- **league codes:** E0 SP1 I1 D1 F1 G1 CY1 N1 P1 SC0 B1 T1, plus UCL UEL UECL
- **selection:** `1`, `X`, `2`, `1X`, `X2`, `12`, `Over 2.5`, `Under 3.5`, `BTTS Yes`,
  `BTTS No`, `Home to score`, `Away Over 1.5`, or a team name
- **Gate 3, optional (0 if blank):** `home_absence_att`, `away_absence_att` (share
  0–1 of attacking output missing), `home_absence_def`, `away_absence_def`,
  `home_gk_out`, `away_gk_out`, `home_euro_midweek`, `away_euro_midweek`.
  Euro midweek is also detected automatically from UCL/UEL/UECL results.
- **Market-anchored mode, optional:** `odds_1,odds_x,odds_2,odds_o25,odds_u25` (that
  fixture's full prices). This adds an "Anchored" column: the market's own probability
  adjusted only by the gates that proved to add information beyond the price.

Team names follow football-data.co.uk spelling (`Man City`, `Nott'm Forest`,
`Ath Madrid`, `Espanol`, `La Coruna`). Use `teams` to list them. The loader maps
other spellings automatically, and you can fix any mistakes with `--aliases aliases.csv`
(`source,target`).

## Data sources

| What | Source | Notes |
|---|---|---|
| History 2000 to Sep 2026 | github.com/xgabora/Club-Football-Match-Data-2000-2025 | results, shots, SoT, average odds, ClubElo |
| Current season | github.com/oladeeayo/open-matchday | refreshed daily: results, shots, SoT for 75 competitions incl. Cyprus, Greece, UCL/UEL/UECL. Non-commercial use with attribution |
| Any extra season | football-data.co.uk CSVs | drop them into `--data`; auto-detected |
| Real xG (optional) | Understat/FBref export | `--xg file.csv` with `date,home,away,home_xg,away_xg`. Otherwise a shots-based proxy is used |
| Manager changes | your CSV `team,date` | `--managers`. Feeds the new-manager feature |

## What the backtest says (v0.1, honest)

Rolling weekly refits, top 5 leagues, 2024/25 and 2025/26 seasons: 3,487 matches, no leakage.

| | Model | Market (de-margined) |
|---|---|---|
| 1X2 log-loss | 0.990 | **0.971** |
| O/U 2.5 log-loss | 0.681 | **0.672** |
| Calibration | good (within ±2% in every bin up to 80%) | — |
| Value bets EV > 5% | ROI −10% | — |

**Safe legs** (odds about 1.15–1.40, the parlay use case): 2,756 legs, 77.6% hit.
By model flag: 🟢 76.6%, 🟡 78.8%, 🟠 76.7%, 🔴 79.2%. **The flags do not predict
which safe legs fail.** A disagreement with the market means "check team news",
not "drop the leg".

**Edge test: do the gates add anything beyond the price?** Market-anchored
regression, trained 2016–24 and tested 2024–26:

| Gate feature | Effect beyond price | Verdict |
|---|---|---|
| xG/shots attack form | +10% goals per unit, z=2.7 | market **under-reacts** to underlying form |
| Results streak | −3.8% goals per unit, z=−2.6 | market **over-reacts** to win/loss runs |
| Elo difference | small, z=−2.1 | marginal |
| H2H | z=−0.2 | fully priced in, which confirms v3.1 |
| Motivation, rest, promoted | \|z\| < 1.7 | priced in |

Out of sample, applying these adjustments did not beat the market's log-loss
(0.97189 vs 0.97190). The effects are real but small, and testing 18 features means
one of them could be noise.

**Takeaway for the City leg:** a losing streak is already over-priced by
bookmakers. If anything, the edge is in trusting underlying shot/xG performance over
results. City outshot Brighton 21–7 and still lost.

## Roadmap

1. Real xG (Understat) in place of the shot proxy, since xG form was the gate with signal.
2. Lineup/injury data to learn the Gate 3 weights rather than using priors.
3. Closing-line (Pinnacle) odds for a sharper benchmark.
4. Correlated-leg handling for same-day parlays.

## Layout

```
footy_prediction_model/
  config.py    gates → features, priors, flag thresholds
  data.py      loaders (football-data, xgabora, open-matchday) + team-name harmonisation
  fetch.py     downloads history + daily current-season data
  features.py  leak-free chronological feature builder (all gates)
  model.py     time-decayed Poisson GLM + Dixon-Coles rho
  markets.py   scoreline matrix → markets, de-margining, flags
  anchored.py  market-anchored model / edge test
  backtest.py  rolling-origin backtest, calibration, safe-leg test, tuning
  predict.py   slip scoring + report
  cli.py       commands
```
