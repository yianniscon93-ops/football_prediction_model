"""Command line interface.

  python -m footy_prediction_model backtest --data Matches.csv --start 2024-08-01 --end 2026-06-30
  python -m footy_prediction_model tune     --data Matches.csv --start 2023-08-01 --end 2024-06-30
  python -m footy_prediction_model coefs    --data Matches.csv
  python -m footy_prediction_model predict  --data Matches.csv E0_2627.csv --slip examples/slip.csv
  python -m footy_prediction_model teams    --data Matches.csv --league E0
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .backtest import evaluate, print_report, run_backtest, tune_xi
from .config import DEFAULT_LEAGUES, ModelConfig
from . import data as _data
from .data import load_managers, load_matches
from .fetch import fetch_current, fetch_history
from .features import build_features
from .model import FootyModel
from .predict import format_report, score_slip


def _common(p):
    p.add_argument("--data", nargs="+", required=True, help="CSV files (football-data.co.uk or xgabora)")
    p.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    p.add_argument("--since", default="2015-07-01")
    p.add_argument("--xg", default=None, help="optional xG CSV: date,home,away,home_xg,away_xg")
    p.add_argument("--managers", default=None, help="optional CSV: team,date")
    p.add_argument("--aliases", default=None, help="optional CSV: source,target team-name fixes")
    p.add_argument("--xi", type=float, default=ModelConfig.xi)
    p.add_argument("--alpha", type=float, default=ModelConfig.alpha)
    p.add_argument("--train-days", type=int, default=ModelConfig.train_days)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="footy_prediction_model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fetch", help="download history + current-season results")
    fp.add_argument("--dest", default="data")
    fp.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    fp.add_argument("--no-history", action="store_true")
    for name in ("backtest", "tune", "coefs", "predict", "teams", "edge-test"):
        sp = sub.add_parser(name)
        _common(sp)
        if name == "edge-test":
            sp.add_argument("--train-end", default="2024-07-01")
            sp.add_argument("--test-end", default="2026-07-01")
        if name in ("backtest", "tune"):
            sp.add_argument("--start", required=True)
            sp.add_argument("--end", required=True)
            sp.add_argument("--refit-days", type=int, default=7 if name == "backtest" else 28)
        if name == "backtest":
            sp.add_argument("--out", default="backtest_predictions.csv")
        if name == "tune":
            sp.add_argument("--grid", nargs="+", type=float,
                            default=[0.0, 0.001, 0.0019, 0.003, 0.005, 0.008])
        if name == "predict":
            sp.add_argument("--slip", required=True)
            sp.add_argument("--ref-date", default=None)
            sp.add_argument("--out", default="slip_report.md")
        if name == "teams":
            sp.add_argument("--league", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        if not a.no_history:
            fetch_history(a.dest)
        fetch_current(a.leagues, a.dest)
        return
    if getattr(a, "aliases", None):
        al = pd.read_csv(a.aliases)
        _data.EXTRA_ALIASES.update(dict(zip(al["source"].astype(str), al["target"].astype(str))))

    cfg = ModelConfig(xi=a.xi, alpha=a.alpha, train_days=a.train_days)
    leagues = a.leagues
    paths = []
    for d in a.data:
        import glob
        import os
        paths += sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)) if os.path.isdir(d) else [d]
    paths = [p for p in paths if "fixtures_" not in p and "goals_" not in p]
    matches = load_matches(paths, leagues=leagues, since=a.since, xg_path=a.xg,
                           verbose=a.cmd in ("teams", "predict"))
    managers = load_managers(a.managers)

    if a.cmd == "teams":
        m = matches if not a.league else matches[matches.league == a.league]
        last = m[m.season == m.season.max()]
        print("\n".join(sorted(set(last.home) | set(last.away))))
        return

    if a.cmd == "predict":
        slip = pd.read_csv(a.slip)
        extra = sorted(set(slip["league"]) - set(leagues))
        if extra:
            print(f"Note: slip has leagues not loaded ({extra}); those legs will be NODATA "
                  f"unless you add them with --leagues.")
        res, summary = score_slip(matches, slip, cfg, managers=managers, ref_date=a.ref_date)
        rep = format_report(res, summary)
        print(rep)
        with open(a.out, "w") as f:
            f.write(rep + "\n")
        res.to_csv(a.out.replace(".md", ".csv"), index=False)
        return

    print(f"Loaded {len(matches)} matches, building features...", flush=True)
    rows = build_features(matches, managers=managers)

    if a.cmd == "coefs":
        ref = matches["date"].max() + pd.Timedelta(days=1)
        model = FootyModel(cfg).fit(rows, ref)
        print(f"Fitted at {ref.date()}  rho={model.rho_:.3f}")
        print(model.coefficients().round(3).to_string(index=False))
    elif a.cmd == "backtest":
        bt = run_backtest(matches, rows, cfg, a.start, a.end, refit_days=a.refit_days)
        bt.to_csv(a.out, index=False)
        res = evaluate(bt)
        print_report(res)
    elif a.cmd == "edge-test":
        from .anchored import evaluate_anchored
        am, res = evaluate_anchored(rows, matches, a.train_end, a.test_end)
        print(f"Out-of-sample {a.train_end} -> {a.test_end}: {res}")
        print("\nDoes each gate add information beyond the bookmaker price?")
        print(am.table().round(3).to_string(index=False))
    elif a.cmd == "tune":
        t = tune_xi(matches, rows, a.grid, a.start, a.end, base=cfg, refit_days=a.refit_days)
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
