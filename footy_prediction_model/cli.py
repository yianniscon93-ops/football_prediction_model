"""Command line interface.

  python -m footy_prediction_model backtest --data Matches.csv --start 2024-08-01 --end 2026-06-30
  python -m footy_prediction_model tune     --data Matches.csv --start 2023-08-01 --end 2024-06-30
  python -m footy_prediction_model coefs    --data Matches.csv
  python -m footy_prediction_model predict  --data Matches.csv E0_2627.csv --slip examples/slip.csv
  python -m footy_prediction_model teams    --data Matches.csv --league E0
  python -m footy_prediction_model xg       --data data            # real xG -> data/xg/xg.csv
"""
from __future__ import annotations

import argparse
import glob
import os

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
    p.add_argument("--xg", default=None, help="xG CSV: date,[league,]home,away,home_xg,away_xg "
                   "(default: <data dir>/xg/xg.csv when it exists)")
    p.add_argument("--no-xg", action="store_true", help="ignore real xG, use the shot proxy only")
    p.add_argument("--xg-weight", type=float, default=ModelConfig.xg_weight,
                   help="0-1: how much of the training target is xG instead of goals")
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
    for name in ("backtest", "tune", "coefs", "predict", "teams", "edge-test", "xg", "scan", "watch"):
        sp = sub.add_parser(name)
        _common(sp)
        if name == "xg":
            sp.add_argument("--from-season", type=int, default=2020, help="first season start year")
            sp.add_argument("--xg-leagues", nargs="+", default=None,
                            help="leagues to fetch (default: --leagues + UCL UEL UECL)")
            sp.add_argument("--out", default=None, help="default <first data dir>/xg/xg.csv")
            sp.add_argument("--workers", type=int, default=4)
            sp.add_argument("--rps", type=float, default=4.0, help="max requests per second")
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
        if name == "scan":
            sp.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
            sp.add_argument("--all-day", action="store_true",
                            help="include fixtures that already kicked off")
            sp.add_argument("--out", default=None, help="default scan_<date>.md")
            sp.add_argument("--min-fair-odds", type=float, default=1.19,
                            help="shortest leg allowed in parlays (fair odds)")
            sp.add_argument("--max-fair-odds", type=float, default=1.43)
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

    cfg = ModelConfig(xi=a.xi, alpha=a.alpha, train_days=a.train_days, xg_weight=a.xg_weight)
    leagues = a.leagues
    paths = []
    for d in a.data:
        paths += sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)) if os.path.isdir(d) else [d]
    paths = [p for p in paths if "fixtures_" not in p and "goals_" not in p
             and f"{os.sep}xg{os.sep}" not in p]
    if a.cmd == "xg":
        return _cmd_xg(a, paths)
    xg_path = None if a.no_xg else a.xg
    if xg_path is None and not a.no_xg:
        auto = [os.path.join(d, "xg", "xg.csv") for d in a.data if os.path.isdir(d)]
        xg_path = next((p for p in auto if os.path.exists(p)), None)
    matches = load_matches(paths, leagues=leagues, since=a.since, xg_path=xg_path,
                           verbose=a.cmd in ("teams", "predict"))
    if xg_path:
        print(f"Real xG on {int(matches['home_xg'].notna().sum())} of "
              f"{int(matches['hg'].notna().sum())} played matches ({xg_path})")
    managers = load_managers(a.managers)

    if a.cmd == "teams":
        m = matches if not a.league else matches[matches.league == a.league]
        last = m[m.season == m.season.max()]
        print("\n".join(sorted(set(last.home) | set(last.away))))
        return

    if a.cmd == "scan":
        return _cmd_scan(a, matches, cfg)
    if a.cmd == "watch":
        from .watch import grade
        d = next((x for x in a.data if os.path.isdir(x)), "data")
        w, sm = grade(os.path.join(d, "watchlist.csv"), matches)
        if not sm["n"]:
            print(f"No settled watch-list bets yet ({sm.get('pending', 0)} pending).")
            return
        print(w.loc[w["won"].notna(), ["date", "home", "away", "leg", "fair_odds", "model_p", "fair_p", "hg", "ag",
                                       "won"]].to_string(index=False))
        print(f"\nSettled {sm['n']} (pending {sm['pending']}): won {sm['won']:.0%} vs {sm['implied']:.0%} implied | "
              f"return at min odds {sm['roi_min']:+.1%}, at fair odds {sm['roi_fair']:+.1%}")
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


def _cmd_scan(a, matches, cfg):
    from .scan import (fetch_fd_fixtures, format_scan, load_fd_fixtures, load_om_fixtures,
                       _match_names, scan)
    date = pd.Timestamp(a.date) if a.date else pd.Timestamp.today().normalize()
    data_dir = next((d for d in a.data if os.path.isdir(d)), "data")
    notes = []
    try:
        fd = load_fd_fixtures(fetch_fd_fixtures(data_dir), date, a.leagues)
    except Exception as e:  # noqa: BLE001
        notes.append(f"football-data fixtures unavailable ({e}) - no prices")
        fd = pd.DataFrame(columns=["league", "date", "kickoff", "home", "away"])
    missing = [l for l in a.leagues if l not in set(fd["league"])]
    om = load_om_fixtures(data_dir, missing, date, matches.attrs.get("name_map", {}))
    fx = pd.concat([fd, om], ignore_index=True)
    if fx.empty:
        print(f"No fixtures on {date.date()} for {a.leagues}")
        return
    now = pd.Timestamp.now()
    if not a.all_day and date == now.normalize():
        started = fx["kickoff"].notna() & (fx["kickoff"] <= now)
        if started.any():
            notes.append(f"{int(started.sum())} fixtures already kicked off - skipped (--all-day to include)")
        fx = fx[~started]
    fx, name_notes = _match_names(fx, matches)
    notes += name_notes
    if len(om):
        notes.append(f"No prices for {sorted(set(om['league']))} - model only, not in parlays")
    last = matches.loc[matches["hg"].notna(), "date"].max()
    if (date - last).days > 10:
        notes.append(f"Latest result in data is {last.date()} - run `fetch` first")
    views, legs, summary = scan(matches, fx, cfg, ref_date=date, min_fair_odds=a.min_fair_odds,
                                max_fair_odds=a.max_fair_odds)
    rep = format_scan(views, legs, summary, notes)
    out = a.out or f"scan_{date.date()}.md"
    with open(out, "w") as f:
        f.write(rep + "\n")
    legs.to_csv(out.replace(".md", "_legs.csv"), index=False)
    from .report import render_html
    from .watch import grade, log_watch, select_watch
    wpath = os.path.join(data_dir, "watchlist.csv")
    watch = select_watch(legs, views, date)
    if len(watch):
        log_watch(watch, wpath)
    _, wsum = grade(wpath, matches)
    html_out = out.replace(".md", ".html")
    with open(html_out, "w") as f:
        f.write(render_html(views, matches, date, notes, legs=legs, watch=watch, watch_record=wsum))
    print(rep)
    print(f"\nSaved {out}, {html_out} and {out.replace('.md', '_legs.csv')}")


def _cmd_xg(a, paths):
    from .data import EURO_CODES
    from .xg import (NO_XG, FotMobClient, coverage, current_season, map_to_reference,
                     update_raw)
    first_dir = next((d for d in a.data if os.path.isdir(d)), "data")
    out = a.out or os.path.join(first_dir, "xg", "xg.csv")
    raw_path = os.path.join(os.path.dirname(out), "fotmob_raw.csv")
    lgs = a.xg_leagues or [l for l in a.leagues if l not in NO_XG] + ["UCL", "UEL", "UECL"]
    seasons = list(range(a.from_season, current_season() + 1))
    print(f"Fetching FotMob xG for {lgs}, seasons {seasons[0]}-{seasons[-1]}")
    client = FotMobClient(max_rps=a.rps)
    dom = [l for l in lgs if l not in EURO_CODES]
    euro = [l for l in lgs if l in EURO_CODES]
    raw = update_raw(dom, seasons, raw_path, client=client, workers=a.workers)
    if euro:   # our results data only has European ties from the current season
        raw = update_raw(euro, seasons[-1:], raw_path, client=client, workers=a.workers)
    matches = load_matches(paths, leagues=a.leagues, since=a.since)
    xg, names = map_to_reference(raw, matches)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    xg.to_csv(out, index=False)
    unmapped = sorted(set(raw.loc[raw["league"].isin(a.leagues), "home_fm"])
                      - set(names) - set(xg["home"]))
    print(f"\nWrote {len(xg)} matches with real xG -> {out}")
    if unmapped:
        print(f"FotMob teams not matched to our data ({len(unmapped)}): {', '.join(unmapped[:30])}")
    print("\nCoverage (played matches with real xG):")
    print(coverage(xg, matches).to_string())


if __name__ == "__main__":
    main()
