"""Rolling-origin backtest: refit every N days using only data before that date,
predict the next N days, compare with the bookmaker market.

Key outputs
  - log-loss / Brier for 1X2 and O/U 2.5: model vs de-margined market
  - calibration table
  - short-price leg test: among "safe" legs (fair prob 70-87%, i.e. odds ~1.15-1.40),
    does the model's flag separate winners from losers?  This is the parlay use case.
  - flat-stake ROI on value bets (model EV > threshold)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FlagConfig, ModelConfig
from .markets import devig, flag, market_probs
from .model import FootyModel


def run_backtest(matches: pd.DataFrame, rows: pd.DataFrame, cfg: ModelConfig,
                 start: str, end: str, refit_days: int = 7, verbose: bool = True) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    played = matches[matches["hg"].notna()].set_index("match_id")
    recs = []
    dates = pd.date_range(start, end, freq=f"{refit_days}D")
    for k, d in enumerate(dates):
        nxt = d + pd.Timedelta(days=refit_days)
        test_ids = played[(played["date"] >= d) & (played["date"] < nxt)].index
        if len(test_ids) == 0:
            continue
        model = FootyModel(cfg).fit(rows, d)
        pr = model.predict_matches(rows[rows["match_id"].isin(test_ids)])
        for mid, r in pr.iterrows():
            M = model.score_matrix(r["lam_h"], r["lam_a"])
            p = market_probs(M)
            mt = played.loc[mid]
            recs.append({
                "match_id": mid, "date": mt["date"], "league": mt["league"],
                "home": mt["home"], "away": mt["away"], "hg": mt["hg"], "ag": mt["ag"],
                "known": bool(r["known"]),
                **{f"p_{x}": p[x] for x in ("1", "X", "2", "1X", "X2", "12", "O2.5", "U2.5",
                                             "O1.5", "U3.5", "BTTS_Y")},
                "lam_h": r["lam_h"], "lam_a": r["lam_a"],
                "odds_h": mt["odds_h"], "odds_d": mt["odds_d"], "odds_a": mt["odds_a"],
                "odds_o25": mt["odds_o25"], "odds_u25": mt["odds_u25"],
            })
        if verbose and k % 10 == 0:
            print(f"  backtest {d.date()}  matches so far: {len(recs)}", flush=True)
    return pd.DataFrame(recs)


def _add_market(bt: pd.DataFrame) -> pd.DataFrame:
    bt = bt.dropna(subset=["odds_h", "odds_d", "odds_a"]).copy()
    q = np.array([devig([a, b, c]) for a, b, c in bt[["odds_h", "odds_d", "odds_a"]].to_numpy()])
    bt["q_1"], bt["q_X"], bt["q_2"] = q[:, 0], q[:, 1], q[:, 2]
    ou = bt[["odds_o25", "odds_u25"]].to_numpy()
    qo = np.array([devig([a, b])[0] if np.isfinite(a) and np.isfinite(b) else np.nan for a, b in ou])
    bt["q_O2.5"], bt["q_U2.5"] = qo, 1 - qo
    bt["q_1X"], bt["q_X2"] = bt["q_1"] + bt["q_X"], bt["q_X"] + bt["q_2"]
    res = np.where(bt.hg > bt.ag, "1", np.where(bt.hg == bt.ag, "X", "2"))
    bt["res"] = res
    bt["over25"] = (bt.hg + bt.ag > 2.5).astype(int)
    return bt


def evaluate(bt: pd.DataFrame, fc: FlagConfig | None = None, ev_threshold: float = 0.05) -> dict:
    fc = fc or FlagConfig()
    bt = _add_market(bt[bt["known"]])
    out = {"n_matches": len(bt)}
    # ---- proper scoring rules
    eps = 1e-12
    for who, pre in (("model", "p_"), ("market", "q_")):
        P = bt[[f"{pre}1", f"{pre}X", f"{pre}2"]].to_numpy()
        Y = np.stack([(bt.res == o).to_numpy() for o in ("1", "X", "2")], 1).astype(float)
        out[f"{who}_1x2_logloss"] = float(-np.mean(np.sum(Y * np.log(P + eps), 1)))
        out[f"{who}_1x2_brier"] = float(np.mean(np.sum((P - Y) ** 2, 1)))
        ok = bt[f"{pre}O2.5"].notna()
        po, yo = bt.loc[ok, f"{pre}O2.5"].to_numpy(), bt.loc[ok, "over25"].to_numpy()
        out[f"{who}_ou25_logloss"] = float(-np.mean(yo * np.log(po + eps) + (1 - yo) * np.log(1 - po + eps)))
    # ---- blend (model + market average) - often the best practical estimate
    P = 0.5 * (bt[["p_1", "p_X", "p_2"]].to_numpy() + bt[["q_1", "q_X", "q_2"]].to_numpy())
    Y = np.stack([(bt.res == o).to_numpy() for o in ("1", "X", "2")], 1).astype(float)
    out["blend_1x2_logloss"] = float(-np.mean(np.sum(Y * np.log(P + eps), 1)))

    # ---- calibration (all 1X2 outcomes pooled)
    pm = bt[["p_1", "p_X", "p_2"]].to_numpy().ravel()
    ym = Y.ravel()
    bins = np.linspace(0, 1, 11)
    cal = pd.DataFrame({"p": pm, "y": ym, "bin": pd.cut(pm, bins)})
    out["calibration"] = cal.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)

    # ---- short-price "safe leg" test (the parlay use case)
    legs = []
    for sel, hit in (("1", bt.res == "1"), ("2", bt.res == "2"), ("1X", bt.res != "2"),
                     ("X2", bt.res != "1"), ("O2.5", bt.over25 == 1), ("U2.5", bt.over25 == 0)):
        q, p = bt[f"q_{sel}"], bt[f"p_{sel}"]
        legs.append(pd.DataFrame({"sel": sel, "q": q, "p": p, "hit": hit.astype(int)}))
    legs = pd.concat(legs).dropna()
    legs = legs[(legs.q >= 0.70) & (legs.q <= 0.87)]
    legs["edge"] = legs.p - legs.q
    legs["flag"] = [flag(p, q, 1 / (q * fc.default_overround), fc) for p, q in zip(legs.p, legs.q)]
    order = ["GREEN", "YELLOW", "ORANGE", "RED"]
    sp = legs.groupby("flag").agg(n=("hit", "size"), market_implied=("q", "mean"),
                                  model_prob=("p", "mean"), actual_hit=("hit", "mean"))
    out["safe_legs_by_flag"] = sp.reindex([o for o in order if o in sp.index]).round(3)
    out["safe_legs_all_hit_rate"] = float(legs.hit.mean())
    out["safe_legs_n"] = int(len(legs))

    # ---- value betting (1X2 + O/U2.5, flat 1u, average market odds)
    bets = []
    for sel, oc, hit in (("1", "odds_h", bt.res == "1"), ("X", "odds_d", bt.res == "X"),
                         ("2", "odds_a", bt.res == "2"), ("O2.5", "odds_o25", bt.over25 == 1),
                         ("U2.5", "odds_u25", bt.over25 == 0)):
        ev = bt[f"p_{sel}"] * bt[oc] - 1
        b = pd.DataFrame({"ev": ev, "odds": bt[oc], "hit": hit}).dropna()
        b = b[b.ev > ev_threshold]
        b["pnl"] = np.where(b.hit, b.odds - 1, -1.0)
        bets.append(b)
    bets = pd.concat(bets)
    out["value_bets_n"] = int(len(bets))
    out["value_bets_roi_%"] = float(100 * bets.pnl.mean()) if len(bets) else float("nan")
    return out


def print_report(res: dict) -> None:
    print(f"\nMatches evaluated: {res['n_matches']}")
    print("\n1X2 log-loss  (lower is better)")
    print(f"  model  {res['model_1x2_logloss']:.4f}   market {res['market_1x2_logloss']:.4f}"
          f"   50/50 blend {res['blend_1x2_logloss']:.4f}")
    print(f"1X2 Brier     model  {res['model_1x2_brier']:.4f}   market {res['market_1x2_brier']:.4f}")
    print(f"O/U2.5 log-loss model {res['model_ou25_logloss']:.4f}   market {res['market_ou25_logloss']:.4f}")
    print("\nCalibration (1X2 outcomes):")
    print(res["calibration"].to_string())
    print(f"\nSafe legs (market fair prob 70-87%, odds ~1.15-1.40): n={res['safe_legs_n']}, "
          f"overall hit rate {res['safe_legs_all_hit_rate']:.3f}")
    print(res["safe_legs_by_flag"].to_string())
    print(f"\nValue bets (EV>5%): n={res['value_bets_n']}  ROI {res['value_bets_roi_%']:.1f}%")


def tune_xi(matches, rows, xis, start, end, base: ModelConfig | None = None, refit_days=28):
    base = base or ModelConfig()
    out = []
    for xi in xis:
        cfg = ModelConfig(**{**base.__dict__, "xi": xi})
        bt = run_backtest(matches, rows, cfg, start, end, refit_days=refit_days, verbose=False)
        r = evaluate(bt)
        out.append({"xi": xi, "half_life_days": round(np.log(2) / xi) if xi > 0 else np.inf,
                    "model_1x2_logloss": r["model_1x2_logloss"],
                    "model_ou25_logloss": r["model_ou25_logloss"],
                    "market_1x2_logloss": r["market_1x2_logloss"]})
        print(out[-1], flush=True)
    return pd.DataFrame(out)
