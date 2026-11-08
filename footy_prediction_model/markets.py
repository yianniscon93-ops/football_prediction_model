"""Turn a scoreline matrix into market probabilities, de-margin bookmaker odds,
and assign v3 flags."""
from __future__ import annotations

import re

import numpy as np

from .config import FlagConfig


def market_probs(M: np.ndarray) -> dict[str, float]:
    n = M.shape[0]
    i, j = np.indices(M.shape)
    tot = i + j
    p = {
        "1": M[i > j].sum(), "X": M[i == j].sum(), "2": M[i < j].sum(),
    }
    p["1X"], p["X2"], p["12"] = p["1"] + p["X"], p["X"] + p["2"], p["1"] + p["2"]
    for k in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        p[f"O{k}"] = M[tot > k].sum()
        p[f"U{k}"] = M[tot < k].sum()
        p[f"HO{k}"] = M[i > k].sum()
        p[f"HU{k}"] = M[i < k].sum()
        p[f"AO{k}"] = M[j > k].sum()
        p[f"AU{k}"] = M[j < k].sum()
    p["BTTS_Y"] = M[(i > 0) & (j > 0)].sum()
    p["BTTS_N"] = 1 - p["BTTS_Y"]
    p["H_SCORE"] = 1 - M[0, :].sum()
    p["A_SCORE"] = 1 - M[:, 0].sum()
    p["xg_h"] = float((i * M).sum())
    p["xg_a"] = float((j * M).sum())
    return {k: float(v) for k, v in p.items()}


_GREEK = {"διπλή ευκαιρία": "", "τελικό αποτέλεσμα": "", "γκολ over/under": ""}


def parse_selection(sel: str, home: str, away: str) -> str:
    """Normalise a selection string to a market key.
    Accepts: 1, X, 2, 1X, X2, 12, home/away/draw, team names,
    'Over 2.5', 'Under 3.5', 'O2.5', 'BTTS Yes/No', 'GG', 'NG',
    'Home to score', 'Away to score', 'Home Over 1.5', 'Away Under 0.5'."""
    s = sel.strip().lower()
    for g, rep in _GREEK.items():
        s = s.replace(g, rep).strip()
    if s in {"1", "home", home.lower()}:
        return "1"
    if s in {"x", "draw", "ισοπαλία"}:
        return "X"
    if s in {"2", "away", away.lower()}:
        return "2"
    if s.upper() in {"1X", "X2", "12", "X1", "2X"}:
        return {"X1": "1X", "2X": "X2"}.get(s.upper(), s.upper())
    if s in {"btts yes", "btts", "gg", "both teams to score", "btts_y"}:
        return "BTTS_Y"
    if s in {"btts no", "ng", "btts_n"}:
        return "BTTS_N"
    if s in {"home to score", f"{home.lower()} to score"}:
        return "H_SCORE"
    if s in {"away to score", f"{away.lower()} to score"}:
        return "A_SCORE"
    m = re.match(r"^(home|away|" + re.escape(home.lower()) + "|" + re.escape(away.lower())
                 + r")?\s*(over|under|o|u)\s*([0-9.]+)$", s)
    if m:
        who, ou, line = m.groups()
        prefix = ""
        if who in ("home", home.lower()):
            prefix = "H"
        elif who in ("away", away.lower()):
            prefix = "A"
        return f"{prefix}{'O' if ou.startswith('o') else 'U'}{float(line)}"
    raise ValueError(f"Can't parse selection '{sel}' for {home} v {away}")


def devig(odds: list[float]) -> list[float]:
    """Power method: find k so that sum((1/o)^k) = 1. Handles favourite-longshot bias
    better than proportional scaling."""
    inv = np.array([1 / o for o in odds])
    lo, hi = 0.5, 2.0
    for _ in range(60):
        k = (lo + hi) / 2
        s = (inv ** k).sum()
        lo, hi = (k, hi) if s > 1 else (lo, k)
    return list(inv ** ((lo + hi) / 2))


def fair_single(odds: float, overround: float) -> float:
    """Fair probability for one price when the rest of the market is unknown."""
    return min((1 / odds) / overround, 0.999)


def flag(model_p: float, fair_p: float, odds: float, fc: FlagConfig) -> str:
    edge = model_p - fair_p
    if edge >= fc.green:
        f = "GREEN"
    elif edge >= fc.yellow:
        f = "YELLOW"
    elif edge >= fc.orange:
        f = "ORANGE"
    else:
        f = "RED"
    if odds > fc.odds_yellow_floor and f == "GREEN":
        f = "YELLOW"
    return f
