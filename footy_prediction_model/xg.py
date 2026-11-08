"""Real xG per match from FotMob (Opta model).

Why FotMob: Understat has been serving a maintenance page since at least Nov 2026, and
FBref lost Opta advanced stats in Jan 2026. FotMob's public JSON (the one its own web
app uses) carries Opta xG for every finished match where Opta tracks it.

Coverage checked Nov 2026:
  top 5 leagues  2020/21 onwards       Greek SL   partial 2024/25, full from 2025/26
  UCL/UEL/UECL   yes                   Cyprus     no xG at all (CY1 stays on the shot proxy)

Pipeline (python -m footy_prediction_model xg --data data):
  1. update_raw():        FotMob fixtures -> per-match xG, cached in data/xg/fotmob_raw.csv
                          (resumable; already-fetched matches are never requested again)
  2. map_to_reference():  FotMob team ids -> our (football-data) team names, learned from
                          fixture overlap: same league, date +-1 day, same score
  3. write data/xg/xg.csv: date,league,home,away,home_xg,away_xg   (feeds --xg)

Unofficial endpoint, personal research use: keep the request rate low and don't
republish the data (it is Opta's).
"""
from __future__ import annotations

import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .data import season_of

FOTMOB = "https://www.fotmob.com/api/data"
FOTMOB_IDS = {"E0": 47, "SP1": 87, "I1": 55, "D1": 54, "F1": 53, "N1": 57, "P1": 61,
              "B1": 40, "SC0": 64, "T1": 71, "G1": 135,
              "UCL": 42, "UEL": 73, "UECL": 10216}
NO_XG = {"CY1"}   # FotMob has no xG for these
RAW_COLS = ["fm_id", "league", "season", "utc", "home_id", "away_id", "home_fm", "away_fm",
            "hg", "ag", "home_xg", "away_xg"]
XG_COLS = ["date", "league", "home", "away", "home_xg", "away_xg"]


def season_label(start_year: int) -> str:
    return f"{start_year}/{start_year + 1}"


def current_season(today: pd.Timestamp | None = None) -> int:
    t = today or pd.Timestamp.today()
    return t.year if t.month >= 7 else t.year - 1


class FotMobClient:
    """Thin client with a global request-rate cap shared by all worker threads."""

    def __init__(self, max_rps: float = 4.0, timeout: tuple = (10, 20), verbose: bool = True):
        self.timeout, self.gap, self.verbose = timeout, 1.0 / max_rps, verbose
        self._lock, self._next = threading.Lock(), 0.0
        self._local = threading.local()   # requests.Session is not thread-safe

    @property
    def s(self):
        if not hasattr(self._local, "s"):
            import requests  # local import: only needed when fetching
            self._local.s = requests.Session()
            self._local.s.headers["User-Agent"] = "Mozilla/5.0"
        return self._local.s

    def _get(self, path: str, **params) -> dict:
        for attempt in range(4):
            with self._lock:
                wait = self._next - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._next = time.monotonic() + self.gap
            try:
                r = self.s.get(f"{FOTMOB}/{path}", params=params, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                if self.verbose:
                    print(f"    retry {path} {params} after {type(e).__name__}: {e}", flush=True)
                self._local.__dict__.pop("s", None)   # fresh connection
                time.sleep(2 ** attempt * 2)
        raise AssertionError("unreachable")

    def finished_matches(self, league_id: int, season: str) -> list[dict]:
        d = self._get("leagues", id=league_id, season=season)
        out = []
        for m in d.get("fixtures", {}).get("allMatches", []):
            st = m.get("status", {})
            if not st.get("finished") or st.get("cancelled") or st.get("awarded"):
                continue
            try:
                hg, ag = (int(x) for x in st["scoreStr"].split(" - "))
            except (KeyError, ValueError):
                continue
            out.append({"fm_id": int(m["id"]), "utc": st["utcTime"],
                        "home_id": int(m["home"]["id"]), "away_id": int(m["away"]["id"]),
                        "home_fm": m["home"]["name"], "away_fm": m["away"]["name"],
                        "hg": hg, "ag": ag})
        return out

    def match_xg(self, match_id: int) -> tuple[float, float]:
        d = self._get("matchDetails", matchId=match_id)
        return parse_match_xg(d)


def parse_match_xg(details: dict) -> tuple[float, float]:
    """Full-match xG from a FotMob matchDetails payload; (nan, nan) when absent."""
    try:
        groups = details["content"]["stats"]["Periods"]["All"]["stats"]
    except (KeyError, TypeError):
        return np.nan, np.nan
    for g in groups:
        for s in g.get("stats", []):
            if s.get("key") == "expected_goals" and s.get("stats") and s["stats"][0] is not None:
                try:
                    return float(s["stats"][0]), float(s["stats"][1])
                except (TypeError, ValueError):
                    return np.nan, np.nan
    return np.nan, np.nan


def _load_raw(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=RAW_COLS)


def update_raw(leagues: list[str], seasons: list[int], raw_path: str,
               client: FotMobClient | None = None, workers: int = 4,
               refetch_days: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Fetch xG for every finished match not yet in the cache. Matches without xG are
    cached as NaN and not asked again, except those finished in the last refetch_days
    (FotMob sometimes adds xG a little after full time). Seasons where the first
    matches have no xG are skipped (Opta doesn't cover them)."""
    client = client or FotMobClient()
    raw = _load_raw(raw_path)
    have = raw.set_index("fm_id") if len(raw) else None
    recent_cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=refetch_days)
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)

    for lg in leagues:
        if lg in NO_XG or lg not in FOTMOB_IDS:
            if verbose:
                print(f"  {lg}: no FotMob xG, skipping")
            continue
        for se in seasons:
            label = season_label(se)
            try:
                fx = client.finished_matches(FOTMOB_IDS[lg], label)
            except Exception as e:  # noqa: BLE001
                print(f"  {lg} {label}: fixtures failed ({e})")
                continue
            todo = []
            for m in fx:
                if have is not None and m["fm_id"] in have.index:
                    old = have.loc[m["fm_id"]]
                    if not (pd.isna(old["home_xg"]) and pd.Timestamp(m["utc"]) >= recent_cut):
                        continue
                todo.append(m)
            if not todo:
                if verbose:
                    print(f"  {lg} {label}: {len(fx)} finished, cache up to date")
                continue
            # coverage probe: don't hammer seasons Opta doesn't cover
            seen_xg = have is not None and have[(have["league"] == lg) & (have["season"] == se)
                                                ]["home_xg"].notna().any()
            new = []
            if not seen_xg and len(todo) > 10:
                probe = [client.match_xg(m["fm_id"]) for m in todo[:3]]
                if all(np.isnan(h) for h, _ in probe):
                    if verbose:
                        print(f"  {lg} {label}: no xG in this season, skipping")
                    continue
                new = [{**m, "league": lg, "season": se, "home_xg": h, "away_xg": a}
                       for m, (h, a) in zip(todo[:3], probe)]
                todo = todo[3:]
            if verbose:
                print(f"  {lg} {label}: fetching {len(todo)} of {len(fx)} matches...", flush=True)
            with ThreadPoolExecutor(workers) as ex:
                futs = {ex.submit(client.match_xg, m["fm_id"]): m for m in todo}
                for k, f in enumerate(as_completed(futs), 1):
                    m = futs[f]
                    try:
                        hx, ax = f.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"    match {m['fm_id']} failed ({e}); will retry next run")
                        continue
                    new.append({**m, "league": lg, "season": se, "home_xg": hx, "away_xg": ax})
                    if k % 100 == 0:
                        raw = _merge_save(raw, new, raw_path)
                        new = []
                        if verbose:
                            print(f"    {k}/{len(todo)}", flush=True)
            raw = _merge_save(raw, new, raw_path)
            have = raw.set_index("fm_id")
    return raw


def _merge_save(raw: pd.DataFrame, new: list[dict], path: str) -> pd.DataFrame:
    if new:
        add = pd.DataFrame(new)[RAW_COLS]
        raw = add if raw.empty else pd.concat([raw[~raw["fm_id"].isin(add["fm_id"])], add])
        raw = raw.sort_values(["league", "utc"]).reset_index(drop=True)
        raw.to_csv(path, index=False)
    return raw


# --------------------------------------------------------------------- name mapping
def map_to_reference(raw: pd.DataFrame, matches: pd.DataFrame, min_votes: int = 2
                     ) -> tuple[pd.DataFrame, dict]:
    """Attach FotMob xG to our matches. Team identity is learned by fixture overlap:
    a FotMob team id and one of our team names "vote" each time they appear in a
    fixture with the same league, date (+-1 day, UTC vs local) and score. Votes are
    pooled across competitions, so a club mapped from its league games is also
    mapped in Europe. Returns (xg frame in XG_COLS, {fotmob name: our name})."""
    fm = raw.copy()
    fm["date"] = pd.to_datetime(fm["utc"], utc=True).dt.tz_localize(None).dt.normalize()
    ref = matches[matches["hg"].notna()][["league", "date", "home", "away", "hg", "ag"]].copy()
    votes: Counter = Counter()
    for shift in (0, 1, -1):
        r = ref.assign(date=ref["date"] + pd.Timedelta(days=shift))
        j = fm.merge(r, on=["league", "date", "hg", "ag"])
        votes.update(zip(j["home_id"], j["home"]))
        votes.update(zip(j["away_id"], j["away"]))
    id_to_name, taken = {}, set()
    for (fid, name), n in votes.most_common():
        if n < min_votes:
            break
        if fid not in id_to_name and name not in taken:
            id_to_name[fid] = name
            taken.add(name)
    fm["home"] = fm["home_id"].map(id_to_name)
    fm["away"] = fm["away_id"].map(id_to_name)
    fm = fm.dropna(subset=["home", "away", "home_xg", "away_xg"])
    fm["season"] = season_of(fm["date"])
    # take our date (local) for the fixture, so the CSV joins on any key
    ref["season"] = season_of(ref["date"])
    out = fm.merge(ref[["league", "season", "home", "away", "date"]],
                   on=["league", "season", "home", "away"], suffixes=("_fm", ""))
    out = out.drop_duplicates(subset=["league", "date", "home", "away"])
    names = {}
    for fid, name in id_to_name.items():
        fn = raw.loc[raw["home_id"] == fid, "home_fm"]
        if len(fn) and fn.iloc[0] != name:
            names[fn.iloc[0]] = name
    return out[XG_COLS].sort_values(["date", "league", "home"]).reset_index(drop=True), names


def coverage(xg: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Share of played matches per league-season that got real xG."""
    m = matches[matches["hg"].notna()].copy()
    k = xg.assign(has_xg=1)[["league", "date", "home", "away", "has_xg"]]
    m = m.merge(k, on=["league", "date", "home", "away"], how="left")
    t = m.groupby(["league", "season"]).agg(matches=("hg", "size"), with_xg=("has_xg", "sum"))
    t["pct"] = (100 * t["with_xg"] / t["matches"]).round(1)
    return t[t["with_xg"] > 0]
