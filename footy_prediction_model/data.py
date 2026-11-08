"""Data loading. Supports two formats, auto-detected:

1. football-data.co.uk season CSVs (Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, HS,
   AS, HST, AST, AvgH/B365H..., Avg>2.5...). Download from
   https://www.football-data.co.uk/data.php  (e.g. mmz4281/2627/E0.csv)
2. xgabora "Club-Football-Match-Data-2000-2025" Matches.csv (includes ClubElo).

Optional extras:
  - xG CSV:       date,home,away,home_xg,away_xg   (e.g. exported from Understat/FBref)
  - managers CSV: team,date                        (date the new manager took over)
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

STD_COLS = ["league", "date", "home", "away", "hg", "ag", "hs", "as_", "hst", "ast",
            "home_elo", "away_elo", "odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25",
            "home_xg", "away_xg"]


def _first(df: pd.DataFrame, names: list[str]):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _from_football_data(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "league": df["Div"],
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "home": df["HomeTeam"], "away": df["AwayTeam"],
        "hg": _first(df, ["FTHG"]), "ag": _first(df, ["FTAG"]),
        "hs": _first(df, ["HS"]), "as_": _first(df, ["AS"]),
        "hst": _first(df, ["HST"]), "ast": _first(df, ["AST"]),
        "home_elo": np.nan, "away_elo": np.nan,
        "odds_h": _first(df, ["AvgH", "BbAvH", "B365H", "PSH"]),
        "odds_d": _first(df, ["AvgD", "BbAvD", "B365D", "PSD"]),
        "odds_a": _first(df, ["AvgA", "BbAvA", "B365A", "PSA"]),
        "odds_o25": _first(df, ["Avg>2.5", "BbAv>2.5", "B365>2.5", "P>2.5"]),
        "odds_u25": _first(df, ["Avg<2.5", "BbAv<2.5", "B365<2.5", "P<2.5"]),
        "home_xg": np.nan, "away_xg": np.nan,
    })
    return out


def _from_xgabora(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "league": df["Division"],
        "date": pd.to_datetime(df["MatchDate"], errors="coerce"),
        "home": df["HomeTeam"], "away": df["AwayTeam"],
        "hg": _first(df, ["FTHome"]), "ag": _first(df, ["FTAway"]),
        "hs": _first(df, ["HomeShots"]), "as_": _first(df, ["AwayShots"]),
        "hst": _first(df, ["HomeTarget"]), "ast": _first(df, ["AwayTarget"]),
        "home_elo": _first(df, ["HomeElo"]), "away_elo": _first(df, ["AwayElo"]),
        "odds_h": _first(df, ["OddHome"]), "odds_d": _first(df, ["OddDraw"]),
        "odds_a": _first(df, ["OddAway"]),
        "odds_o25": _first(df, ["Over25"]), "odds_u25": _first(df, ["Under25"]),
        "home_xg": np.nan, "away_xg": np.nan,
    })


def _from_openmatchday(df: pd.DataFrame, league: str) -> pd.DataFrame:
    return pd.DataFrame({
        "league": league,
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "home": df["home"], "away": df["away"],
        "hg": _first(df, ["ft_home"]), "ag": _first(df, ["ft_away"]),
        "hs": _first(df, ["home_shots"]), "as_": _first(df, ["away_shots"]),
        "hst": _first(df, ["home_target"]), "ast": _first(df, ["away_target"]),
        "home_elo": np.nan, "away_elo": np.nan,
        "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan,
        "odds_o25": np.nan, "odds_u25": np.nan, "home_xg": np.nan, "away_xg": np.nan,
    })


def load_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
    if "Division" in df.columns and "FTHome" in df.columns:
        out = _from_xgabora(df)
    elif "Div" in df.columns and "HomeTeam" in df.columns:
        out = _from_football_data(df)
    elif "ft_home" in df.columns and "gw" in df.columns:
        from .fetch import league_code_from_path
        out = _from_openmatchday(df, league_code_from_path(path))
        out["_src"] = "om"
        return out
    else:
        raise ValueError(f"Unrecognised data format: {path}")
    out["_src"] = "ref"
    return out


def season_of(d: pd.Series) -> pd.Series:
    return np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)


EURO_CODES = {"UCL", "UEL", "UECL"}

# Hard cases the automatic matcher gets wrong. Extend via --aliases CSV (source,target[,scope]).
ALIASES_DOMESTIC = {"Borussia Mönchengladbach": "M'gladbach", "Bor. Mönchengladbach": "M'gladbach",
                    "Mönchengladbach": "M'gladbach"}
ALIASES_EURO = {"Paris": "Paris SG", "PSG": "Paris SG", "Man Utd": "Man United", "Spurs": "Tottenham",
                "Inter Milan": "Inter", "Internazionale": "Inter", "Atlético de Madrid": "Ath Madrid",
                "Atletico Madrid": "Ath Madrid", "Atleti": "Ath Madrid", "Athletic Club": "Ath Bilbao",
                "Bayern München": "Bayern Munich", "FC Bayern München": "Bayern Munich",
                "B. Dortmund": "Dortmund", "Borussia Dortmund": "Dortmund",
                "Bayer 04 Leverkusen": "Leverkusen", "Leipzig": "RB Leipzig",
                "Eintracht Frankfurt": "Ein Frankfurt", "Real Sociedad": "Sociedad",
                "Real Betis": "Betis", "Olympiacos": "Olympiakos", "AEK Athens": "AEK",
                "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon",
                "LOSC Lille": "Lille", "AS Monaco": "Monaco", "Sporting CP": "Sp Lisbon",
                "SL Benfica": "Benfica", "FC Porto": "Porto", "SC Braga": "Sp Braga",
                "PSV": "PSV Eindhoven", "AZ": "AZ Alkmaar", "Club Brugge": "Club Brugge"}
EXTRA_ALIASES: dict = {}   # filled from --aliases
_STOP = {"fc", "cf", "sc", "ac", "as", "rc", "ud", "cd", "sv", "afc", "ssc", "ss", "us", "club",
         "de", "1.", "sl", "cp", "sco", "ogc", "losc", "stade", "olympique", "real", "athletic"}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _STOP and not t.isdigit()]
    return " ".join(toks) or s.strip()


def harmonize_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map open-matchday team names onto the historical (football-data style) names.
    1) exact fixture overlap (same league, date +-1 day, same score)
    2) fuzzy match within league  3) Europe: union map, then fuzzy across all names."""
    ref, om = df[df["_src"] == "ref"], df[df["_src"] == "om"]
    if om.empty or ref.empty:
        return df, {}
    mapping: dict[str, str] = {}
    recent = ref[ref["date"] >= ref["date"].max() - pd.Timedelta(days=800)]
    for lg in sorted(set(om["league"]) - EURO_CODES):
        o = om[(om["league"] == lg) & om["hg"].notna()]
        r = recent[recent["league"] == lg]
        if r.empty:
            continue
        votes = Counter()
        for shift in (0, 1, -1):
            rr = r.assign(date=r["date"] + pd.Timedelta(days=shift))
            j = o.merge(rr, on=["date", "hg", "ag"], suffixes=("_o", "_r"))
            votes.update(zip(j["home_o"], j["home_r"]))
            votes.update(zip(j["away_o"], j["away_r"]))
        names_o = sorted(set(o["home"]) | set(o["away"]))
        taken = set()
        for on in names_o:
            tgt = EXTRA_ALIASES.get(on) or ALIASES_DOMESTIC.get(on)
            if tgt:
                mapping[on] = tgt
                taken.add(tgt)
        for (on, rn), _ in votes.most_common():
            if on not in mapping and rn not in taken and on in names_o:
                if votes[(on, rn)] >= 2 or on == rn:
                    mapping[on] = rn
                    taken.add(rn)
        ref_names = sorted((set(r["home"]) | set(r["away"])) - taken)
        normed = {_norm(n): n for n in ref_names}
        for on in names_o:
            if on in mapping:
                continue
            if on in ref_names:
                mapping[on] = on
                continue
            no = _norm(on)
            sub = [k for k in normed if k and (set(k.split()) <= set(no.split())
                                                or set(no.split()) <= set(k.split()))]
            m = sub[:1] if len(sub) == 1 else difflib.get_close_matches(no, list(normed), n=1, cutoff=0.72)
            if m:
                mapping[on] = normed[m[0]]
                del normed[m[0]]
    # European competitions get their own mapping: aliases, exact, domestic map, strict fuzzy
    all_ref = set(ref["home"]) | set(ref["away"])
    normed_all = {_norm(n): n for n in all_ref}
    euro_map: dict[str, str] = {}
    is_e = om["league"].isin(EURO_CODES)
    for on in sorted(set(om.loc[is_e, "home"]) | set(om.loc[is_e, "away"])):
        tgt = EXTRA_ALIASES.get(on) or ALIASES_EURO.get(on)
        if tgt:
            euro_map[on] = tgt
        elif on in all_ref:
            continue
        elif on in mapping:
            euro_map[on] = mapping[on]
        else:
            m = difflib.get_close_matches(_norm(on), list(normed_all), n=1, cutoff=0.85)
            if m:
                euro_map[on] = normed_all[m[0]]
    df = df.copy()
    is_dom = (df["_src"] == "om") & ~df["league"].isin(EURO_CODES)
    is_eu = (df["_src"] == "om") & df["league"].isin(EURO_CODES)
    for col in ("home", "away"):
        df.loc[is_dom, col] = df.loc[is_dom, col].replace(mapping)
        df.loc[is_eu, col] = df.loc[is_eu, col].replace(euro_map)
    is_om = df["_src"] == "om"
    # drop open-matchday duplicates of fixtures already in the reference data
    df["season"] = season_of(df["date"])
    key = ["league", "season", "home", "away"]
    dup = df[is_om].merge(df[~is_om][key].drop_duplicates(), on=key, how="left", indicator=True)
    keep_om = dup["_merge"].eq("left_only").to_numpy()
    df = pd.concat([df[~is_om], df[is_om][keep_om]], ignore_index=True)
    # same name in two domestic leagues (e.g. Aris Thessaloniki / Aris Limassol)
    dom = df[~df["league"].isin(EURO_CODES)]
    long = pd.concat([dom[["league", "home"]].rename(columns={"home": "t"}),
                      dom[["league", "away"]].rename(columns={"away": "t"})])
    cnt = long.groupby(["t", "league"]).size().reset_index(name="n")
    for t, g in cnt.groupby("t"):
        if len(g) > 1:
            main = g.sort_values("n").iloc[-1]["league"]
            for lg in g["league"]:
                if lg != main:
                    for col in ("home", "away"):
                        df.loc[(df["league"] == lg) & (df[col] == t), col] = f"{t} ({lg})"
    allmap = {**{k: v for k, v in mapping.items() if k != v},
              **{f"[EU] {k}": v for k, v in euro_map.items() if k != v}}
    return df, allmap


def load_matches(paths: list[str], leagues: list[str] | None = None,
                 since: str | None = "2015-07-01", xg_path: str | None = None,
                 include_europe: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Load and merge sources. Later files override earlier ones for the same match.
    European competition results (UCL/UEL/UECL) are kept when include_europe: they
    feed rest days / form / Elo but are not used to train the domestic model."""
    frames = [load_file(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date", "home", "away"])
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    if leagues:
        keep = set(leagues) | (EURO_CODES if include_europe else set())
        df = df[df["league"].isin(keep)]
    if since:
        df = df[df["date"] >= pd.Timestamp(since)]
    df = df.drop_duplicates(subset=["league", "date", "home", "away"], keep="last")
    df, mapping = harmonize_names(df)
    if verbose and mapping:
        print("Team names mapped:", ", ".join(f"{a}->{b}" for a, b in sorted(mapping.items())))
    df = df.drop(columns=["_src"])
    if xg_path:
        xg = pd.read_csv(xg_path, parse_dates=["date"])
        df = df.drop(columns=["home_xg", "away_xg"]).merge(
            xg[["date", "home", "away", "home_xg", "away_xg"]], on=["date", "home", "away"], how="left")
    df = df.sort_values(["date", "league", "home"]).reset_index(drop=True)
    df["season"] = season_of(df["date"])
    df["match_id"] = np.arange(len(df))
    return df[["match_id", "season"] + STD_COLS]


def load_managers(path: str | None) -> dict[str, list[pd.Timestamp]]:
    if not path:
        return {}
    m = pd.read_csv(path, parse_dates=["date"])
    out: dict[str, list] = {}
    for team, g in m.groupby("team"):
        out[str(team).strip()] = sorted(g["date"].tolist())
    return out


def xg_proxy(goals, shots, sot):
    """Shot-based xG proxy when real xG is missing: ~0.28 per shot on target,
    ~0.03 per shot off target. Falls back to goals if shots are missing."""
    goals = np.asarray(goals, float)
    shots = np.asarray(shots, float)
    sot = np.asarray(sot, float)
    proxy = 0.28 * sot + 0.03 * np.clip(shots - sot, 0, None)
    return np.where(np.isnan(proxy), goals, proxy)
