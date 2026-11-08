"""Fetch fresh current-season results.

Source: open-matchday (github.com/oladeeayo/open-matchday) - free daily-refreshed
CSVs (results, shots, shots on target) for 75 competitions. Licence: informational,
non-commercial use with attribution.

Historical seasons (with odds + ClubElo): xgabora/Club-Football-Match-Data-2000-2025
(github.com/xgabora/Club-Football-Match-Data-2000-2025).

  python -m footy_prediction_model fetch --dest data
"""
from __future__ import annotations

import os
import re
import urllib.request

OM_BASE = "https://raw.githubusercontent.com/oladeeayo/open-matchday/master/data"
HIST_URL = ("https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/"
            "main/data/Matches.csv")

# open-matchday folder -> football-data.co.uk style league code
OM_FOLDERS = {
    "ENG_1_premier-league": "E0", "ENG_2_championship": "E1",
    "ESP_1_la-liga": "SP1", "ITA_1_serie-a": "I1", "GER_1_bundesliga": "D1",
    "FRA_1_ligue-1": "F1", "NED_1_eredivisie": "N1", "POR_1_primeira-liga": "P1",
    "GRE_1_super-league": "G1", "CYP_1_first-division": "CY1", "SCO_1_premiership": "SC0",
    "BEL_1_pro-league": "B1", "TUR_1_super-lig": "T1",
    "UEFA_1_champions-league": "UCL", "UEFA_1_europa-league": "UEL",
    "UEFA_1_conference-league": "UECL",
}
CODE_TO_FOLDER = {v: k for k, v in OM_FOLDERS.items()}


def league_code_from_path(path: str) -> str:
    for folder, code in OM_FOLDERS.items():
        if folder in path:
            return code
    m = re.search(r"om_([A-Za-z0-9]+)_", os.path.basename(path))
    if m:
        return m.group(1)
    raise ValueError(f"Can't infer league from {path}; name it om_<CODE>_results_2627.csv")


def _get(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def fetch_current(leagues: list[str], dest: str = "data", season: str = "2627",
                  europe: bool = True) -> list[str]:
    codes = list(leagues) + (["UCL", "UEL", "UECL"] if europe else [])
    paths = []
    for code in codes:
        folder = CODE_TO_FOLDER.get(code)
        if not folder:
            print(f"  no open-matchday folder for {code}, skipping")
            continue
        out = os.path.join(dest, folder, f"results_{season}.csv")
        try:
            paths.append(_get(f"{OM_BASE}/{folder}/results_{season}.csv", out))
            print(f"  {code}: {out}")
        except Exception as e:  # noqa: BLE001
            print(f"  {code}: failed ({e})")
    return paths


def fetch_history(dest: str = "data") -> str:
    out = os.path.join(dest, "history", "Matches.csv")
    print("  downloading history (~45 MB)...")
    return _get(HIST_URL, out)
