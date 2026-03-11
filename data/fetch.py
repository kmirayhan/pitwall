"""
PITWALL — Phase 1: FIA PDF Pipeline
Downloads History Chart + Pit Stop Summary PDFs from fia.com.

Round numbers = F1 race weekend round numbers (NOT F2 series round numbers).
All circuit codes and round numbers verified by direct URL probe.

Usage: py data\fetch.py
Already-downloaded PDFs are cached and skipped automatically.
"""

import requests
import pdfplumber
import pandas as pd
import numpy as np
import json
import time
import io
import re
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).parent.parent
PDF_DIR    = ROOT / "data" / "pdfs"
OUTPUT_DIR = ROOT / "data" / "races"
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

# ── VERIFIED RACE CALENDAR ────────────────────────────────────────────────────
# (year, f1_round, fia_circuit_code, display_name)
# Round numbers confirmed by direct HTTP probe against FIA PDF URLs.
# Gaps = FIA does not publish history chart for that race (no F2 support weekend
# or PDF not available publicly).

RACES = [
    # ── 2022 ──────────────────────────────────────────────────────────────────
    # Confirmed: ksa=02, ita=04(Imola), hun=13, bel=14
    (2022,  2, "ksa",  "Jeddah"),
    (2022,  4, "ita",  "Imola"),        # 'ita' here = Imola (Italy R04 2022)
    (2022, 13, "hun",  "Budapest"),
    (2022, 14, "bel",  "Spa"),

    # ── 2023 ──────────────────────────────────────────────────────────────────
    # Confirmed: aus=03, mon=07(Barcelona/Montmelo), aut=10, gbr=11, hun=12, ita=15
    (2023,  3, "aus",  "Melbourne"),
    (2023,  7, "mon",  "Barcelona"),    # 'mon'=Montmelo circuit, F1 R07=Canada
    (2023, 10, "aut",  "Red_Bull_Ring"),
    (2023, 11, "gbr",  "Silverstone"),
    (2023, 12, "hun",  "Budapest"),
    (2023, 15, "ita",  "Monza"),

    # ── 2024 ──────────────────────────────────────────────────────────────────
    # Confirmed: all 16 rounds verified
    (2024,  1, "brn",  "Bahrain"),
    (2024,  2, "ksa",  "Jeddah"),
    (2024,  3, "aus",  "Melbourne"),
    (2024,  7, "imla", "Imola"),
    (2024,  8, "mon",  "Monaco"),
    (2024,  9, "can",  "Montreal"),
    (2024, 10, "esp",  "Barcelona"),
    (2024, 11, "aut",  "Red_Bull_Ring"),
    (2024, 12, "gbr",  "Silverstone"),
    (2024, 13, "hun",  "Budapest"),
    (2024, 14, "bel",  "Spa"),
    (2024, 15, "ned",  "Zandvoort"),
    (2024, 16, "ita",  "Monza"),
    (2024, 17, "bak",  "Baku"),
    (2024, 23, "qat",  "Qatar"),
    (2024, 24, "adh",  "Abu_Dhabi"),
]


def pdf_url(year, rnd, circuit, doc):
    return (f"https://www.fia.com/sites/default/files/"
            f"{year}_{rnd:02d}_{circuit}_f2_r2_timing_featurerace{doc}_v01.pdf")


def download_pdf(url, cache_path):
    """Download with cache. Returns bytes or None."""
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        return cache_path.read_bytes()
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200 and b"%PDF" in r.content[:10]:
                cache_path.write_bytes(r.content)
                time.sleep(1.5)   # polite — FIA rate limits hard
                return r.content
            return None
        except Exception:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    return None


def laptime_to_sec(t):
    try:
        t = str(t).strip()
        parts = t.split(":")
        if len(parts) == 2:
            return round(int(parts[0]) * 60 + float(parts[1]), 3)
        return round(float(parts[0]), 3)
    except Exception:
        return None


# ── HISTORY CHART PARSER ──────────────────────────────────────────────────────
# Format (confirmed from real Bahrain/Jeddah/Melbourne 2024 PDFs):
#   Header: "LAP 1 GAP TIME  LAP 2 GAP TIME ..." (5 laps per page, 7 pages)
#   Leader row: car_no  lap_time  (no gap)
#   Normal row: car_no  gap_sec  lap_time
#   Pit row:    car_no  PIT  lap_time

def parse_history_chart(pdf_bytes):
    rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = [l.strip() for l in text.split("\n") if l.strip()]

            # Find header: "LAP 1 GAP TIME  LAP 2 ..."
            lap_nums   = []
            header_idx = None
            for i, line in enumerate(lines):
                found = re.findall(r'LAP\s+(\d+)', line)
                if len(found) >= 2:
                    lap_nums   = [int(n) for n in found]
                    header_idx = i
                    break

            if not lap_nums or header_idx is None:
                continue

            for line in lines[header_idx + 1:]:
                if any(s in line for s in ["©", "Page", "FORMULA", "FIA", "trade",
                                            "marks", "prior", "results", "part"]):
                    continue
                tokens = line.split()
                if not tokens:
                    continue

                i   = 0
                col = 0
                while i < len(tokens) and col < len(lap_nums):
                    lap = lap_nums[col]
                    tok = tokens[i]
                    if not tok.isdigit() or not (1 <= int(tok) <= 99):
                        i += 1
                        continue
                    car_no = int(tok)
                    i += 1
                    if i >= len(tokens):
                        break
                    next_tok = tokens[i]

                    if next_tok.upper() == "PIT":
                        is_pit = True
                        gap    = None
                        i += 1
                    elif re.match(r'^\d+\.\d+$', next_tok) and ":" not in next_tok:
                        is_pit = False
                        gap    = float(next_tok)
                        i += 1
                    else:
                        is_pit = False
                        gap    = 0.0

                    if i < len(tokens):
                        t_sec = laptime_to_sec(tokens[i])
                        i += 1
                    else:
                        t_sec = None

                    if t_sec and t_sec > 60:
                        rows.append({
                            "car_no":        car_no,
                            "lap":           lap,
                            "lap_time_sec":  t_sec,
                            "gap_to_leader": gap,
                            "is_pit_lap":    is_pit,
                        })
                    col += 1

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["car_no", "lap", "lap_time_sec", "gap_to_leader", "is_pit_lap"])


# ── PIT STOP PARSER ───────────────────────────────────────────────────────────
# Confirmed format: NO DRIVER TEAM LAP TIME_OF_DAY STOP DURATION TOTAL_TIME
# Tokens from right: [-1]=total, [-2]=duration, [-3]=stop, [-4]=time_of_day, [-5]=lap

def parse_pitstops(pdf_bytes):
    rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if any(s in line for s in ["NO ", "DRIVER", "©", "FORMULA", "FIA",
                                            "trade", "part", "results", "marks"]):
                    continue
                tokens = line.split()
                if len(tokens) < 6:
                    continue
                if not tokens[0].isdigit() or not (1 <= int(tokens[0]) <= 99):
                    continue
                try:
                    car_no   = int(tokens[0])
                    lap      = int(tokens[-5])
                    stop_num = int(tokens[-3])
                    dur_sec  = laptime_to_sec(tokens[-2])
                    if dur_sec and lap > 0:
                        rows.append({
                            "car_no":       car_no,
                            "pit_lap":      lap,
                            "stop_num":     stop_num,
                            "duration_sec": dur_sec,
                        })
                except (ValueError, IndexError):
                    continue
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["car_no", "pit_lap", "stop_num", "duration_sec"])


# ── SC INFERENCE ──────────────────────────────────────────────────────────────

def infer_sc_laps(laps_df):
    if laps_df.empty:
        return []
    valid = laps_df[
        ~laps_df["is_pit_lap"] &
        (laps_df["lap_time_sec"] > 60) &
        (laps_df["lap"] > 1)
    ]
    if valid.empty:
        return []
    race_med = valid["lap_time_sec"].median()
    lap_meds = valid.groupby("lap")["lap_time_sec"].median()
    return sorted(lap_meds[lap_meds > race_med * 1.08].index.tolist())


def find_windows(sc_laps):
    if not sc_laps:
        return []
    events = []
    start = prev = sc_laps[0]
    for lap in sc_laps[1:]:
        if lap > prev + 1:
            events.append({"start": start, "end": prev,
                           "duration_laps": prev - start + 1})
            start = lap
        prev = lap
    events.append({"start": start, "end": prev,
                   "duration_laps": prev - start + 1})
    return events


# ── MAIN PROCESSOR ────────────────────────────────────────────────────────────

def process_race(year, rnd, circuit, name):
    tag = f"{year}_R{rnd:02d}_{name}"
    print(f"  {tag}", end="", flush=True)

    hist_bytes = download_pdf(
        pdf_url(year, rnd, circuit, "historychart"),
        PDF_DIR / f"{tag}_historychart.pdf"
    )
    pit_bytes = download_pdf(
        pdf_url(year, rnd, circuit, "pitstopsummary"),
        PDF_DIR / f"{tag}_pitstops.pdf"
    )

    if not hist_bytes:
        print(f"  → SKIP")
        return None

    laps_df = parse_history_chart(hist_bytes)
    pits_df = parse_pitstops(pit_bytes) if pit_bytes else pd.DataFrame(
        columns=["car_no", "pit_lap", "stop_num", "duration_sec"])

    if laps_df.empty:
        print(f"  → SKIP (parse empty)")
        return None

    # Merge pit flags from pit stop summary
    pit_set = set(zip(pits_df["car_no"], pits_df["pit_lap"])) if not pits_df.empty else set()
    laps_df["is_pit_lap"] = laps_df["is_pit_lap"] | laps_df.apply(
        lambda r: (r["car_no"], r["lap"]) in pit_set, axis=1)

    sc_laps    = infer_sc_laps(laps_df)
    sc_windows = find_windows(sc_laps)

    # Clean filter
    clean = laps_df[
        ~laps_df["is_pit_lap"] &
        ~laps_df["lap"].isin(sc_laps) &
        (laps_df["lap_time_sec"] > 60) &
        (laps_df["lap"] > 1)
    ].copy()

    # Per-car outlier filter
    car_meds = clean.groupby("car_no")["lap_time_sec"].median()
    clean    = clean[clean.apply(
        lambda r: r["lap_time_sec"] <= car_meds.get(r["car_no"], 999) * 1.07, axis=1)]

    total_laps = int(laps_df["lap"].max())

    out  = OUTPUT_DIR / f"{tag}.csv"
    meta = OUTPUT_DIR / f"{tag}_meta.json"
    clean.to_csv(out, index=False)
    if not pits_df.empty:
        pits_df.to_csv(OUTPUT_DIR / f"{tag}_pits.csv", index=False)

    valid_laps = laps_df[~laps_df["is_pit_lap"] & (laps_df["lap"] > 1)]
    metadata = {
        "year": year, "round": rnd,
        "circuit": circuit, "circuit_name": name,
        "total_laps": total_laps,
        "total_cars": int(laps_df["car_no"].nunique()),
        "clean_laps": len(clean),
        "race_median_sec": round(valid_laps["lap_time_sec"].median(), 3) if not valid_laps.empty else 0,
        "sc_laps": sc_laps,
        "sc_windows": sc_windows,
        "sc_count": len(sc_windows),
        "pit_stops": len(pits_df),
        "output_path": str(out),
    }
    with open(meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  → {len(clean):,} laps | {len(sc_windows)} SC | {total_laps} laps | "
          f"{int(laps_df['car_no'].nunique())} cars")
    return metadata


def fetch_all():
    print(f"PITWALL — Phase 1: FIA PDF Pipeline")
    print(f"Fetching {len(RACES)} F2 Feature Races\n")

    all_meta = []
    for year, rnd, circuit, name in RACES:
        result = process_race(year, rnd, circuit, name)
        if result:
            all_meta.append(result)

    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE")
    print(f"  Races fetched:  {len(all_meta)}")
    print(f"  SC events:      {sum(r['sc_count'] for r in all_meta)}")
    print(f"  Clean laps:     {sum(r['clean_laps'] for r in all_meta):,}")

    idx = OUTPUT_DIR / "index.json"
    with open(idx, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"  Index: {idx}")

    cstats = defaultdict(lambda: {"races": 0, "sc": 0})
    for r in all_meta:
        cstats[r["circuit_name"]]["races"] += 1
        cstats[r["circuit_name"]]["sc"]    += r["sc_count"]

    print(f"\nSC FREQUENCY BY CIRCUIT:")
    print(f"  {'Circuit':<22} {'SC/race':>8}  {'SC total':>9}  Races")
    print(f"  {'-'*48}")
    for name, s in sorted(cstats.items(),
                           key=lambda x: x[1]["sc"] / max(x[1]["races"], 1),
                           reverse=True):
        print(f"  {name:<22} {s['sc']/s['races']:>8.2f}  {s['sc']:>9}  {s['races']}")

    return all_meta


if __name__ == "__main__":
    fetch_all()
