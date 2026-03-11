"""
PITWALL — Phase 2: Tire Degradation Model (v2)
===============================================
Fix: Stint reconstruction uses pit stop CSV directly (lap gaps in car's
lap sequence), not is_pit_lap flag (which is absent from clean CSVs).
Also applies fuel correction: F2 burns ~0.08 s/lap improvement from fuel,
so corrected_deg = raw_slope + 0.080.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats

ROOT       = Path(__file__).parent.parent
RACES_DIR  = ROOT / "data" / "races"
OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

# F2 fuel burn improvement rate (sec/lap).
# ~2.5 kg/lap fuel consumption × 0.030-0.035 s/kg = 0.075-0.088 s/lap.
# We use 0.080 as the central estimate.
FUEL_CORRECTION = 0.080


def load_race(meta):
    csv = Path(meta["output_path"])
    if not csv.exists():
        return None, None
    laps = pd.read_csv(csv)
    laps["lap"]    = laps["lap"].astype(int)
    laps["car_no"] = laps["car_no"].astype(int)
    pits_path = csv.parent / (csv.stem + "_pits.csv")
    pits = pd.read_csv(pits_path) if pits_path.exists() else pd.DataFrame(
        columns=["car_no", "pit_lap", "stop_num", "duration_sec"])
    return laps, pits


def reconstruct_stints(laps_df, pits_df, sc_laps):
    """
    Reconstruct stints by detecting lap gaps in each car's sequence.
    A car that pits on lap N will be missing lap N from the clean CSV
    (pit laps are excluded during fetch). So a jump in lap number
    (current_lap > prev_lap + 1) signals a pit stop occurred.
    We also use the pit stop CSV to confirm.
    """
    sc_set  = set(sc_laps)
    # Build set of known pit laps per car from pit stop summary
    pit_set = {}
    if not pits_df.empty:
        for car_no, grp in pits_df.groupby("car_no"):
            pit_set[int(car_no)] = set(grp["pit_lap"].astype(int).tolist())

    result = []

    for car_no, car_laps in laps_df.groupby("car_no"):
        car_laps     = car_laps.sort_values("lap").reset_index(drop=True)
        car_pit_laps = pit_set.get(int(car_no), set())
        lap_list     = car_laps["lap"].tolist()

        stint_num    = 1
        lap_in_stint = 0

        for idx, row in car_laps.iterrows():
            lap = int(row["lap"])

            # New stint if: lap gap detected (pit lap was excluded from CSV)
            # OR previous lap was a known pit lap
            if idx > 0:
                prev_lap = lap_list[idx - 1]
                gap      = lap - prev_lap
                # Gap > 1 means one or more laps missing → pit stop
                # Also check if prev_lap itself was a pit lap
                if gap > 1 or prev_lap in car_pit_laps:
                    stint_num    += 1
                    lap_in_stint  = 0

            lap_in_stint += 1

            result.append({
                "car_no":        int(car_no),
                "lap":           lap,
                "lap_time_sec":  row["lap_time_sec"],
                "gap_to_leader": row["gap_to_leader"],
                "is_sc_lap":     lap in sc_set,
                "stint_num":     stint_num,
                "lap_in_stint":  lap_in_stint,
            })

    return pd.DataFrame(result) if result else pd.DataFrame()


def fit_stint_deg(stint_df, min_laps=4):
    """
    Linear regression on lap_time ~ lap_in_stint.
    Skip first 2 laps (out-lap noise). Apply ±3% outlier filter.
    Returns slope (raw, before fuel correction) or None.
    """
    df = stint_df[stint_df["lap_in_stint"] > 2].copy()
    if len(df) < min_laps:
        return None

    med = df["lap_time_sec"].median()
    df  = df[(df["lap_time_sec"] >= med * 0.97) & (df["lap_time_sec"] <= med * 1.03)]
    if len(df) < min_laps:
        return None

    x = df["lap_in_stint"].values.astype(float)
    y = df["lap_time_sec"].values
    slope, intercept, r, p, se = stats.linregress(x, y)

    if abs(slope) > 1.0:
        return None

    return {
        "slope_raw":   round(slope, 5),
        "slope_corr":  round(slope + FUEL_CORRECTION, 5),  # fuel-corrected
        "intercept":   round(intercept, 3),
        "r2":          round(r**2, 4),
        "n_laps":      len(df),
        "stint_laps":  int(stint_df["lap_in_stint"].max()),
        "mean_pace":   round(df["lap_time_sec"].mean(), 3),
    }


def analyse_race(meta):
    laps, pits = load_race(meta)
    if laps is None or laps.empty:
        return []

    sc_laps = meta.get("sc_laps", [])
    stints  = reconstruct_stints(laps, pits, sc_laps)
    if stints.empty:
        return []

    clean = stints[~stints["is_sc_lap"]].copy()

    records = []
    for (car_no, stint_num), grp in clean.groupby(["car_no", "stint_num"]):
        grp = grp.sort_values("lap_in_stint")
        fit = fit_stint_deg(grp)
        if fit is None:
            continue
        records.append({
            "year":        meta["year"],
            "circuit":     meta["circuit_name"],
            "car_no":      int(car_no),
            "stint_num":   int(stint_num),
            "race_median": meta["race_median_sec"],
            **fit,
        })

    return records


def build_circuit_profiles(all_records):
    df = pd.DataFrame(all_records)
    if df.empty:
        return {}
    profiles = {}
    for circuit, grp in df.groupby("circuit"):
        raw   = grp["slope_raw"].values
        corr  = grp["slope_corr"].values
        profiles[circuit] = {
            "deg_mean":          round(float(np.mean(corr)),            5),
            "deg_median":        round(float(np.median(corr)),          5),
            "deg_p25":           round(float(np.percentile(corr, 25)),  5),
            "deg_p75":           round(float(np.percentile(corr, 75)),  5),
            "deg_std":           round(float(np.std(corr)),             5),
            "raw_mean":          round(float(np.mean(raw)),             5),
            "n_stints":          int(len(corr)),
            "mean_stint_laps":   round(float(grp["stint_laps"].mean()), 1),
            "mean_pace_sec":     round(float(grp["mean_pace"].mean()),  3),
        }
    return profiles


def run():
    idx_path = RACES_DIR / "index.json"
    if not idx_path.exists():
        print("ERROR: index.json not found. Run fetch.py first.")
        return

    with open(idx_path) as f:
        all_meta = json.load(f)

    print("PITWALL — Phase 2: Tire Degradation Model (v2)")
    print(f"Fuel correction: +{FUEL_CORRECTION:.3f} s/lap")
    print(f"Processing {len(all_meta)} races...\n")

    all_records = []
    for meta in all_meta:
        records = analyse_race(meta)
        all_records.extend(records)
        n = len(records)
        if n:
            raw_slopes  = [r["slope_raw"]  for r in records]
            corr_slopes = [r["slope_corr"] for r in records]
            avg_stint   = np.mean([r["stint_laps"] for r in records])
            print(f"  {meta['year']} {meta['circuit_name']:<22} "
                  f"{n:>3} stints  "
                  f"raw={np.mean(raw_slopes):+.4f}  "
                  f"corr={np.mean(corr_slopes):+.4f} s/lap  "
                  f"avg_stint={avg_stint:.1f} laps")
        else:
            print(f"  {meta['year']} {meta['circuit_name']:<22}  SKIP")

    print(f"\n{'='*70}")
    print(f"Total stints: {len(all_records)}")

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_DIR / "stint_deg_raw.csv", index=False)

    profiles = build_circuit_profiles(all_records)

    print(f"\nCIRCUIT DEG PROFILES (fuel-corrected, sec/lap):")
    print(f"  {'Circuit':<22} {'mean':>8}  {'median':>8}  {'p25':>8}  {'p75':>8}  "
          f"{'stint':>6}  {'stints':>7}")
    print(f"  {'-'*72}")
    for circuit, p in sorted(profiles.items(),
                              key=lambda x: x[1]["deg_mean"], reverse=True):
        print(f"  {circuit:<22} {p['deg_mean']:>+8.4f}  "
              f"{p['deg_median']:>+8.4f}  "
              f"{p['deg_p25']:>+8.4f}  {p['deg_p75']:>+8.4f}  "
              f"{p['mean_stint_laps']:>6.1f}  {p['n_stints']:>7}")

    output = {
        "fuel_correction_sec_per_lap": FUEL_CORRECTION,
        "circuit_profiles": profiles,
        "global": {
            "deg_mean":   round(float(np.mean([r["slope_corr"] for r in all_records])), 5),
            "deg_median": round(float(np.median([r["slope_corr"] for r in all_records])), 5),
            "n_stints":   len(all_records),
            "n_races":    len(all_meta),
        }
    }
    with open(OUTPUT_DIR / "deg_rates.json", "w") as f:
        json.dump(output, f, indent=2)

    g = output["global"]
    print(f"\nGLOBAL (corrected): mean={g['deg_mean']:+.4f}  "
          f"median={g['deg_median']:+.4f} s/lap")
    print(f"\nSaved: models/stint_deg_raw.csv  |  models/deg_rates.json")
    return output


if __name__ == "__main__":
    run()