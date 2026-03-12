"""
PITWALL — Phase 5: Backtest
============================
Runs the decision engine against every race in the dataset.

For each race, for each car:
  - Reconstructs stints from pit stop data
  - At every lap from lap 5 onward, queries the engine
  - Records what the engine said vs what actually happened
  - Flags "key moments": laps where engine called UNDERCUT/BOX
    and the car either pitted that lap or within 2 laps (correct call),
    or didn't pit for 5+ more laps (missed opportunity)

Output:
  validation/backtest_summary.json    — aggregate stats per race
  validation/backtest_traces/         — lap-by-lap trace per race
  validation/key_moments.json         — top cases for the write-up
"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent
RACES_DIR  = ROOT / "data" / "races"
VAL_DIR    = ROOT / "validation"
TRACES_DIR = VAL_DIR / "backtest_traces"
VAL_DIR.mkdir(exist_ok=True)
TRACES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "models"))
from engine import decide, OVERTAKE_FACTOR, DEFAULT_OVERTAKE_FACTOR

PIT_DELTAS = {
    "Bahrain":       22.0,
    "Jeddah":        20.0,
    "Melbourne":     23.0,
    "Imola":         22.0,
    "Monaco":        28.0,
    "Montreal":      21.0,
    "Barcelona":     22.0,
    "Red_Bull_Ring": 21.0,
    "Silverstone":   22.0,
    "Budapest":      21.0,
    "Spa":           23.0,
    "Zandvoort":     21.0,
    "Monza":         20.0,
    "Baku":          21.0,
    "Qatar":         22.0,
    "Abu_Dhabi":     23.0,
}
DEFAULT_PIT_DELTA = 22.0


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


def reconstruct_stints(laps_df, pits_df):
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
            if idx > 0:
                prev_lap = lap_list[idx - 1]
                # Primary: pit CSV says prev lap was a pit lap
                # Secondary: gap in lap sequence (pit lap filtered from clean CSV)
                pit_boundary = (prev_lap in car_pit_laps) or (lap - prev_lap) > 1
                # Also split if current lap immediately follows a known pit lap
                # (handles case where pit lap itself is present in clean CSV)
                pit_boundary = pit_boundary or (lap - 1) in car_pit_laps
                if pit_boundary:
                    stint_num    += 1
                    lap_in_stint  = 0
            lap_in_stint += 1
            result.append({
                "car_no":        int(car_no),
                "lap":           lap,
                "lap_time_sec":  row["lap_time_sec"],
                "gap_to_leader": row["gap_to_leader"],
                "stint_num":     stint_num,
                "lap_in_stint":  lap_in_stint,
            })
    return pd.DataFrame(result) if result else pd.DataFrame()


def get_gaps(stints_df, car_no, lap):
    lap_data = stints_df[stints_df["lap"] == lap].sort_values("gap_to_leader")
    if lap_data.empty:
        return None, None
    car_row = lap_data[lap_data["car_no"] == car_no]
    if car_row.empty:
        return None, None

    positions = lap_data.reset_index(drop=True)
    pos       = positions[positions["car_no"] == car_no].index[0]
    car_gap   = float(positions.iloc[pos]["gap_to_leader"])

    gap_ahead  = None
    gap_behind = None
    if pos > 0:
        gap_ahead  = round(car_gap - float(positions.iloc[pos-1]["gap_to_leader"]), 3)
    if pos < len(positions) - 1:
        gap_behind = round(float(positions.iloc[pos+1]["gap_to_leader"]) - car_gap, 3)

    return gap_ahead, gap_behind


def backtest_race(meta):
    laps_df, pits_df = load_race(meta)
    if laps_df is None or laps_df.empty:
        return [], []

    circuit    = meta["circuit_name"]
    total_laps = meta["total_laps"]
    sc_laps    = set(meta.get("sc_laps", []))
    pit_delta  = PIT_DELTAS.get(circuit, DEFAULT_PIT_DELTA)

    stints_df  = reconstruct_stints(laps_df, pits_df)
    if stints_df.empty:
        return [], []

    actual_pits = {}
    if not pits_df.empty:
        for car_no, grp in pits_df.groupby("car_no"):
            actual_pits[int(car_no)] = sorted(grp["pit_lap"].astype(int).tolist())

    records     = []
    key_moments = []

    for car_no, car_stints in stints_df.groupby("car_no"):
        car_stints   = car_stints.sort_values("lap").reset_index(drop=True)
        car_pit_laps = actual_pits.get(int(car_no), [])

        for _, row in car_stints.iterrows():
            lap          = int(row["lap"])
            lap_in_stint = int(row["lap_in_stint"])

            if lap < 5 or lap in sc_laps or lap_in_stint < 3:
                continue

            gap_ahead, gap_behind = get_gaps(stints_df, car_no, lap)

            try:
                result = decide(circuit, lap, total_laps, lap_in_stint,
                                gap_ahead, gap_behind, pit_delta)
            except Exception:
                continue

            rec              = result["recommendation"]
            actually_pitted  = any(abs(lap - p) <= 2 for p in car_pit_laps)
            pitted_soon      = any(0 < (p - lap) <= 5 for p in car_pit_laps)
            stayed_long      = (rec in ("UNDERCUT", "BOX") and
                                not any((p - lap) <= 5 for p in car_pit_laps
                                        if p > lap))

            record = {
                "year":            meta["year"],
                "circuit":         circuit,
                "car_no":          int(car_no),
                "lap":             lap,
                "total_laps":      total_laps,
                "lap_in_stint":    lap_in_stint,
                "gap_ahead":       gap_ahead,
                "recommendation":  rec,
                "confidence":      result["confidence"],
                "raw_gain":        result["raw_gain"],
                "sc_adj_gain":     result["sc_adj_gain"],
                "eff_gain":        result["eff_gain"],
                "sc_prob":         result["sc_prob"],
                "deg_loss":        result["deg_loss_per_lap"],
                "actually_pitted": actually_pitted,
                "pitted_soon":     pitted_soon,
                "stayed_long":     stayed_long,
            }
            records.append(record)

            is_key = (
                rec in ("UNDERCUT", "BOX") and
                result["confidence"] >= 0.60 and
                lap_in_stint >= 8 and
                (actually_pitted or stayed_long)
            )
            if is_key:
                key_moments.append({
                    **record,
                    "reasoning":      result["reasoning"],
                    "deg_rate":       result["deg_rate"],
                    "action_taken":   "PITTED" if actually_pitted else "STAYED",
                    "engine_correct": actually_pitted,
                })

    return records, key_moments


def run():
    idx_path = RACES_DIR / "index.json"
    if not idx_path.exists():
        print("ERROR: Run fetch.py first.")
        return

    with open(idx_path) as f:
        all_meta = json.load(f)

    print("PITWALL — Phase 5: Backtest")
    print(f"Running engine on {len(all_meta)} races...\n")

    all_records     = []
    all_key_moments = []
    race_summaries  = []

    for meta in all_meta:
        circuit  = meta["circuit_name"]
        records, key_moments = backtest_race(meta)

        if not records:
            print(f"  {meta['year']} {circuit:<22}  SKIP")
            continue

        df       = pd.DataFrame(records)
        pit_calls = df[df["recommendation"].isin(["UNDERCUT", "BOX"])]
        agreement = (pit_calls["actually_pitted"].sum() / len(pit_calls)
                     if len(pit_calls) > 0 else 0)
        missed    = df[df["stayed_long"]].shape[0]

        summary = {
            "year":             meta["year"],
            "circuit":          circuit,
            "total_laps":       meta["total_laps"],
            "n_cars":           int(df["car_no"].nunique()),
            "n_laps_assessed":  len(df),
            "undercut_calls":   int(len(df[df["recommendation"]=="UNDERCUT"])),
            "stay_calls":       int(len(df[df["recommendation"]=="STAY"])),
            "box_calls":        int(len(df[df["recommendation"]=="BOX"])),
            "pit_agreement":    round(float(agreement), 3),
            "missed_calls":     int(missed),
            "key_moments":      len(key_moments),
        }
        race_summaries.append(summary)
        all_records.extend(records)
        all_key_moments.extend(key_moments)

        trace_path = TRACES_DIR / f"{meta['year']}_{circuit}.csv"
        df.to_csv(trace_path, index=False)

        print(f"  {meta['year']} {circuit:<22}  "
              f"{len(df):>4} laps  "
              f"UC:{summary['undercut_calls']:>3}  "
              f"agree:{agreement:.0%}  "
              f"missed:{missed:>2}  "
              f"key:{len(key_moments):>2}")

    print(f"\n{'='*65}")
    print(f"BACKTEST COMPLETE")
    total_laps   = sum(s["n_laps_assessed"] for s in race_summaries)
    total_uc     = sum(s["undercut_calls"]   for s in race_summaries)
    avg_agree    = np.mean([s["pit_agreement"] for s in race_summaries])
    total_missed = sum(s["missed_calls"]     for s in race_summaries)
    print(f"  Laps assessed:  {total_laps:,}")
    print(f"  Undercut calls: {total_uc:,}")
    print(f"  Avg agreement:  {avg_agree:.1%}")
    print(f"  Missed calls:   {total_missed}")
    print(f"  Key moments:    {len(all_key_moments)}")

    correct_calls = sorted([m for m in all_key_moments if m["engine_correct"]],
                            key=lambda x: x["eff_gain"], reverse=True)
    missed_calls  = sorted([m for m in all_key_moments if not m["engine_correct"]],
                            key=lambda x: x["eff_gain"], reverse=True)

    print(f"\nTOP CORRECT CALLS (engine said pit → car pitted):")
    print(f"  {'Year':<5} {'Circuit':<16} {'Car':>4} {'Lap':>4}  "
          f"{'Stint':>5}  {'EffGain':>8}  {'SC%':>5}  {'Conf':>5}")
    print(f"  {'-'*62}")
    for m in correct_calls[:10]:
        print(f"  {m['year']:<5} {m['circuit']:<16} {m['car_no']:>4} "
              f"{m['lap']:>4}  {m['lap_in_stint']:>5}  "
              f"{m['eff_gain']:>+8.1f}  "
              f"{m['sc_prob']:>5.0%}  {m['confidence']:.0%}")

    print(f"\nTOP MISSED CALLS (engine said pit → car stayed 5+ laps):")
    print(f"  {'Year':<5} {'Circuit':<16} {'Car':>4} {'Lap':>4}  "
          f"{'Stint':>5}  {'EffGain':>8}  {'SC%':>5}  {'Conf':>5}")
    print(f"  {'-'*62}")
    for m in missed_calls[:10]:
        print(f"  {m['year']:<5} {m['circuit']:<16} {m['car_no']:>4} "
              f"{m['lap']:>4}  {m['lap_in_stint']:>5}  "
              f"{m['eff_gain']:>+8.1f}  "
              f"{m['sc_prob']:>5.0%}  {m['confidence']:.0%}")

    with open(VAL_DIR / "backtest_summary.json", "w") as f:
        json.dump(race_summaries, f, indent=2)
    with open(VAL_DIR / "key_moments.json", "w") as f:
        json.dump({"correct_calls": correct_calls[:20],
                   "missed_calls":  missed_calls[:20]}, f, indent=2)
    pd.DataFrame(all_records).to_csv(VAL_DIR / "backtest_all.csv", index=False)

    print(f"\nSaved:")
    print(f"  validation/backtest_summary.json")
    print(f"  validation/backtest_all.csv")
    print(f"  validation/key_moments.json")
    print(f"  validation/backtest_traces/  ({len(race_summaries)} files)")

    return race_summaries, all_key_moments


if __name__ == "__main__":
    run()