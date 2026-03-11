"""
PITWALL — Phase 3: Safety Car Probability Model (v2)
=====================================================
Fixes vs v1:
  - Budapest zero fallback: circuits with sc_per_race=0 use
    global_sc_per_race × p_sc_race as effective rate
  - SC table now correctly differentiates circuits with same
    p_sc_race but different sc_per_race (Monaco vs Bahrain)
  - remaining_mass applied correctly to effective_sc_per_race
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).parent.parent
RACES_DIR  = ROOT / "data" / "races"
OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

GLOBAL_SC_PER_RACE = 36 / 20  # 1.8 — from dataset


def load_index():
    idx = RACES_DIR / "index.json"
    if not idx.exists():
        raise FileNotFoundError("data/races/index.json not found.")
    with open(idx) as f:
        return json.load(f)


def build_circuit_rates(all_meta):
    stats = defaultdict(lambda: {"races": 0, "sc_races": 0, "sc_events": 0,
                                  "sc_laps": [], "total_laps": []})
    for m in all_meta:
        c = m["circuit_name"]
        stats[c]["races"]      += 1
        stats[c]["total_laps"].append(m["total_laps"])
        if m["sc_count"] > 0:
            stats[c]["sc_races"] += 1
        stats[c]["sc_events"] += m["sc_count"]
        stats[c]["sc_laps"].extend(
            [w["start"] for w in m.get("sc_windows", [])]
        )

    rates = {}
    for c, s in stats.items():
        n    = s["races"]
        sc_n = s["sc_races"]
        sc_per_race = s["sc_events"] / n

        # For circuits with zero observed SC, use global rate scaled by
        # Laplace-smoothed probability as effective rate
        effective_sc_per_race = sc_per_race if sc_per_race > 0 \
            else GLOBAL_SC_PER_RACE * ((sc_n + 1) / (n + 2))

        rates[c] = {
            "races":                  n,
            "sc_races":               sc_n,
            "sc_events":              s["sc_events"],
            "p_sc_race":              round((sc_n + 1) / (n + 2), 4),
            "p_sc_race_raw":          round(sc_n / n, 4),
            "sc_per_race":            round(sc_per_race, 3),
            "effective_sc_per_race":  round(effective_sc_per_race, 3),
            "mean_total_laps":        round(np.mean(s["total_laps"]), 1),
            "sc_lap_numbers":         sorted(s["sc_laps"]),
        }
    return rates


def build_lap_distribution(all_meta):
    fractions = []
    for m in all_meta:
        total = m["total_laps"]
        if total == 0:
            continue
        for w in m.get("sc_windows", []):
            fractions.append(w["start"] / total)

    if not fractions:
        return [0.1] * 10, list(np.linspace(0.05, 0.95, 10))

    fractions = np.array(fractions)
    bins      = np.linspace(0, 1, 11)
    counts, _ = np.histogram(fractions, bins=bins)
    counts    = counts + 1   # Laplace smooth
    probs     = counts / counts.sum()
    centers   = [(bins[i] + bins[i+1]) / 2 for i in range(10)]
    return probs.tolist(), centers


def p_sc_remaining(circuit_rate, lap_dist_probs, lap_dist_centers,
                    current_lap, total_laps):
    """
    P(at least one SC in laps [current_lap+1 .. total_laps]).

    Uses Poisson approximation:
      lambda = effective_sc_per_race × remaining_mass
      P(>=1) = 1 - e^(-lambda)

    remaining_mass = fraction of lap distribution in the remaining race window.
    """
    laps_remaining = max(0, total_laps - current_lap)
    if laps_remaining == 0:
        return 0.0

    f_current = current_lap / total_laps

    # Sum distribution mass after current race fraction
    remaining_mass = sum(
        p for p, c in zip(lap_dist_probs, lap_dist_centers)
        if c > f_current
    )
    remaining_mass = float(np.clip(remaining_mass, 0.0, 1.0))

    lam = circuit_rate["effective_sc_per_race"] * remaining_mass
    return round(float(1.0 - np.exp(-lam)), 4)


def build_sc_tables(circuit_rates, lap_dist_probs, lap_dist_centers):
    tables = {}
    fractions = [round(f, 1) for f in np.arange(0.1, 1.0, 0.1)]

    for circuit, rate in circuit_rates.items():
        total = int(rate["mean_total_laps"])
        row   = {}
        for f in fractions:
            lap = int(f * total)
            p   = p_sc_remaining(rate, lap_dist_probs, lap_dist_centers, lap, total)
            row[str(f)] = p
        tables[circuit] = row

    return tables


def run():
    all_meta = load_index()

    print("PITWALL — Phase 3: Safety Car Probability Model (v2)")
    print(f"Input: {len(all_meta)} races\n")

    circuit_rates                    = build_circuit_rates(all_meta)
    lap_dist_probs, lap_dist_centers = build_lap_distribution(all_meta)

    print("CIRCUIT SC RATES:")
    print(f"  {'Circuit':<22} {'Races':>6}  {'SC/race':>8}  "
          f"{'eff SC/race':>12}  {'P(SC≥1)':>8}")
    print(f"  {'-'*64}")
    for c, r in sorted(circuit_rates.items(),
                        key=lambda x: x[1]["effective_sc_per_race"], reverse=True):
        print(f"  {c:<22} {r['races']:>6}  {r['sc_per_race']:>8.2f}  "
              f"{r['effective_sc_per_race']:>12.3f}  {r['p_sc_race']:>8.3f}")

    print(f"\nSC LAP DISTRIBUTION (fraction of race distance):")
    print(f"  {'Bucket':>8}  {'Prob':>6}  Bar")
    print(f"  {'-'*45}")
    for p, c in zip(lap_dist_probs, lap_dist_centers):
        bar = "█" * int(p * 80)
        print(f"  {c:>8.2f}  {p:>6.3f}  {bar}")

    sc_tables = build_sc_tables(circuit_rates, lap_dist_probs, lap_dist_centers)

    print(f"\nP(SC IN REMAINING RACE) by circuit and lap:")
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sample    = ["Bahrain", "Monza", "Spa", "Monaco", "Budapest"]
    print(f"  {'Lap%':<6}", end="")
    for c in sample:
        if c in sc_tables:
            print(f"  {c:>10}", end="")
    print()
    print(f"  {'-'*60}")
    for f in fractions:
        print(f"  {f:<6.0%}", end="")
        for c in sample:
            if c in sc_tables:
                p = sc_tables[c].get(str(f), 0)
                print(f"  {p:>10.3f}", end="")
        print()

    output = {
        "circuit_rates":    circuit_rates,
        "lap_distribution": {
            "probs":       lap_dist_probs,
            "centers":     lap_dist_centers,
            "description": "P(SC starts in this 10% bucket of race distance)",
        },
        "sc_tables": sc_tables,
        "global": {
            "total_races":        len(all_meta),
            "total_sc_events":    sum(m["sc_count"] for m in all_meta),
            "global_sc_per_race": round(GLOBAL_SC_PER_RACE, 3),
            "p_sc_any_race":      round(
                sum(1 for m in all_meta if m["sc_count"] > 0) / len(all_meta), 3),
        }
    }

    def to_python(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return obj

    with open(OUTPUT_DIR / "sc_model.json", "w") as f:
        json.dump(output, f, indent=2, default=to_python)

    g = output["global"]
    print(f"\nGLOBAL: {g['total_sc_events']} SC events / {g['total_races']} races  "
          f"= {g['global_sc_per_race']:.2f}/race  |  "
          f"P(SC | any race) = {g['p_sc_any_race']:.1%}")
    print(f"\nSaved: models/sc_model.json")
    return output


if __name__ == "__main__":
    run()