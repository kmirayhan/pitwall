"""
PITWALL — Phase 4: Pit Decision Engine (v3)
===========================================
Decision logic:
  1. UNDERCUT — adj_gain > 0 (SC-adjusted), effective gain > 2s after
     overtake discount, NOT sc_wait
  2. STAY (SC wait) — SC prob > 40% AND raw_gain < 8s (marginal pit value)
  3. BOX — adj_gain <= 0 but tires critically slow (deg_loss > 2.5s this lap)
  4. STAY — everything else

Key formulas:
  raw_gain      = deg_rate × lap_in_stint × laps_remaining - pit_delta
  sc_adj_gain   = raw_gain + pit_delta × sc_prob - SC_DELTA × sc_prob
  eff_gain      = sc_adj_gain × overtake_factor  (when car ahead present)
  deg_loss_lap  = deg_rate × (lap_in_stint - 1)  (regression definition)
"""

import json
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

DEFAULT_PIT_DELTA    = 22.0
DEFAULT_SC_PIT_DELTA =  5.0
MIN_LAPS_TO_GO       =  3
SC_WAIT_THRESHOLD    =  0.40   # P(SC) above this → consider waiting
SC_WAIT_MAX_GAIN     =  8.0    # Only wait if raw gain < this (marginal pit)
BOX_DEG_THRESHOLD    =  2.5    # s/lap slower than stint lap 1 → BOX if gain negative
EFF_GAIN_MIN         =  2.0    # Minimum effective gain to call UNDERCUT

# Overtake difficulty: 1.0 = easy overtake, 0.0 = impossible
# When car_ahead present, effective gain = sc_adj_gain × overtake_factor
OVERTAKE_FACTOR = {
    "Monaco":        0.05,
    "Budapest":      0.35,
    "Barcelona":     0.45,
    "Imola":         0.50,
    "Zandvoort":     0.50,
    "Red_Bull_Ring": 0.55,
    "Silverstone":   0.60,
    "Abu_Dhabi":     0.60,
    "Spa":           0.65,
    "Melbourne":     0.65,
    "Bahrain":       0.70,
    "Qatar":         0.70,
    "Monza":         0.75,
    "Jeddah":        0.80,
    "Baku":          0.85,
    "Montreal":      0.80,
}
DEFAULT_OVERTAKE_FACTOR = 0.60


def load_models():
    deg_path = MODELS_DIR / "deg_rates.json"
    sc_path  = MODELS_DIR / "sc_model.json"
    if not deg_path.exists() or not sc_path.exists():
        raise FileNotFoundError("Run deg.py and sc.py first.")
    with open(deg_path) as f:
        deg_model = json.load(f)
    with open(sc_path) as f:
        sc_model  = json.load(f)
    return deg_model, sc_model


def get_deg_rate(deg_model, circuit):
    p = deg_model["circuit_profiles"]
    return p[circuit]["deg_mean"] if circuit in p else deg_model["global"]["deg_median"]


def get_sc_prob(sc_model, circuit, current_lap, total_laps):
    tables = sc_model.get("sc_tables", {})
    rates  = sc_model.get("circuit_rates", {})
    if circuit not in rates:
        frac = (total_laps - current_lap) / total_laps
        lam  = sc_model["global"]["global_sc_per_race"] * frac
        return round(float(1.0 - np.exp(-lam)), 4)
    lap_frac = max(0.1, min(0.9, round(current_lap / total_laps, 1)))
    table    = tables.get(circuit, {})
    keys     = sorted(float(k) for k in table.keys())
    nearest  = min(keys, key=lambda k: abs(k - lap_frac))
    return table.get(str(nearest), table.get(str(round(nearest, 1)), 0.0))


def compute_gains(deg_rate, lap_in_stint, laps_remaining, sc_prob, pit_delta):
    """
    raw_gain    = pace advantage from fresh tires over remaining laps - cost
    sc_adj_gain = raw_gain adjusted for SC probability reducing effective delta
    """
    pace_gain   = deg_rate * lap_in_stint * laps_remaining
    raw_gain    = pace_gain - pit_delta
    sc_delta    = pit_delta * (1 - sc_prob) + DEFAULT_SC_PIT_DELTA * sc_prob
    sc_adj_gain = pace_gain - sc_delta
    return round(raw_gain, 3), round(sc_adj_gain, 3)


def effective_gain(sc_adj_gain, gap_ahead, overtake_factor):
    """
    When car is ahead, pace gain is only useful if you can overtake.
    Effective gain = sc_adj_gain × overtake_factor.
    When leading (gap_ahead=None), full gain applies.
    """
    if gap_ahead is None:
        return sc_adj_gain   # leading — pace gain is pure lap time
    return round(sc_adj_gain * overtake_factor, 3)


def decide(circuit, current_lap, total_laps, lap_in_stint,
           gap_ahead=None, gap_behind=None,
           pit_delta=DEFAULT_PIT_DELTA):

    laps_remaining = total_laps - current_lap

    if laps_remaining < MIN_LAPS_TO_GO:
        return _result("STAY", 0.95, circuit, current_lap, total_laps,
                        lap_in_stint, gap_ahead, gap_behind, pit_delta,
                        0, 0, 0, 0, 0, 0,
                        f"Only {laps_remaining} laps remaining. Stay out.")

    deg_model, sc_model = load_models()

    deg_rate       = get_deg_rate(deg_model, circuit)
    sc_prob        = get_sc_prob(sc_model, circuit, current_lap, total_laps)
    deg_loss       = round(deg_rate * max(0, lap_in_stint - 1), 3)
    raw_gain, adj  = compute_gains(deg_rate, lap_in_stint, laps_remaining,
                                    sc_prob, pit_delta)
    ot_factor      = OVERTAKE_FACTOR.get(circuit, DEFAULT_OVERTAKE_FACTOR)
    eff            = effective_gain(adj, gap_ahead, ot_factor)

    # ── FLAGS ─────────────────────────────────────────────────
    sc_wait      = sc_prob >= SC_WAIT_THRESHOLD and raw_gain < SC_WAIT_MAX_GAIN
    undercut_ok  = eff >= EFF_GAIN_MIN and not sc_wait
    tire_critical = deg_loss >= BOX_DEG_THRESHOLD

    # ── DECISION ──────────────────────────────────────────────
    if undercut_ok:
        rec        = "UNDERCUT"
        confidence = min(0.92, 0.55 + eff * 0.005)
        if gap_ahead is not None and gap_ahead < pit_delta:
            reasoning = (f"Effective gain {eff:+.1f}s. Gap {gap_ahead:.1f}s < "
                          f"pit delta {pit_delta:.0f}s — emerge ahead or close.")
        else:
            reasoning = (f"Effective gain {eff:+.1f}s (SC-adj {adj:+.1f}s, "
                          f"overtake factor {ot_factor:.2f}). Pace delta justifies stop.")

    elif sc_wait:
        rec        = "STAY"
        confidence = min(0.84, 0.50 + sc_prob * 0.45)
        reasoning  = (f"SC probability {sc_prob:.0%} — free pit worth waiting for. "
                       f"Raw gain {raw_gain:+.1f}s too marginal to pay full "
                       f"{pit_delta:.0f}s delta now.")

    elif tire_critical:
        rec        = "BOX"
        confidence = min(0.90, 0.65 + (deg_loss - BOX_DEG_THRESHOLD) * 0.05)
        reasoning  = (f"Current lap {deg_loss:.2f}s slower than stint lap 1. "
                       f"Tires below performance threshold. Box.")

    else:
        rec        = "STAY"
        confidence = min(0.80, 0.55 + max(0, -eff) * 0.005)
        reasoning  = (f"Effective gain {eff:+.1f}s — pitting costs more than gains. "
                       f"Tires {deg_loss:.2f}s off lap 1 pace. Stay out.")

    return _result(rec, confidence, circuit, current_lap, total_laps,
                    lap_in_stint, gap_ahead, gap_behind, pit_delta,
                    deg_rate, deg_loss, sc_prob, raw_gain, adj, eff, reasoning)


def _result(rec, conf, circuit, current_lap, total_laps, lap_in_stint,
             gap_ahead, gap_behind, pit_delta,
             deg_rate, deg_loss, sc_prob, raw_gain, adj_gain, eff_gain, reasoning):
    return {
        "recommendation":   rec,
        "confidence":       round(conf, 3),
        "circuit":          circuit,
        "current_lap":      current_lap,
        "total_laps":       total_laps,
        "laps_remaining":   total_laps - current_lap,
        "lap_in_stint":     lap_in_stint,
        "gap_ahead":        gap_ahead,
        "gap_behind":       gap_behind,
        "pit_delta":        pit_delta,
        "deg_rate":         round(float(deg_rate), 4),
        "deg_loss_per_lap": deg_loss,
        "sc_prob":          sc_prob,
        "raw_gain":         raw_gain,
        "sc_adj_gain":      adj_gain,
        "eff_gain":         eff_gain,
        "overtake_factor":  OVERTAKE_FACTOR.get(circuit, DEFAULT_OVERTAKE_FACTOR),
        "reasoning":        reasoning,
    }


def print_decision(r):
    sym = {"STAY": "·", "UNDERCUT": "▲", "BOX": "■"}[r["recommendation"]]
    print(f"\n{'='*56}")
    print(f"  {sym} {r['recommendation']}  ({r['confidence']:.0%} confidence)")
    print(f"{'='*56}")
    print(f"  Circuit:        {r['circuit']}  "
          f"(overtake {r['overtake_factor']:.2f})")
    print(f"  Lap:            {r['current_lap']}/{r['total_laps']}  "
          f"({r['laps_remaining']} remaining)")
    print(f"  Stint lap:      {r['lap_in_stint']}")
    g = r['gap_ahead']
    print(f"  Gap ahead:      {f'{g:.1f}s' if g is not None else '— (leading)'}")
    print(f"{'─'*56}")
    print(f"  Deg rate:       {r['deg_rate']:+.4f} s/lap")
    print(f"  Deg this lap:   {r['deg_loss_per_lap']:+.3f} s  (vs stint lap 1)")
    print(f"  SC probability: {r['sc_prob']:.1%}  (remaining race)")
    print(f"  Raw gain:       {r['raw_gain']:+.2f} s")
    print(f"  SC-adj gain:    {r['sc_adj_gain']:+.2f} s")
    print(f"  Eff gain:       {r['eff_gain']:+.2f} s  (overtake-adjusted)")
    print(f"{'─'*56}")
    print(f"  {r['reasoning']}")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    print("PITWALL — Phase 4: Decision Engine (v3)")
    print("Demo scenarios\n")

    scenarios = [
        # label, circuit, lap, total, stint, gap_ahead, gap_behind, pit_delta, expected
        ("Bahrain 2024 — lap 14, undercut window",
         "Bahrain",    14, 32, 14,  1.8, 12.0, 22.0, "UNDERCUT"),
        ("Monaco 2024 — lap 20, track position circuit",
         "Monaco",     20, 42, 20,  3.5, 28.0, 28.0, "STAY"),
        ("Monza 2023 — lap 18, SC chaos, wait",
         "Monza",      18, 30, 12,  4.2, 18.0, 20.0, "STAY"),
        ("Spa 2024 — lap 10, high deg",
         "Spa",        10, 25, 10,  8.5,  5.0, 23.0, "UNDERCUT"),
        ("Barcelona 2024 — lap 25, tight gap",
         "Barcelona",  25, 37, 15,  0.8, 20.0, 22.0, "UNDERCUT"),
        ("Budapest 2023 — lap 20, clear undercut",
         "Budapest",   20, 37, 20, 14.0, 25.0, 21.0, "UNDERCUT"),
    ]

    correct = 0
    for label, circuit, lap, total, stint, gap_a, gap_b, delta, expected in scenarios:
        print(f"SCENARIO: {label}")
        r = decide(circuit, lap, total, stint, gap_a, gap_b, delta)
        print_decision(r)
        got = r["recommendation"]
        ok  = "✓" if got == expected else f"✗ got {got}"
        print(f"  Expected: {expected}  {ok}\n")
        if got == expected:
            correct += 1

    print(f"{'='*56}")
    print(f"  Scenarios correct: {correct}/{len(scenarios)}")
    print(f"{'='*56}")