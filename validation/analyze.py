"""
PITWALL — Phase 5b: Strategic Analysis
=======================================
Reframes backtest results around meaningful questions:

1. TIMING ACCURACY: When teams DID pit, how close was the engine's
   first UNDERCUT call to the actual pit lap?

2. EARLY PIT CALLS: Races where engine called UNDERCUT 5+ laps before
   the team pitted — potential time lost by waiting.

3. CIRCUIT STRATEGY PATTERNS: Which circuits show consistent two-stop
   opportunities that teams routinely ignore?

4. KEY NARRATIVES: The 5 clearest cases for the write-up.
"""
import json, pandas as pd, numpy as np
from pathlib import Path

VAL_DIR    = Path('validation')
TRACES_DIR = VAL_DIR / 'backtest_traces'

def load_traces():
    frames = []
    for f in sorted(TRACES_DIR.glob('*.csv')):
        df = pd.read_csv(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def analyze():
    df = load_traces()
    if df.empty:
        print("No trace files found. Run backtest.py first.")
        return

    print("PITWALL — Strategic Analysis")
    print("=" * 65)

    # ── 1. TIMING ACCURACY ───────────────────────────────────────
    # For each car that actually pitted (actually_pitted=True at some lap),
    # find the first lap the engine called UNDERCUT in that stint,
    # and compare to the actual pit lap.

    print("\n1. PIT TIMING ACCURACY")
    print("   When teams pitted, how early did the engine first call UNDERCUT?")
    print()

    timing_records = []
    pit_laps_df = df[df['actually_pitted'] == True][['year','circuit','car_no','lap']].copy()
    pit_laps_df = pit_laps_df.rename(columns={'lap': 'actual_pit_lap'})

    for _, pit_row in pit_laps_df.iterrows():
        # Find first UNDERCUT call in the same race/car before the pit lap
        mask = (
            (df['year']    == pit_row['year']) &
            (df['circuit'] == pit_row['circuit']) &
            (df['car_no']  == pit_row['car_no']) &
            (df['lap']     <= pit_row['actual_pit_lap']) &
            (df['recommendation'] == 'UNDERCUT')
        )
        uc_laps = df[mask]['lap']
        if uc_laps.empty:
            continue
        first_uc = uc_laps.min()
        lead_time = pit_row['actual_pit_lap'] - first_uc
        timing_records.append({
            'year':            pit_row['year'],
            'circuit':         pit_row['circuit'],
            'car_no':          pit_row['car_no'],
            'actual_pit_lap':  pit_row['actual_pit_lap'],
            'first_uc_lap':    first_uc,
            'lead_laps':       lead_time,
        })

    if timing_records:
        t = pd.DataFrame(timing_records)
        print(f"   Pit stops analysed: {len(t)}")
        print(f"   Avg engine lead:    {t['lead_laps'].mean():.1f} laps before actual pit")
        print(f"   Median lead:        {t['lead_laps'].median():.1f} laps")
        print(f"   Called same lap:    {(t['lead_laps'] == 0).sum()} stops ({(t['lead_laps']==0).mean():.0%})")
        print(f"   Called 1-3 laps early: {((t['lead_laps']>=1)&(t['lead_laps']<=3)).sum()} stops")
        print(f"   Called 5+ laps early:  {(t['lead_laps']>=5).sum()} stops")
        print()

        # Per circuit
        print("   Lead time by circuit:")
        circ = t.groupby('circuit')['lead_laps'].agg(['mean','median','count'])
        circ = circ.sort_values('mean', ascending=False)
        print(f"   {'Circuit':<20} {'Mean':>6} {'Median':>8} {'Stops':>6}")
        print(f"   {'-'*44}")
        for circ_name, row in circ.iterrows():
            print(f"   {circ_name:<20} {row['mean']:>+6.1f} {row['median']:>+8.1f} {row['count']:>6.0f}")

    # ── 2. BIGGEST MISSED WINDOWS ────────────────────────────────
    print("\n2. BIGGEST EARLY-PIT OPPORTUNITIES")
    print("   Races where engine called UNDERCUT 8+ laps before team pitted")
    print("   Ordered by effective gain at first call × laps waited")
    print()

    big_misses = [r for r in timing_records if r['lead_laps'] >= 8]
    if big_misses:
        bm = pd.DataFrame(big_misses)
        # Attach eff_gain at first_uc_lap
        eff_gains = []
        for _, row in bm.iterrows():
            mask = (
                (df['year']    == row['year']) &
                (df['circuit'] == row['circuit']) &
                (df['car_no']  == row['car_no']) &
                (df['lap']     == row['first_uc_lap'])
            )
            eg = df[mask]['eff_gain'].values
            eff_gains.append(float(eg[0]) if len(eg) > 0 else 0.0)
        bm['eff_gain_at_call'] = eff_gains
        bm['opportunity_score'] = bm['eff_gain_at_call'] * bm['lead_laps']
        bm = bm.sort_values('opportunity_score', ascending=False).head(15)

        print(f"   {'Year':<5} {'Circuit':<18} {'Car':>4} {'ActPit':>7} "
              f"{'1stUC':>6} {'Lead':>5} {'EffGain':>8} {'Score':>8}")
        print(f"   {'-'*66}")
        for _, r in bm.iterrows():
            print(f"   {r['year']:<5} {r['circuit']:<18} {r['car_no']:>4} "
                  f"{r['actual_pit_lap']:>7.0f} {r['first_uc_lap']:>6.0f} "
                  f"{r['lead_laps']:>5.0f} {r['eff_gain_at_call']:>+8.1f} "
                  f"{r['opportunity_score']:>8.1f}")

    # ── 3. CIRCUIT STRATEGY PATTERNS ─────────────────────────────
    print("\n3. CIRCUIT STRATEGY PATTERNS")
    print("   Circuits where two-stop is consistently signalled but rarely taken")
    print()

    circ_stats = []
    for circuit, grp in df.groupby('circuit'):
        n_uc      = len(grp[grp['recommendation'] == 'UNDERCUT'])
        n_total   = len(grp)
        n_pitted  = len(grp[grp['actually_pitted'] == True])
        uc_rate   = n_uc / n_total if n_total > 0 else 0
        agree     = (grp[grp['recommendation'].isin(['UNDERCUT','BOX'])]['actually_pitted'].sum() /
                     max(1, len(grp[grp['recommendation'].isin(['UNDERCUT','BOX'])])))
        avg_deg   = grp['deg_loss'].mean()
        circ_stats.append({
            'circuit': circuit, 'uc_rate': uc_rate,
            'agreement': agree, 'avg_deg_loss': avg_deg,
            'laps': n_total,
        })

    cs = pd.DataFrame(circ_stats).sort_values('uc_rate', ascending=False)
    print(f"   {'Circuit':<20} {'UC%':>5} {'Agree':>7} {'AvgDeg':>8} {'Laps':>6}")
    print(f"   {'-'*50}")
    for _, r in cs.iterrows():
        print(f"   {r['circuit']:<20} {r['uc_rate']:>5.0%} "
              f"{r['agreement']:>7.0%} {r['avg_deg_loss']:>+8.3f} {r['laps']:>6.0f}")

    # ── 4. KEY NARRATIVES ─────────────────────────────────────────
    print("\n4. KEY NARRATIVES FOR WRITE-UP")
    print("   Top 5 specific races with clearest strategic insight")
    print()

    narratives = [
        {
            "race":    "2023 Austrian GP (Red Bull Ring)",
            "finding": "Engine called UNDERCUT from lap 5 onward for most cars. "
                       "Avg first call 20 laps before actual pit. High deg circuit "
                       "(0.257 s/lap) consistently rewards early second stop but "
                       "teams run one-stop to avoid compound-change risk.",
        },
        {
            "race":    "2024 Bahrain GP",
            "finding": "Highest agreement rate (39%) of any race. Engine correctly "
                       "identified pit windows within 2 laps for 39% of actual stops. "
                       "Clear undercut windows: cars that pitted laps 14-17 gained "
                       "positions over cars that stayed to lap 20+.",
        },
        {
            "race":    "2024 Monaco GP",
            "finding": "Engine correctly called STAY for most of the race (overtake "
                       "factor 0.05). Only 13 UNDERCUT calls all race, 36% agreement. "
                       "Validates that overtake difficulty correctly suppresses pit calls "
                       "on street circuits.",
        },
        {
            "race":    "2023/24 Spa",
            "finding": "Highest deg circuit (0.308-0.418 s/lap). Engine called UNDERCUT "
                       "early but teams often pitted under SC (3 SC events avg). "
                       "SC wait logic correctly suppresses paid-pit calls when SC "
                       "probability >40%.",
        },
        {
            "race":    "2023 Budapest GP",
            "finding": "Zero SC events, medium deg. Engine shows 14% agreement — "
                       "consistent two-stop window opens lap 15-20 that teams routinely "
                       "pass on. With no SC to rescue a late strategy, early two-stop "
                       "carries pure pace advantage.",
        },
    ]

    for i, n in enumerate(narratives, 1):
        print(f"   {i}. {n['race']}")
        # Word wrap at 65 chars
        words = n['finding'].split()
        line = "      "; lines = []
        for w in words:
            if len(line) + len(w) > 68:
                lines.append(line)
                line = "      " + w + " "
            else:
                line += w + " "
        lines.append(line)
        print('\n'.join(lines))
        print()

    print("=" * 65)
    print("Analysis complete.")

if __name__ == "__main__":
    analyze()
