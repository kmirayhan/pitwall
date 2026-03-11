# PITWALL — F2 Pit Decision Engine

A data-driven pit stop strategy engine for FIA Formula 2, built on real race data and physics-based tire modeling.

## What it does

Given a live race state — current lap, gap to car ahead, laps on current tires, safety car probability — PITWALL outputs a strategy recommendation: **STAY / UNDERCUT / BOX**, with a confidence percentage and the three numbers behind it.

## Architecture

```
Phase 1  data/fetch.py        FIA PDF pipeline — lap times, pit stops, SC detection
Phase 2  models/deg.py        Tire degradation model — sec/lap lost by circuit
Phase 3  models/sc.py         Safety car probability — circuit × lap-of-race
Phase 4  models/engine.py     Decision engine — combines all three inputs
Phase 5  validation/           Backtested against every F2 Feature Race 2022–2024
```

## Data

- **Source**: FIA official timing PDFs (History Chart + Pit Stop Summary)
- **Coverage**: 20 F2 Feature Races, 2022–2024, 12 circuits
- **Clean laps**: 9,378 | **Stints analysed**: 582 | **SC events**: 36

## Tire Degradation (Phase 2 results)

Fuel-corrected degradation rate (seconds lost per lap of tire age):

| Circuit | deg (s/lap) | Circuit | deg (s/lap) |
|---------|-------------|---------|-------------|
| Spa | +0.308 | Silverstone | +0.134 |
| Red Bull Ring | +0.230 | Imola | +0.134 |
| Bahrain | +0.224 | Monaco | +0.081 |
| Barcelona | +0.189 | Melbourne | +0.079 |
| Budapest | +0.156 | Monza | +0.073 |
| Qatar | +0.143 | Jeddah | +0.018 |

Global median: **+0.119 s/lap**

## Status

- [x] Phase 1 — Data pipeline
- [x] Phase 2 — Tire degradation model
- [ ] Phase 3 — Safety car probability
- [ ] Phase 4 — Decision engine
- [ ] Phase 5 — Backtesting + validation

Target: published before 2026 Bahrain F2 round (April 10–12).

## Stack

Python 3.13 · pandas · scipy · pdfplumber · numpy

## Author

Mohammed Rayhan — [@kmirayhan](https://github.com/kmirayhan)
