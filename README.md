# F1 2026 Pre-Season Testing Analysis

Data-driven analysis of the 2026 F1 Bahrain pre-season test (Test 1, Feb 11–13), with a 2025 testing baseline comparison. Built with FastF1.

## Modules

**Reliability & Programme Maturity** — Lap counts, stint structures, and programme completeness by team and day.

**Lap Time Distributions** — Full distributional analysis of running pace by team and compound. Median vs headline time gap.

**Long Run Consistency** — Coefficient of variation and lap time stability on stints of 10+ laps. Compound-separated analysis.

**Speed Trace Comparison** — 2026 vs 2025 telemetry overlay showing the character differences of the new regulation era: SLM profiles, ERS deployment, braking signatures.

## Setup

```
pip install -r requirements.txt
```

## Usage

Run the full analysis:

```
python run_analysis.py
```

Outputs are saved to `output/`.

To run individual modules or customize parameters, edit `config.py` 

## Configuration

`config.py` contains all tunable parameters:

- `TEST_DAYS` — which days to include (default: all 3)
- `LONG_RUN_MIN_LAPS` — minimum stint length for long run analysis
- `INLAP_THRESHOLD_FACTOR` — filtering threshold for in/out laps
- `TEAM_COLORS` — official team hex colors (update if FastF1 names differ)

## Data Source

All data sourced via [FastF1](https://github.com/theOehrly/Fast-F1) from F1 live timing.

```python
import fastf1
session = fastf1.get_testing_session(2026, 1, 1)
session.load(telemetry=True, weather=False)
```

## Limitations

- No fuel load data available; absolute pace comparisons are not meaningful
- Tire degradation cannot be separated from fuel burn within stints
- Engine modes and run plans are unknown
- This analysis focuses on what the data can defensibly tell us
