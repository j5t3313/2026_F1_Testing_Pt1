from pathlib import Path
from config import OUTPUT_DIR
from data_loader import setup, load_2026, load_2025, get_clean_laps
from plotting import apply_theme, save_figure
import reliability
import distributions
import long_runs
import speed_traces
import calibration


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    setup()
    apply_theme()

    print("Loading 2026 testing data...")
    sessions_2026, laps_2026 = load_2026()
    clean_2026 = get_clean_laps(laps_2026)
    print(f"  {len(laps_2026)} total laps, {len(clean_2026)} after filtering")

    print("Loading 2025 testing data...")
    sessions_2025, laps_2025 = load_2025()
    clean_2025 = get_clean_laps(laps_2025)
    print(f"  {len(laps_2025)} total laps, {len(clean_2025)} after filtering")

    print("\n--- Module 1: Reliability & Program Maturity ---")
    rel_figs = reliability.generate_all(laps_2026)
    for name, fig in rel_figs.items():
        if fig is not None:
            path = save_figure(fig, f"reliability_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 2: Lap Time Distributions ---")
    dist_figs = distributions.generate_all(clean_2026)
    for name, fig in dist_figs.items():
        if fig is not None:
            path = save_figure(fig, f"distributions_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 3: Long Run Consistency ---")
    lr_figs = long_runs.generate_all(clean_2026)
    for name, fig in lr_figs.items():
        if fig is not None:
            path = save_figure(fig, f"long_runs_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 4: Speed Traces (2026 vs 2025) ---")
    st_figs = speed_traces.generate_speed_traces(sessions_2026, sessions_2025)
    for name, fig in st_figs.items():
        if fig is not None:
            path = save_figure(fig, f"speed_traces_{name}.png")
            print(f"  Saved: {path}")

    print("\n--- Module 5: Calibration (2025 Testing vs Season vs 2026 Testing) ---")
    cal_result = calibration.generate_all(clean_2025, clean_2026)
    if cal_result is not None:
        cal_figs, pace_2025, pace_2026, comparison = cal_result
        for name, fig in cal_figs.items():
            if fig is not None:
                path = save_figure(fig, f"calibration_{name}.png")
                print(f"  Saved: {path}")

        print("\n  2025 Long Run Pace (testing):")
        print(pace_2025[["Team", "MeanLongRunPace", "DeltaToLeader", "TestingRank", "NumLongRuns"]].to_string(index=False))

        print("\n  2026 Long Run Pace (testing):")
        print(pace_2026[["Team", "MeanLongRunPace", "DeltaToLeader", "TestingRank", "NumLongRuns"]].to_string(index=False))

        print("\n  Comparison Table:")
        display_cols = [
            "Team_2026", "Testing_Delta_2025", "WCC_Finish_2025",
            "Testing_Delta_2026", "Testing_Rank_2026", "NumLongRuns_2026",
        ]
        available = [c for c in display_cols if c in comparison.columns]
        print(comparison[available].to_string(index=False))

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
