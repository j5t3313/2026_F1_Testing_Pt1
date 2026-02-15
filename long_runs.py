import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import LONG_RUN_MIN_LAPS
from plotting import (
    apply_theme, create_figure, build_color_maps,
    get_compound_color, add_watermark, save_figure,
)


def identify_long_runs(laps, min_laps=None):
    threshold = min_laps or LONG_RUN_MIN_LAPS

    valid = laps.dropna(subset=["LapTimeSeconds"]).copy()
    valid = valid[valid["LapTimeSeconds"] > 0]

    stints = (
        valid.groupby(["Team", "Driver", "Day", "Stint"])
        .agg(
            StintLaps=("LapTimeSeconds", "count"),
            Compound=("Compound", "first"),
            MeanTime=("LapTimeSeconds", "mean"),
            StdTime=("LapTimeSeconds", "std"),
            MinTime=("LapTimeSeconds", "min"),
            MaxTime=("LapTimeSeconds", "max"),
        )
        .reset_index()
    )

    long_runs = stints[stints["StintLaps"] >= threshold].copy()
    long_runs["CoV"] = long_runs["StdTime"] / long_runs["MeanTime"]
    long_runs["Range"] = long_runs["MaxTime"] - long_runs["MinTime"]

    return long_runs


def get_long_run_laps(laps, long_runs):
    result = []
    for _, run in long_runs.iterrows():
        mask = (
            (laps["Team"] == run["Team"])
            & (laps["Driver"] == run["Driver"])
            & (laps["Day"] == run["Day"])
            & (laps["Stint"] == run["Stint"])
            & (laps["LapTimeSeconds"].notna())
            & (laps["LapTimeSeconds"] > 0)
        )
        stint_laps = laps[mask].copy()
        stint_laps = stint_laps.sort_values("LapNumber")
        stint_laps["StintLapNumber"] = range(1, len(stint_laps) + 1)
        stint_laps["DeltaFromMean"] = stint_laps["LapTimeSeconds"] - stint_laps["LapTimeSeconds"].mean()
        result.append(stint_laps)

    if not result:
        return pd.DataFrame()
    return pd.concat(result, ignore_index=True)


def compute_consistency_by_team(long_runs):
    return (
        long_runs.groupby("Team")
        .agg(
            MeanCoV=("CoV", "mean"),
            MedianCoV=("CoV", "median"),
            NumLongRuns=("CoV", "count"),
            MeanRange=("Range", "mean"),
        )
        .reset_index()
        .sort_values("MedianCoV")
    )


def plot_long_run_traces(laps, long_runs):
    apply_theme()
    team_colors, driver_colors = build_color_maps(laps)
    run_laps = get_long_run_laps(laps, long_runs)

    if run_laps.empty:
        return None

    fig, ax = create_figure(width=14, height=8)

    run_keys = run_laps.groupby(["Team", "Driver", "Day", "Stint"]).ngroups
    for (team, driver, day, stint), group in run_laps.groupby(["Team", "Driver", "Day", "Stint"]):
        color = driver_colors.get(driver, team_colors.get(team, "#888888"))
        ax.plot(
            group["StintLapNumber"], group["DeltaFromMean"],
            color=color, alpha=0.5, linewidth=1.2,
        )

    ax.axhline(y=0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lap Within Stint")
    ax.set_ylabel("Delta from Stint Mean (seconds)")
    ax.set_title(f"Long Run Lap Time Traces (stints ≥ {LONG_RUN_MIN_LAPS} laps)")

    teams_in_data = run_laps["Team"].unique()
    handles = [
        plt.Line2D([0], [0], color=team_colors.get(t, "#888888"), linewidth=2, label=t)
        for t in sorted(teams_in_data)
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=9)

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_consistency_rankings(long_runs):
    apply_theme()
    consistency = compute_consistency_by_team(long_runs)

    if consistency.empty:
        return None

    from plotting import build_color_maps
    import matplotlib.patches as mpatches

    fig, ax = create_figure(width=12, height=7)

    colors = []
    for team in consistency["Team"]:
        from config import TEAM_COLORS, FALLBACK_COLOR
        colors.append(TEAM_COLORS.get(team, FALLBACK_COLOR))

    bars = ax.barh(
        consistency["Team"], consistency["MedianCoV"] * 100,
        color=colors, edgecolor="white",
    )

    for bar, count in zip(bars, consistency["NumLongRuns"]):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"n={count}", va="center", fontsize=10, color="#666666",
        )

    ax.set_xlabel("Median Coefficient of Variation (%)")
    ax.set_title("Long Run Consistency by Team (lower = more consistent)")
    ax.invert_yaxis()

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_long_runs_by_compound(laps, long_runs):
    apply_theme()
    team_colors, driver_colors = build_color_maps(laps)
    run_laps = get_long_run_laps(laps, long_runs)

    if run_laps.empty:
        return None

    compounds = sorted(run_laps["Compound"].dropna().unique())
    n = len(compounds)
    if n == 0:
        return None

    fig, axes = create_figure(width=14, height=5 * n, nrows=n)
    if n == 1:
        axes = [axes]

    for idx, compound in enumerate(compounds):
        ax = axes[idx]
        compound_data = run_laps[run_laps["Compound"] == compound]

        for (team, driver, day, stint), group in compound_data.groupby(
            ["Team", "Driver", "Day", "Stint"]
        ):
            color = driver_colors.get(driver, team_colors.get(team, "#888888"))
            ax.plot(
                group["StintLapNumber"], group["DeltaFromMean"],
                color=color, alpha=0.5, linewidth=1.2,
            )

        ax.axhline(y=0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lap Within Stint")
        ax.set_ylabel("Delta from Stint Mean (s)")
        compound_color = get_compound_color(compound)
        ax.set_title(f"{compound}", color=compound_color, fontweight="bold")

    fig.suptitle(
        f"Long Run Traces by Compound (stints ≥ {LONG_RUN_MIN_LAPS} laps)",
        fontsize=16, fontweight="bold", y=1.01,
    )
    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_all(laps):
    long_runs = identify_long_runs(laps)
    figures = {}
    figures["long_run_traces"] = plot_long_run_traces(laps, long_runs)
    figures["consistency_rankings"] = plot_consistency_rankings(long_runs)
    figures["long_runs_by_compound"] = plot_long_runs_by_compound(laps, long_runs)
    return figures
