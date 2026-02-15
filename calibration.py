import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import LONG_RUN_MIN_LAPS, TEAM_COLORS, FALLBACK_COLOR
from plotting import (
    apply_theme, create_figure, add_watermark, save_figure,
)
from long_runs import identify_long_runs


WCC_2025 = {
    "McLaren": 1,
    "Mercedes": 2,
    "Red Bull Racing": 3,
    "Red Bull": 3,
    "Ferrari": 4,
    "Williams": 5,
    "Williams Racing": 5,
    "Racing Bulls": 6,
    "RB": 6,
    "Aston Martin": 7,
    "Aston Martin Racing": 7,
    "Haas F1 Team": 8,
    "Haas": 8,
    "Kick Sauber": 9,
    "Alpine": 10,
    "Alpine F1 Team": 10,
}

WCC_2025_POINTS = {
    "McLaren": 833,
    "Mercedes": 469,
    "Red Bull Racing": 451,
    "Red Bull": 451,
    "Ferrari": 398,
    "Williams": 137,
    "Williams Racing": 137,
    "Racing Bulls": 92,
    "RB": 92,
    "Aston Martin": 89,
    "Aston Martin Racing": 89,
    "Haas F1 Team": 79,
    "Haas": 79,
    "Kick Sauber": 70,
    "Alpine": 22,
    "Alpine F1 Team": 22,
}

WCC_2025_NOTES = {
    "Red Bull Racing": "Verstappen scored 421 of 451 team points",
    "Red Bull": "Verstappen scored 421 of 451 team points",
}

TEAM_NAME_MAP_2025_TO_2026 = {
    "McLaren": "McLaren",
    "Mercedes": "Mercedes",
    "Red Bull Racing": "Red Bull Racing",
    "Red Bull": "Red Bull",
    "Ferrari": "Ferrari",
    "Williams": "Williams",
    "Williams Racing": "Williams",
    "Racing Bulls": "Racing Bulls",
    "RB": "Racing Bulls",
    "Aston Martin": "Aston Martin",
    "Aston Martin Racing": "Aston Martin",
    "Haas F1 Team": "Haas F1 Team",
    "Haas": "Haas",
    "Kick Sauber": "Audi",
    "Alpine": "Alpine",
    "Alpine F1 Team": "Alpine",
}


def compute_long_run_pace(laps, min_laps=None):
    long_runs = identify_long_runs(laps, min_laps=min_laps)

    if long_runs.empty:
        return pd.DataFrame()

    team_pace = (
        long_runs.groupby("Team")
        .agg(
            MeanLongRunPace=("MeanTime", "mean"),
            NumLongRuns=("MeanTime", "count"),
            TotalLongRunLaps=("StintLaps", "sum"),
        )
        .reset_index()
    )

    leader_pace = team_pace["MeanLongRunPace"].min()
    team_pace["DeltaToLeader"] = team_pace["MeanLongRunPace"] - leader_pace
    team_pace["TestingRank"] = team_pace["MeanLongRunPace"].rank().astype(int)

    return team_pace.sort_values("TestingRank")


def build_calibration_table(pace_2025):
    if pace_2025.empty:
        return pd.DataFrame()

    table = pace_2025.copy()
    table["WCC_Finish"] = table["Team"].map(WCC_2025).fillna(0).astype(int)
    table["WCC_Points"] = table["Team"].map(WCC_2025_POINTS).fillna(0).astype(int)
    table["PositionShift"] = table["TestingRank"] - table["WCC_Finish"]
    table["Notes"] = table["Team"].map(WCC_2025_NOTES).fillna("")

    return table.sort_values("TestingRank")


def build_comparison_table(pace_2025, pace_2026):
    if pace_2025.empty or pace_2026.empty:
        return pd.DataFrame()

    cal_2025 = build_calibration_table(pace_2025)

    rows = []
    for _, row_25 in cal_2025.iterrows():
        team_2025 = row_25["Team"]
        team_2026 = TEAM_NAME_MAP_2025_TO_2026.get(team_2025, team_2025)

        match_2026 = pace_2026[pace_2026["Team"] == team_2026]

        entry = {
            "Team_2025": team_2025,
            "Team_2026": team_2026,
            "Testing_Delta_2025": row_25["DeltaToLeader"],
            "Testing_Rank_2025": row_25["TestingRank"],
            "WCC_Finish_2025": row_25["WCC_Finish"],
            "WCC_Points_2025": row_25["WCC_Points"],
            "Position_Shift_2025": row_25["PositionShift"],
            "Notes": row_25["Notes"],
        }

        if not match_2026.empty:
            r26 = match_2026.iloc[0]
            entry["Testing_Delta_2026"] = r26["DeltaToLeader"]
            entry["Testing_Rank_2026"] = r26["TestingRank"]
            entry["NumLongRuns_2026"] = r26["NumLongRuns"]
        else:
            entry["Testing_Delta_2026"] = np.nan
            entry["Testing_Rank_2026"] = np.nan
            entry["NumLongRuns_2026"] = 0

        rows.append(entry)

    new_2026_teams = set(pace_2026["Team"]) - set(
        TEAM_NAME_MAP_2025_TO_2026.get(t, t) for t in pace_2025["Team"]
    )
    for team_2026 in sorted(new_2026_teams):
        match = pace_2026[pace_2026["Team"] == team_2026]
        if not match.empty:
            r26 = match.iloc[0]
            rows.append({
                "Team_2025": "(new entry)",
                "Team_2026": team_2026,
                "Testing_Delta_2025": np.nan,
                "Testing_Rank_2025": np.nan,
                "WCC_Finish_2025": np.nan,
                "WCC_Points_2025": np.nan,
                "Position_Shift_2025": np.nan,
                "Notes": "New team for 2026",
                "Testing_Delta_2026": r26["DeltaToLeader"],
                "Testing_Rank_2026": r26["TestingRank"],
                "NumLongRuns_2026": r26["NumLongRuns"],
            })

    return pd.DataFrame(rows)


def plot_bump_chart(calibration_table):
    apply_theme()

    valid = calibration_table[
        calibration_table["WCC_Finish"].notna()
        & (calibration_table["WCC_Finish"] > 0)
    ].copy()

    if valid.empty:
        return None

    fig, ax = create_figure(width=12, height=9)

    max_rank = max(valid["TestingRank"].max(), valid["WCC_Finish"].max())

    for _, row in valid.iterrows():
        team = row["Team"]
        color = TEAM_COLORS.get(team, FALLBACK_COLOR)
        test_rank = row["TestingRank"]
        wcc_rank = row["WCC_Finish"]

        ax.plot(
            [0, 1], [test_rank, wcc_rank],
            color=color, linewidth=3, alpha=0.8,
            marker="o", markersize=10, markeredgecolor="white", markeredgewidth=1.5,
        )

        ax.text(
            -0.05, test_rank, team,
            ha="right", va="center", fontsize=10, fontweight="bold", color=color,
        )
        label = team
        if row.get("Notes"):
            label = f"{team}*"
        ax.text(
            1.05, wcc_rank, label,
            ha="left", va="center", fontsize=10, fontweight="bold", color=color,
        )

    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(max_rank + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Testing Long Run Rank", "2025 WCC Finish"], fontsize=13, fontweight="bold")
    ax.set_yticks(range(1, int(max_rank) + 1))
    ax.set_ylabel("Position")
    ax.set_title("2025 Testing Long Run Pace vs Season Outcome")
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")

    if any(valid["Notes"].str.len() > 0):
        ax.text(
            0.5, max_rank + 0.3,
            "* Verstappen scored 421 of Red Bull's 451 points",
            ha="center", va="top", fontsize=9, style="italic", color="#666666",
            transform=ax.transData,
        )

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_delta_comparison(comparison_table):
    apply_theme()

    valid = comparison_table.dropna(subset=["Testing_Delta_2025", "Testing_Delta_2026"]).copy()
    if valid.empty:
        return None

    valid = valid.sort_values("Testing_Delta_2026")

    fig, ax = create_figure(width=14, height=9)

    y_positions = range(len(valid))
    bar_height = 0.35

    for i, (_, row) in enumerate(valid.iterrows()):
        team = row["Team_2026"]
        color = TEAM_COLORS.get(team, FALLBACK_COLOR)

        ax.barh(
            i - bar_height / 2, row["Testing_Delta_2025"],
            height=bar_height, color=color, alpha=0.4,
            edgecolor=color, linewidth=1,
        )
        ax.barh(
            i + bar_height / 2, row["Testing_Delta_2026"],
            height=bar_height, color=color, alpha=0.9,
            edgecolor=color, linewidth=1,
        )

        wcc = row.get("WCC_Finish_2025")
        if pd.notna(wcc) and wcc > 0:
            ax.text(
                max(row["Testing_Delta_2025"], row["Testing_Delta_2026"]) + 0.15,
                i,
                f"WCC P{int(wcc)}",
                va="center", fontsize=9, color="#666666",
            )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(valid["Team_2026"], fontsize=11)
    ax.set_xlabel("Delta to Long Run Pace Leader (seconds)")
    ax.set_title("Testing Long Run Pace Gap: 2025 vs 2026 (with 2025 Season Outcome)")

    legend_elements = [
        mpatches.Patch(facecolor="#888888", alpha=0.4, edgecolor="#888888", label="2025 Testing Gap"),
        mpatches.Patch(facecolor="#888888", alpha=0.9, edgecolor="#888888", label="2026 Testing Gap"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

    ax.invert_yaxis()
    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_shift_analysis(comparison_table):
    apply_theme()

    valid = comparison_table.dropna(
        subset=["Testing_Rank_2025", "Testing_Rank_2026", "WCC_Finish_2025"]
    ).copy()
    if valid.empty:
        return None

    fig, axes = create_figure(width=16, height=7, ncols=2)

    valid["TestingToSeason_2025"] = abs(valid["Testing_Rank_2025"] - valid["WCC_Finish_2025"])
    mean_shift = valid["TestingToSeason_2025"].mean()

    valid_sorted = valid.sort_values("Testing_Rank_2026")
    for i, (_, row) in enumerate(valid_sorted.iterrows()):
        team = row["Team_2026"]
        color = TEAM_COLORS.get(team, FALLBACK_COLOR)

        axes[0].plot(
            [0, 1, 2],
            [row["Testing_Rank_2025"], row["WCC_Finish_2025"], row["Testing_Rank_2026"]],
            color=color, linewidth=2.5, alpha=0.8,
            marker="o", markersize=8, markeredgecolor="white",
        )

    max_rank = max(
        valid["Testing_Rank_2025"].max(),
        valid["WCC_Finish_2025"].max(),
        valid["Testing_Rank_2026"].max(),
    )
    axes[0].set_xlim(-0.3, 2.3)
    axes[0].set_ylim(max_rank + 0.5, 0.5)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["2025 Testing", "2025 WCC", "2026 Testing"], fontsize=10)
    axes[0].set_ylabel("Position")
    axes[0].set_title("Team Trajectory: Testing \u2192 Season \u2192 Testing")

    for _, row in valid_sorted.iterrows():
        team = row["Team_2026"]
        color = TEAM_COLORS.get(team, FALLBACK_COLOR)
        axes[0].text(
            2.1, row["Testing_Rank_2026"], team,
            ha="left", va="center", fontsize=9, color=color, fontweight="bold",
        )

    colors = [TEAM_COLORS.get(t, FALLBACK_COLOR) for t in valid_sorted["Team_2026"]]
    bars = axes[1].barh(
        valid_sorted["Team_2026"],
        valid_sorted["TestingToSeason_2025"],
        color=colors, alpha=0.7,
    )
    axes[1].axvline(x=mean_shift, color="#333333", linewidth=1.5, linestyle="--", label=f"Mean: {mean_shift:.1f}")
    axes[1].set_xlabel("Absolute Position Change (Testing → Season)")
    axes[1].set_title("How Much Did 2025 Testing Rank Differ from Season Finish?")
    axes[1].legend(fontsize=10)
    axes[1].invert_yaxis()

    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_all(clean_2025, clean_2026):
    figures = {}

    pace_2025 = compute_long_run_pace(clean_2025)
    pace_2026 = compute_long_run_pace(clean_2026)

    if pace_2025.empty:
        print("  Warning: No long runs found in 2025 data")
        return figures
    if pace_2026.empty:
        print("  Warning: No long runs found in 2026 data")
        return figures

    calibration = build_calibration_table(pace_2025)
    comparison = build_comparison_table(pace_2025, pace_2026)

    figures["bump_chart"] = plot_bump_chart(calibration)
    figures["delta_comparison"] = plot_delta_comparison(comparison)
    figures["shift_analysis"] = plot_shift_analysis(comparison)

    return figures, pace_2025, pace_2026, comparison
