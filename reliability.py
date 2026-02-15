import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plotting import (
    apply_theme, create_figure, build_color_maps,
    add_watermark, save_figure,
)


def compute_laps_per_team_day(laps):
    return (
        laps.groupby(["Team", "Day"])
        .size()
        .reset_index(name="Laps")
        .pivot(index="Team", columns="Day", values="Laps")
        .fillna(0)
        .astype(int)
    )


def compute_total_laps(laps):
    totals = laps.groupby("Team").size().reset_index(name="TotalLaps")
    return totals.sort_values("TotalLaps", ascending=False)


def compute_stint_summary(laps):
    stints = (
        laps.groupby(["Team", "Driver", "Day", "Stint"])
        .agg(StintLaps=("LapNumber", "count"))
        .reset_index()
    )

    summary = (
        stints.groupby("Team")
        .agg(
            TotalStints=("Stint", "count"),
            MaxStintLength=("StintLaps", "max"),
            MeanStintLength=("StintLaps", "mean"),
        )
        .reset_index()
        .sort_values("TotalStints", ascending=False)
    )
    return summary


def compute_laps_per_driver(laps):
    return (
        laps.groupby(["Team", "Driver"])
        .size()
        .reset_index(name="Laps")
        .sort_values(["Team", "Laps"], ascending=[True, False])
    )


def plot_laps_heatmap(laps):
    apply_theme()
    grid = compute_laps_per_team_day(laps)
    grid["Total"] = grid.sum(axis=1)
    grid = grid.sort_values("Total", ascending=True)
    display = grid.drop(columns="Total")

    fig, ax = create_figure(width=12, height=8)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#F5F5F5", "#2166AC"])
    im = ax.imshow(display.values, cmap=cmap, aspect="auto")

    ax.set_yticks(range(len(display.index)))
    ax.set_yticklabels(display.index)
    ax.set_xticks(range(len(display.columns)))
    ax.set_xticklabels([f"Day {c}" for c in display.columns])

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display.iloc[i, j]
            text_color = "white" if val > display.values.max() * 0.6 else "#333333"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Laps Completed")
    ax.set_title("Programme Maturity: Laps Completed Per Day")
    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_total_laps_bar(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)
    totals = compute_total_laps(laps)

    fig, ax = create_figure(width=12, height=7)
    colors = [team_colors.get(t, "#888888") for t in totals["Team"]]
    bars = ax.barh(totals["Team"], totals["TotalLaps"], color=colors, edgecolor="white")

    for bar, val in zip(bars, totals["TotalLaps"]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Total Laps")
    ax.set_title("Total Laps Completed Across All Test Days")
    ax.invert_yaxis()
    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_stint_lengths(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)
    summary = compute_stint_summary(laps)

    fig, axes = create_figure(width=14, height=6, ncols=2)

    summary_sorted = summary.sort_values("MaxStintLength", ascending=True)
    colors = [team_colors.get(t, "#888888") for t in summary_sorted["Team"]]
    axes[0].barh(summary_sorted["Team"], summary_sorted["MaxStintLength"], color=colors)
    axes[0].set_xlabel("Laps")
    axes[0].set_title("Longest Single Stint")

    summary_sorted2 = summary.sort_values("TotalStints", ascending=True)
    colors2 = [team_colors.get(t, "#888888") for t in summary_sorted2["Team"]]
    axes[1].barh(summary_sorted2["Team"], summary_sorted2["TotalStints"], color=colors2)
    axes[1].set_xlabel("Stints")
    axes[1].set_title("Total Stint Count")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_all(laps):
    figures = {}
    figures["laps_heatmap"] = plot_laps_heatmap(laps)
    figures["total_laps"] = plot_total_laps_bar(laps)
    figures["stint_lengths"] = plot_stint_lengths(laps)
    return figures
