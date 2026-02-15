import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting import (
    apply_theme, create_figure, build_color_maps,
    get_compound_color, add_watermark, save_figure,
)


def compute_team_stats(laps):
    stats = (
        laps.groupby("Team")["LapTimeSeconds"]
        .agg(["min", "median", "mean", "std", "count"])
        .reset_index()
    )
    stats["headline_gap"] = stats["median"] - stats["min"]
    return stats.sort_values("median")


def plot_team_violins(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)

    teams_ordered = (
        laps.groupby("Team")["LapTimeSeconds"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = create_figure(width=14, height=8)

    for i, team in enumerate(teams_ordered):
        team_data = laps[laps["Team"] == team]["LapTimeSeconds"].dropna()
        if team_data.empty:
            continue

        color = team_colors.get(team, "#888888")
        parts = ax.violinplot(
            team_data, positions=[i], showmeans=False,
            showmedians=True, showextrema=False, widths=0.7,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("#333333")
        parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(teams_ordered)))
    ax.set_xticklabels(teams_ordered, rotation=45, ha="right")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title("Lap Time Distributions by Team")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_compound_distributions(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)
    compounds = [c for c in ["SOFT", "MEDIUM", "HARD"] if c in laps["Compound"].unique()]

    if not compounds:
        compounds = laps["Compound"].dropna().unique().tolist()

    n_compounds = len(compounds)
    if n_compounds == 0:
        return None

    fig, axes = create_figure(width=14, height=5 * n_compounds, nrows=n_compounds)
    if n_compounds == 1:
        axes = [axes]

    for idx, compound in enumerate(compounds):
        ax = axes[idx]
        compound_laps = laps[laps["Compound"] == compound]

        teams_ordered = (
            compound_laps.groupby("Team")["LapTimeSeconds"]
            .median()
            .sort_values()
            .index.tolist()
        )

        for i, team in enumerate(teams_ordered):
            team_data = compound_laps[compound_laps["Team"] == team]["LapTimeSeconds"].dropna()
            if team_data.empty:
                continue

            color = team_colors.get(team, "#888888")
            parts = ax.violinplot(
                team_data, positions=[i], showmeans=False,
                showmedians=True, showextrema=False, widths=0.7,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("#333333")
            parts["cmedians"].set_linewidth(2)

        ax.set_xticks(range(len(teams_ordered)))
        ax.set_xticklabels(teams_ordered, rotation=45, ha="right")
        ax.set_ylabel("Lap Time (seconds)")

        compound_color = get_compound_color(compound)
        ax.set_title(f"{compound} Compound", color=compound_color, fontweight="bold")

    fig.suptitle("Lap Time Distributions by Compound", fontsize=18, fontweight="bold", y=1.01)
    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_headline_vs_median(laps):
    apply_theme()
    team_colors, _ = build_color_maps(laps)
    stats = compute_team_stats(laps)

    fig, ax = create_figure(width=12, height=7)

    for _, row in stats.iterrows():
        color = team_colors.get(row["Team"], "#888888")
        ax.scatter(
            row["headline_gap"], row["median"],
            s=row["count"] * 2, color=color,
            edgecolors="#333333", linewidth=0.5, zorder=5,
        )
        ax.annotate(
            row["Team"], (row["headline_gap"], row["median"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("Gap: Median to Fastest Lap (seconds)")
    ax.set_ylabel("Median Lap Time (seconds)")
    ax.set_title("Program Focus: Headline Time vs Typical Running Pace")
    ax.invert_yaxis()

    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_all(laps):
    figures = {}
    figures["team_violins"] = plot_team_violins(laps)
    figures["compound_distributions"] = plot_compound_distributions(laps)
    figures["headline_vs_median"] = plot_headline_vs_median(laps)
    return figures
