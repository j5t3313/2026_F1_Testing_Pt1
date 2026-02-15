import colorsys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from config import (
    TEAM_COLORS, COMPOUND_COLORS, FALLBACK_COLOR,
    FIGURE_DPI, FIGURE_WIDTH, FIGURE_HEIGHT,
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE,
)


def apply_theme():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.5,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "font.family": "sans-serif",
        "figure.dpi": FIGURE_DPI,
        "figure.figsize": (FIGURE_WIDTH, FIGURE_HEIGHT),
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
    })


def create_figure(width=None, height=None, nrows=1, ncols=1):
    w = width or FIGURE_WIDTH
    h = height or FIGURE_HEIGHT
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.set_facecolor("white")
    return fig, axes


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def adjust_lightness(hex_color, offset):
    r, g, b = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.05, min(0.95, l + offset))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return rgb_to_hex(r2, g2, b2)


def resolve_team_color(team_name):
    return TEAM_COLORS.get(team_name, FALLBACK_COLOR)


def generate_driver_variants(base_color, n_drivers):
    if n_drivers <= 1:
        return [base_color]

    r, g, b = hex_to_rgb(base_color)
    _, l, _ = colorsys.rgb_to_hls(r, g, b)

    if l >= 0.5:
        return [base_color, adjust_lightness(base_color, -0.2)]
    return [base_color, adjust_lightness(base_color, 0.2)]


def build_color_maps(laps):
    teams = laps["Team"].dropna().unique()
    team_colors = {}
    driver_colors = {}

    for team in sorted(teams):
        color = resolve_team_color(team)
        team_colors[team] = color

        team_drivers = sorted(laps[laps["Team"] == team]["Driver"].dropna().unique())
        variants = generate_driver_variants(color, len(team_drivers))
        for driver, variant in zip(team_drivers, variants):
            driver_colors[driver] = variant

    return team_colors, driver_colors


def get_compound_color(compound):
    return COMPOUND_COLORS.get(str(compound).upper(), COMPOUND_COLORS["UNKNOWN"])


def add_watermark(fig, text="@formulasteele"):
    fig.text(
        0.99, 0.01, text,
        fontsize=9, color="#AAAAAA",
        ha="right", va="bottom",
        transform=fig.transFigure,
    )


def save_figure(fig, filename, output_dir=None):
    from config import OUTPUT_DIR
    path = (output_dir or OUTPUT_DIR) / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
