import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting import (
    apply_theme, create_figure, build_color_maps,
    add_watermark, save_figure,
)


def get_fastest_soft_lap(sessions, driver=None):
    for session in sessions:
        laps = session.laps
        soft_laps = laps[laps["Compound"] == "SOFT"]

        if driver:
            soft_laps = soft_laps[soft_laps["Driver"] == driver]

        if soft_laps.empty:
            continue

        fastest = soft_laps.pick_fastest()
        if fastest is not None and not fastest.empty:
            return fastest, session

    return None, None


def get_fastest_lap(sessions, driver=None):
    for session in sessions:
        laps = session.laps

        if driver:
            laps = laps[laps["Driver"] == driver]

        if laps.empty:
            continue

        fastest = laps.pick_fastest()
        if fastest is not None and not fastest.empty:
            return fastest, session

    return None, None


def extract_telemetry(lap):
    try:
        tel = lap.get_telemetry()
        if tel is not None and not tel.empty:
            return tel
    except Exception:
        pass
    return None


def interpolate_to_common_distance(tel, n_points=1000):
    if tel is None or tel.empty:
        return None

    dist = tel["Distance"].values
    speed = tel["Speed"].values

    common_dist = np.linspace(dist.min(), dist.max(), n_points)
    interp_speed = np.interp(common_dist, dist, speed)

    result = pd.DataFrame({"Distance": common_dist, "Speed": interp_speed})

    if "Throttle" in tel.columns:
        result["Throttle"] = np.interp(common_dist, dist, tel["Throttle"].values)
    if "Brake" in tel.columns:
        brake_vals = tel["Brake"].values.astype(float)
        result["Brake"] = np.interp(common_dist, dist, brake_vals)
    if "nGear" in tel.columns:
        result["nGear"] = np.interp(common_dist, dist, tel["nGear"].values)
    if "DRS" in tel.columns:
        result["DRS"] = np.interp(common_dist, dist, tel["DRS"].values.astype(float))

    return result


def plot_speed_comparison(tel_2026, tel_2025, label_2026="2026", label_2025="2025"):
    apply_theme()
    fig, ax = create_figure(width=16, height=7)

    ax.plot(
        tel_2025["Distance"], tel_2025["Speed"],
        color="#AAAAAA", linewidth=1.5, alpha=0.8, label=label_2025,
    )
    ax.plot(
        tel_2026["Distance"], tel_2026["Speed"],
        color="#E8002D", linewidth=1.8, alpha=0.9, label=label_2026,
    )

    ax.fill_between(
        tel_2026["Distance"],
        tel_2025["Speed"],
        tel_2026["Speed"],
        where=tel_2026["Speed"] > tel_2025["Speed"],
        alpha=0.15, color="#2166AC", label="2026 faster",
    )
    ax.fill_between(
        tel_2026["Distance"],
        tel_2025["Speed"],
        tel_2026["Speed"],
        where=tel_2026["Speed"] < tel_2025["Speed"],
        alpha=0.15, color="#B2182B", label="2025 faster",
    )

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed Trace: 2026 New Era vs 2025 Regulations")
    ax.legend(loc="lower right")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_full_telemetry_comparison(tel_2026, tel_2025, label_2026="2026", label_2025="2025"):
    apply_theme()

    has_throttle = "Throttle" in tel_2026.columns and "Throttle" in tel_2025.columns
    has_brake = "Brake" in tel_2026.columns and "Brake" in tel_2025.columns
    has_gear = "nGear" in tel_2026.columns and "nGear" in tel_2025.columns

    n_rows = 1 + int(has_throttle) + int(has_brake) + int(has_gear)
    height_ratios = [3] + [1] * (n_rows - 1)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 4 + 2.5 * n_rows),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    fig.set_facecolor("white")

    row = 0

    axes[row].plot(tel_2025["Distance"], tel_2025["Speed"],
                   color="#AAAAAA", linewidth=1.5, alpha=0.8, label=label_2025)
    axes[row].plot(tel_2026["Distance"], tel_2026["Speed"],
                   color="#E8002D", linewidth=1.8, alpha=0.9, label=label_2026)
    axes[row].set_ylabel("Speed (km/h)")
    axes[row].set_title("Speed Trace: 2026 vs 2025 Bahrain Testing")
    axes[row].legend(loc="lower right")
    row += 1

    if has_throttle:
        axes[row].plot(tel_2025["Distance"], tel_2025["Throttle"],
                       color="#AAAAAA", linewidth=1.2, alpha=0.8)
        axes[row].plot(tel_2026["Distance"], tel_2026["Throttle"],
                       color="#2166AC", linewidth=1.5, alpha=0.9)
        axes[row].set_ylabel("Throttle %")
        axes[row].set_ylim(-5, 105)
        row += 1

    if has_brake:
        axes[row].plot(tel_2025["Distance"], tel_2025["Brake"],
                       color="#AAAAAA", linewidth=1.2, alpha=0.8)
        axes[row].plot(tel_2026["Distance"], tel_2026["Brake"],
                       color="#B2182B", linewidth=1.5, alpha=0.9)
        axes[row].set_ylabel("Brake")
        row += 1

    if has_gear:
        axes[row].plot(tel_2025["Distance"], tel_2025["nGear"],
                       color="#AAAAAA", linewidth=1.2, alpha=0.8)
        axes[row].plot(tel_2026["Distance"], tel_2026["nGear"],
                       color="#333333", linewidth=1.5, alpha=0.9)
        axes[row].set_ylabel("Gear")
        row += 1

    axes[-1].set_xlabel("Distance (m)")

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_speed_delta(tel_2026, tel_2025):
    apply_theme()
    fig, ax = create_figure(width=16, height=5)

    delta = tel_2026["Speed"].values - tel_2025["Speed"].values
    distance = tel_2026["Distance"].values

    ax.fill_between(
        distance, 0, delta,
        where=delta >= 0, alpha=0.6, color="#2166AC", label="2026 faster",
    )
    ax.fill_between(
        distance, 0, delta,
        where=delta < 0, alpha=0.6, color="#B2182B", label="2025 faster",
    )
    ax.axhline(y=0, color="#333333", linewidth=0.8)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed Delta (km/h)")
    ax.set_title("Speed Advantage: 2026 vs 2025")
    ax.legend()

    add_watermark(fig)
    fig.tight_layout()
    return fig


def plot_sector_comparison(laps_2026, laps_2025):
    apply_theme()

    sectors_2026 = (
        laps_2026.dropna(subset=["Sector1Time"])
        .groupby("Team")
        .agg({
            "Sector1Time": lambda x: x.dt.total_seconds().median(),
            "Sector2Time": lambda x: x.dt.total_seconds().median(),
            "Sector3Time": lambda x: x.dt.total_seconds().median(),
        })
        .reset_index()
    )

    sectors_2025 = (
        laps_2025.dropna(subset=["Sector1Time"])
        .agg({
            "Sector1Time": lambda x: x.dt.total_seconds().median(),
            "Sector2Time": lambda x: x.dt.total_seconds().median(),
            "Sector3Time": lambda x: x.dt.total_seconds().median(),
        })
    )

    fig, axes = create_figure(width=14, height=5, ncols=3)

    for idx, sector in enumerate(["Sector1Time", "Sector2Time", "Sector3Time"]):
        ax = axes[idx]
        data = sectors_2026.sort_values(sector)

        from config import TEAM_COLORS, FALLBACK_COLOR
        colors = [TEAM_COLORS.get(t, FALLBACK_COLOR) for t in data["Team"]]

        ax.barh(data["Team"], data[sector], color=colors, alpha=0.8)
        ax.axvline(x=sectors_2025[sector], color="#AAAAAA",
                   linewidth=2, linestyle="--", label="2025 Median")
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Sector {idx + 1}")
        ax.legend(fontsize=8)
        ax.invert_yaxis()

    fig.suptitle("Sector Time Comparison: 2026 Teams vs 2025 Median", fontsize=14, fontweight="bold")
    add_watermark(fig)
    fig.tight_layout()
    return fig


def generate_speed_traces(sessions_2026, sessions_2025, driver_2026=None, driver_2025=None):
    figures = {}

    lap_2026, _ = get_fastest_soft_lap(sessions_2026, driver=driver_2026)
    lap_2025, _ = get_fastest_soft_lap(sessions_2025, driver=driver_2025)

    if lap_2026 is None:
        lap_2026, _ = get_fastest_lap(sessions_2026, driver=driver_2026)
    if lap_2025 is None:
        lap_2025, _ = get_fastest_lap(sessions_2025, driver=driver_2025)

    if lap_2026 is None or lap_2025 is None:
        return figures

    tel_2026 = extract_telemetry(lap_2026)
    tel_2025 = extract_telemetry(lap_2025)

    if tel_2026 is None or tel_2025 is None:
        return figures

    tel_2026_interp = interpolate_to_common_distance(tel_2026)
    tel_2025_interp = interpolate_to_common_distance(tel_2025)

    if tel_2026_interp is None or tel_2025_interp is None:
        return figures

    drv_26 = driver_2026 or "Best"
    drv_25 = driver_2025 or "Best"
    label_26 = f"2026 ({drv_26})"
    label_25 = f"2025 ({drv_25})"

    figures["speed_comparison"] = plot_speed_comparison(
        tel_2026_interp, tel_2025_interp, label_26, label_25
    )
    figures["full_telemetry"] = plot_full_telemetry_comparison(
        tel_2026_interp, tel_2025_interp, label_26, label_25
    )
    figures["speed_delta"] = plot_speed_delta(tel_2026_interp, tel_2025_interp)

    return figures
