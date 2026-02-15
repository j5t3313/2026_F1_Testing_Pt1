from pathlib import Path

CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")

YEAR = 2026
TEST_NUMBER = 1
TEST_DAYS = [1, 2, 3]

BASELINE_YEAR = 2025
BASELINE_TEST_NUMBER = 1
BASELINE_TEST_DAYS = [1, 2, 3]

INLAP_THRESHOLD_FACTOR = 1.3
LONG_RUN_MIN_LAPS = 10

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Red Bull": "#3671C6",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Aston Martin Racing": "#229971",
    "Alpine": "#FF87BC",
    "Alpine F1 Team": "#FF87BC",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
    "Racing Bulls": "#6692FF",
    "RB": "#6692FF",
    "Williams": "#64C4FF",
    "Williams Racing": "#64C4FF",
    "Audi": "#2D826D",
    "Audi F1 Team": "#2D826D",
    "Kick Sauber": "#52E252",
    "Cadillac": "#C0C0C0",
    "Cadillac Racing": "#C0C0C0",
    "Cadillac F1 Team": "#C0C0C0",
}

COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC633",
    "HARD": "#CCCCCC",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
    "UNKNOWN": "#888888",
    "TEST_UNKNOWN": "#888888",
}

FALLBACK_COLOR = "#888888"

FIGURE_DPI = 200
FIGURE_WIDTH = 14
FIGURE_HEIGHT = 8
TITLE_SIZE = 16
LABEL_SIZE = 12
TICK_SIZE = 10
