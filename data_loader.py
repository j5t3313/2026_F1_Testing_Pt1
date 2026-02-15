import fastf1
import pandas as pd
from packaging.version import Version
from config import (
    CACHE_DIR, YEAR, TEST_NUMBER, TEST_DAYS,
    BASELINE_YEAR, BASELINE_TEST_NUMBER, BASELINE_TEST_DAYS,
    INLAP_THRESHOLD_FACTOR,
)

MIN_FASTF1_VERSION = "3.8.0"


def setup():
    if Version(fastf1.__version__) < Version(MIN_FASTF1_VERSION):
        raise RuntimeError(
            f"FastF1 {MIN_FASTF1_VERSION}+ required for 2026 testing data. "
            f"Installed: {fastf1.__version__}. Run: pip install fastf1 --upgrade"
        )
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_session(year, test_number, day):
    session = fastf1.get_testing_session(year, test_number, day)
    session.load(telemetry=True, weather=False)
    return session


def load_test(year, test_number, days):
    sessions = []
    frames = []

    for day in days:
        session = load_session(year, test_number, day)
        sessions.append(session)

        laps = session.laps.copy()
        laps["Day"] = day
        laps["Year"] = year
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        frames.append(laps)

    combined = pd.concat(frames, ignore_index=True)
    return sessions, combined


def load_2026():
    return load_test(YEAR, TEST_NUMBER, TEST_DAYS)


def load_2025():
    return load_test(BASELINE_YEAR, BASELINE_TEST_NUMBER, BASELINE_TEST_DAYS)


def filter_representative(laps):
    valid = laps.dropna(subset=["LapTime"]).copy()
    if valid.empty:
        return valid

    best = valid["LapTimeSeconds"].min()
    threshold = best * INLAP_THRESHOLD_FACTOR
    return valid[valid["LapTimeSeconds"] <= threshold].copy()


def filter_accurate(laps):
    if "IsAccurate" in laps.columns:
        return laps[laps["IsAccurate"] == True].copy()
    return laps.copy()


def get_clean_laps(laps):
    filtered = filter_representative(laps)
    filtered = filter_accurate(filtered)
    return filtered
