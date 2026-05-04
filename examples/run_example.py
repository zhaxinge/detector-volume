"""Example script: single-intersection QC and Synchro export.

Usage:
    python examples/run_example.py

Generates a synthetic 3-day dataset, runs all QC checks, averages volumes,
and writes a Synchro CSV to examples/data/output_synchro.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_engine.detector_checks import run_checks_for_all_detectors
from qc_engine.volume_averaging import average_approach_volumes
from qc_engine.synchro_export import export_synchro_volumes, identify_peak_hour

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
INTERSECTION_ID = "DEMO-001"
DETECTORS = ["DET_NBT1", "DET_NBT2", "DET_EBL1", "DET_SBT1", "DET_WBL1"]
DATES = ["2024-03-05", "2024-03-06", "2024-03-07"]

# Detector dictionary
DETECTOR_DICT = pd.DataFrame(
    [
        {"KITS_ID": "DEMO", "Phase": 2, "movement_name": "NBT", "KITS_det_name": "DET_NBT1"},
        {"KITS_ID": "DEMO", "Phase": 2, "movement_name": "NBT", "KITS_det_name": "DET_NBT2"},
        {"KITS_ID": "DEMO", "Phase": 4, "movement_name": "EBL", "KITS_det_name": "DET_EBL1"},
        {"KITS_ID": "DEMO", "Phase": 6, "movement_name": "SBT", "KITS_det_name": "DET_SBT1"},
        {"KITS_ID": "DEMO", "Phase": 8, "movement_name": "WBL", "KITS_det_name": "DET_WBL1"},
    ]
)


def _day_profile(rng, n=96) -> np.ndarray:
    """Generate a realistic 15-min volume profile for a through movement."""
    profile = np.zeros(n)
    for i in range(n):
        hour = i / 4
        # Morning peak 7-9 AM, evening peak 4-6 PM
        if 7 <= hour < 9:
            base = 90
        elif 16 <= hour < 18:
            base = 85
        elif 9 <= hour < 16:
            base = 50
        elif 5 <= hour < 7 or 18 <= hour < 20:
            base = 30
        else:
            base = 5
        profile[i] = max(0, base + rng.normal(0, base * 0.1))
    return profile.round().astype(int)


def make_daily_df(date: str, bad_det: str = None) -> pd.DataFrame:
    ts = pd.date_range(date, periods=96, freq="15min")
    data = {"timestamp": ts}
    for det in DETECTORS:
        vals = _day_profile(RNG)
        if det == bad_det:
            # Inject a continuous-zero failure for 4 hours (16 intervals)
            vals[20:36] = 0
        data[det] = vals
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Signal Detector QC — {INTERSECTION_ID} ===\n")

    # Build daily DataFrames (inject a fault on day 2 for DET_EBL1)
    daily_dfs = {
        DATES[0]: make_daily_df(DATES[0]),
        DATES[1]: make_daily_df(DATES[1], bad_det="DET_EBL1"),
        DATES[2]: make_daily_df(DATES[2]),
    }

    # ── QC ──────────────────────────────────────────────────────────────────
    print("Running QC checks…")
    flagged_detectors: set[str] = set()

    for date, df in daily_dfs.items():
        flags = run_checks_for_all_detectors(
            df, DETECTORS, timestamp_col="timestamp", date=date
        )
        for det, det_flags in flags.items():
            for f in det_flags:
                print(
                    f"  [{f.severity.upper()}] {date} | {det} | "
                    f"{f.flag_type.value} | {f.description}"
                )
                flagged_detectors.add(det)

    if not flagged_detectors:
        print("  No flags detected.")

    # ── Decide exclusions ────────────────────────────────────────────────────
    # In a real workflow an engineer reviews the flags above.
    # Here we exclude the known bad day for DEMO purposes.
    bad_dates: list[str] = []
    bad_detectors: list[str] = ["DET_EBL1"]  # has 4-hour zero run on day 2
    print(f"\nExcluding detectors: {bad_detectors}")
    print(f"Excluding dates:     {bad_dates or 'none'}\n")

    # ── Average ─────────────────────────────────────────────────────────────
    avg_df, factors = average_approach_volumes(
        daily_dfs=daily_dfs,
        detector_dict=DETECTOR_DICT,
        timestamp_col="timestamp",
        bad_dates=bad_dates,
        bad_detectors=bad_detectors,
    )
    print("Approach-level averaged volumes (first 8 intervals):")
    print(avg_df.head(8).to_string(index=False))

    # ── Adjustment factors ───────────────────────────────────────────────────
    print("\nAdjustment factors:")
    for mvt, factor in factors.items():
        note = "" if factor == 1.0 else f"  ← {factor:.2f}x scale (partial coverage)"
        print(f"  {mvt}: {factor:.3f}{note}")

    # ── Peak hours ────────────────────────────────────────────────────────────
    am_start, am_end = identify_peak_hour(avg_df, "AM")
    pm_start, pm_end = identify_peak_hour(avg_df, "PM")
    print(f"\nAM peak hour: {am_start} – {am_end}")
    print(f"PM peak hour: {pm_start} – {pm_end}")

    # ── Synchro export ────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "output_synchro.csv")
    synchro_df = export_synchro_volumes(
        avg_df, INTERSECTION_ID, factors, output_path=out_path
    )
    print(f"\nSynchro file written to: {out_path}")
    print(synchro_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
