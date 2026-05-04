"""Multi-day volume averaging with bad-date and bad-detector exclusion."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_approach_detectors(
    detector_dict: pd.DataFrame,
    approach: str,
    movement_col: str = "movement_name",
    det_col: str = "KITS_det_name",
) -> list[str]:
    """Return detector names associated with a given approach/movement."""
    mask = detector_dict[movement_col].str.strip().str.upper() == approach.strip().upper()
    return detector_dict.loc[mask, det_col].dropna().tolist()


def calculate_adjustment_factor(
    all_detectors: list[str],
    valid_detectors: list[str],
    detector_dict: pd.DataFrame,
    approach: str,
    movement_col: str = "movement_name",
    det_col: str = "KITS_det_name",
) -> float:
    """Return a scale factor compensating for excluded detectors on an approach.

    If an approach has 3 lane-detectors and 1 is excluded, the factor is 3/2
    so the remaining 2-lane average is scaled up to approximate 3-lane volume.
    """
    approach_dets = _get_approach_detectors(detector_dict, approach, movement_col, det_col)
    if not approach_dets:
        return 1.0

    total = len(approach_dets)
    usable = [d for d in approach_dets if d in valid_detectors]
    n_usable = len(usable)

    if n_usable == 0:
        return 1.0  # No data available; caller should flag this
    return total / n_usable


def _interval_key(ts: pd.Timestamp) -> str:
    """Return HH:MM string for a timestamp, used as the 15-min key."""
    return ts.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Core averaging
# ---------------------------------------------------------------------------

def average_approach_volumes(
    daily_dfs: dict[str, pd.DataFrame],
    detector_dict: pd.DataFrame,
    timestamp_col: str = "timestamp",
    bad_dates: Optional[list[str]] = None,
    bad_detectors: Optional[list[str]] = None,
    movement_col: str = "movement_name",
    det_col: str = "KITS_det_name",
    phase_col: str = "Phase",
    kits_id_col: str = "KITS_ID",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Average 15-min volumes across valid days at the approach level.

    Args:
        daily_dfs: Mapping of date-string → single-day DataFrame.
        detector_dict: Columns KITS_ID, Phase, movement_name, KITS_det_name.
        timestamp_col: Column in daily DataFrames containing datetime values.
        bad_dates: List of date strings to exclude entirely.
        bad_detectors: List of detector names to exclude from all dates.
        movement_col / det_col / phase_col / kits_id_col: Column aliases.

    Returns:
        (averaged_df, adjustment_factors)
        averaged_df: 96-row DataFrame (15-min intervals) with approach columns.
        adjustment_factors: {approach: scale_factor}
    """
    bad_dates = set(bad_dates or [])
    bad_detectors = set(bad_detectors or [])

    valid_dates = {d: df for d, df in daily_dfs.items() if d not in bad_dates}
    if not valid_dates:
        raise ValueError("No valid dates remain after exclusions.")

    approaches = detector_dict[movement_col].dropna().unique().tolist()
    all_detectors = detector_dict[det_col].dropna().tolist()
    valid_detectors = [d for d in all_detectors if d not in bad_detectors]

    # Build a canonical 15-min interval index from the first valid day
    reference_df = next(iter(valid_dates.values()))
    ref_ts = pd.to_datetime(reference_df[timestamp_col])
    interval_keys = [_interval_key(ts) for ts in ref_ts]

    averaged: dict[str, list[float]] = {app: [] for app in approaches}
    adjustment_factors: dict[str, float] = {}

    for app in approaches:
        app_dets = _get_approach_detectors(detector_dict, app, movement_col, det_col)
        usable_dets = [d for d in app_dets if d in valid_detectors]

        adjustment_factors[app] = calculate_adjustment_factor(
            all_detectors=app_dets,
            valid_detectors=usable_dets,
            detector_dict=detector_dict,
            approach=app,
            movement_col=movement_col,
            det_col=det_col,
        )

        # Collect interval-level totals per approach per valid day
        day_series: list[pd.Series] = []
        for date, df in valid_dates.items():
            avail = [d for d in usable_dets if d in df.columns]
            if not avail:
                continue
            ts_aligned = pd.to_datetime(df[timestamp_col])
            keys = [_interval_key(ts) for ts in ts_aligned]
            day_vol = df[avail].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            day_vol.index = keys
            day_series.append(day_vol)

        if not day_series:
            averaged[app] = [0.0] * len(interval_keys)
            continue

        combined = pd.concat(day_series, axis=1)
        mean_vol = combined.mean(axis=1)

        # Reindex to canonical interval order; fill missing with adjacent-interval mean
        mean_vol = mean_vol.reindex(interval_keys)
        mean_vol = mean_vol.interpolate(method="linear", limit_direction="both")

        # Apply adjustment factor for excluded detectors
        scaled = mean_vol * adjustment_factors[app]
        averaged[app] = scaled.tolist()

    result = pd.DataFrame(averaged, index=interval_keys)
    result.index.name = "interval"
    result = result.reset_index()
    return result, adjustment_factors


# ---------------------------------------------------------------------------
# Missing-interval interpolation (within a single-day series)
# ---------------------------------------------------------------------------

def interpolate_from_peer_days(
    target_series: pd.Series,
    peer_series_list: list[pd.Series],
    interval_keys: list[str],
) -> pd.Series:
    """Fill NaN/zero intervals in target_series using the mean of peer days.

    Peer series are assumed to share the same index as target_series.
    """
    if not peer_series_list:
        return target_series

    peer_df = pd.concat(peer_series_list, axis=1)
    peer_mean = peer_df.mean(axis=1)

    result = target_series.copy()
    missing = (result.isna()) | (result == 0)
    result[missing] = peer_mean[missing]
    return result


# ---------------------------------------------------------------------------
# Multi-intersection network averaging
# ---------------------------------------------------------------------------

def network_average_volumes(
    network_data: dict[str, dict[str, pd.DataFrame]],
    network_dicts: dict[str, pd.DataFrame],
    bad_dates: Optional[dict[str, list[str]]] = None,
    bad_detectors: Optional[dict[str, list[str]]] = None,
    **kwargs,
) -> dict[str, tuple[pd.DataFrame, dict[str, float]]]:
    """Run approach-level averaging for every intersection in a network.

    Args:
        network_data: {kits_id: {date: df}}
        network_dicts: {kits_id: detector_dict_df}
        bad_dates: {kits_id: [date, ...]}  (optional, defaults to empty)
        bad_detectors: {kits_id: [det, ...]}  (optional, defaults to empty)

    Returns:
        {kits_id: (averaged_df, adjustment_factors)}
    """
    results: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}
    bad_dates = bad_dates or {}
    bad_detectors = bad_detectors or {}

    for kits_id, daily_dfs in network_data.items():
        if kits_id not in network_dicts:
            continue
        results[kits_id] = average_approach_volumes(
            daily_dfs=daily_dfs,
            detector_dict=network_dicts[kits_id],
            bad_dates=bad_dates.get(kits_id),
            bad_detectors=bad_detectors.get(kits_id),
            **kwargs,
        )
    return results
