"""Deterministic QC checks for KITS traffic detector data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class FlagType(Enum):
    SYSTEM_FLAG = "system_flag"
    CONTINUOUS_ZERO = "continuous_zero"
    OSCILLATING = "oscillating"
    EXTREME_SPIKE = "extreme_spike"
    TEMPORAL_IMPLAUSIBLE = "temporal_implausible"
    INVERTED_PATTERN = "inverted_pattern"
    STUCK_DETECTOR = "stuck_detector"


SEVERITY = {
    FlagType.SYSTEM_FLAG: "critical",
    FlagType.CONTINUOUS_ZERO: "critical",
    FlagType.OSCILLATING: "critical",
    FlagType.EXTREME_SPIKE: "warning",
    FlagType.TEMPORAL_IMPLAUSIBLE: "warning",
    FlagType.INVERTED_PATTERN: "warning",
    FlagType.STUCK_DETECTOR: "critical",
}


@dataclass
class DetectorFlag:
    detector: str
    date: str
    flag_type: FlagType
    description: str
    severity: str
    affected_intervals: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "date": self.date,
            "flag_type": self.flag_type.value,
            "description": self.description,
            "severity": self.severity,
            "affected_intervals": self.affected_intervals,
        }


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_system_flags(
    df: pd.DataFrame,
    detector: str,
    date: str,
    flag_columns: Optional[list[str]] = None,
    flag_keywords: tuple[str, ...] = ("invalid", "max out", "missing", "error"),
) -> list[DetectorFlag]:
    """Detect KITS system flags embedded in flag columns or string values."""
    flags: list[DetectorFlag] = []

    # If the detector column itself contains string flags
    if df[detector].dtype == object:
        mask = df[detector].str.lower().str.contains(
            "|".join(flag_keywords), na=False
        )
        if mask.any():
            affected = df.index[mask].tolist()
            flags.append(
                DetectorFlag(
                    detector=detector,
                    date=date,
                    flag_type=FlagType.SYSTEM_FLAG,
                    description=f"System flag values present in {detector} column",
                    severity=SEVERITY[FlagType.SYSTEM_FLAG],
                    affected_intervals=affected,
                )
            )
        return flags

    # Check dedicated flag columns (e.g. "<detector>_flag")
    if flag_columns:
        for fc in flag_columns:
            if fc not in df.columns:
                continue
            mask = df[fc].astype(str).str.lower().str.contains(
                "|".join(flag_keywords), na=False
            )
            if mask.any():
                affected = df.index[mask].tolist()
                flags.append(
                    DetectorFlag(
                        detector=detector,
                        date=date,
                        flag_type=FlagType.SYSTEM_FLAG,
                        description=f"System flag in column '{fc}': {df.loc[mask, fc].unique().tolist()}",
                        severity=SEVERITY[FlagType.SYSTEM_FLAG],
                        affected_intervals=affected,
                    )
                )
    return flags


def check_continuous_zeros(
    series: pd.Series,
    detector: str,
    date: str,
    min_hours: float = 3.0,
    interval_minutes: int = 15,
) -> list[DetectorFlag]:
    """Detect runs of consecutive zero readings >= min_hours."""
    min_intervals = int(min_hours * 60 / interval_minutes)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)

    flags: list[DetectorFlag] = []
    run_start: Optional[int] = None
    run_len = 0

    for i, val in enumerate(numeric):
        if val == 0:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len >= min_intervals:
                flags.append(
                    DetectorFlag(
                        detector=detector,
                        date=date,
                        flag_type=FlagType.CONTINUOUS_ZERO,
                        description=(
                            f"{run_len} consecutive zero intervals "
                            f"({run_len * interval_minutes / 60:.1f} h)"
                        ),
                        severity=SEVERITY[FlagType.CONTINUOUS_ZERO],
                        affected_intervals=list(range(run_start, run_start + run_len)),
                    )
                )
            run_start = None
            run_len = 0

    # Catch run that ends at the last interval
    if run_len >= min_intervals and run_start is not None:
        flags.append(
            DetectorFlag(
                detector=detector,
                date=date,
                flag_type=FlagType.CONTINUOUS_ZERO,
                description=(
                    f"{run_len} consecutive zero intervals "
                    f"({run_len * interval_minutes / 60:.1f} h)"
                ),
                severity=SEVERITY[FlagType.CONTINUOUS_ZERO],
                affected_intervals=list(range(run_start, run_start + run_len)),
            )
        )
    return flags


def check_oscillating_pattern(
    series: pd.Series,
    detector: str,
    date: str,
    low_thresh: int = 20,
    high_thresh: int = 100,
    min_cycles: int = 3,
) -> list[DetectorFlag]:
    """Detect rapid alternation between very low and very high values."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2 * min_cycles:
        return []

    low_mask = numeric < low_thresh
    high_mask = numeric >= high_thresh

    # Count how many times we alternate low→high or high→low
    alternations = 0
    affected: list[int] = []
    prev_state: Optional[str] = None

    for idx, val in zip(numeric.index, numeric):
        if val < low_thresh:
            state = "low"
        elif val >= high_thresh:
            state = "high"
        else:
            state = "mid"
            prev_state = state
            continue

        if prev_state and prev_state != state and state != "mid":
            alternations += 1
            affected.append(idx)

        prev_state = state

    if alternations >= min_cycles * 2:
        return [
            DetectorFlag(
                detector=detector,
                date=date,
                flag_type=FlagType.OSCILLATING,
                description=(
                    f"Oscillating pattern detected: {alternations} alternations "
                    f"between <{low_thresh} and >={high_thresh} veh/interval"
                ),
                severity=SEVERITY[FlagType.OSCILLATING],
                affected_intervals=affected,
            )
        ]
    return []


def check_extreme_spikes(
    series: pd.Series,
    detector: str,
    date: str,
    threshold_pct: float = 300.0,
    min_base: int = 5,
) -> list[DetectorFlag]:
    """Detect >threshold_pct% change between consecutive 15-min intervals."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    prev = numeric.shift(1)

    # Avoid division by zero; only compare where previous value is meaningful
    base = prev.where(prev >= min_base, np.nan)
    pct_change = ((numeric - prev) / base * 100).abs()

    spike_mask = pct_change > threshold_pct
    if not spike_mask.any():
        return []

    affected = numeric.index[spike_mask].tolist()
    max_spike = pct_change[spike_mask].max()
    return [
        DetectorFlag(
            detector=detector,
            date=date,
            flag_type=FlagType.EXTREME_SPIKE,
            description=(
                f"Extreme volume spike detected: max {max_spike:.0f}% change "
                f"(threshold {threshold_pct}%)"
            ),
            severity=SEVERITY[FlagType.EXTREME_SPIKE],
            affected_intervals=affected,
        )
    ]


def check_temporal_plausibility(
    series: pd.Series,
    timestamps: pd.Series,
    detector: str,
    date: str,
    night_hours: tuple[int, int] = (22, 5),
    night_volume_thresh: int = 60,
) -> list[DetectorFlag]:
    """Flag high volumes during deep-night hours (e.g., 10 PM – 5 AM)."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    hours = pd.to_datetime(timestamps).dt.hour

    night_start, night_end = night_hours
    if night_start > night_end:
        night_mask = (hours >= night_start) | (hours < night_end)
    else:
        night_mask = (hours >= night_start) & (hours < night_end)

    suspicious = night_mask & (numeric > night_volume_thresh)
    if not suspicious.any():
        return []

    affected = series.index[suspicious].tolist()
    peak_val = numeric[suspicious].max()
    return [
        DetectorFlag(
            detector=detector,
            date=date,
            flag_type=FlagType.TEMPORAL_IMPLAUSIBLE,
            description=(
                f"High volume ({peak_val:.0f} veh/15min) during night hours "
                f"({night_start}:00–{night_end}:00), threshold={night_volume_thresh}"
            ),
            severity=SEVERITY[FlagType.TEMPORAL_IMPLAUSIBLE],
            affected_intervals=affected,
        )
    ]


def check_inverted_pattern(
    series: pd.Series,
    timestamps: pd.Series,
    detector: str,
    date: str,
    day_hours: tuple[int, int] = (7, 19),
    ratio_thresh: float = 1.5,
) -> list[DetectorFlag]:
    """Flag when total night volume significantly exceeds total day volume."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    hours = pd.to_datetime(timestamps).dt.hour

    day_start, day_end = day_hours
    day_mask = (hours >= day_start) & (hours < day_end)

    day_vol = numeric[day_mask].sum()
    night_vol = numeric[~day_mask].sum()

    if day_vol == 0:
        return []

    ratio = night_vol / day_vol
    if ratio >= ratio_thresh:
        return [
            DetectorFlag(
                detector=detector,
                date=date,
                flag_type=FlagType.INVERTED_PATTERN,
                description=(
                    f"Night volume ({night_vol:.0f}) is {ratio:.1f}x day volume ({day_vol:.0f}); "
                    f"pattern appears inverted (threshold ratio={ratio_thresh})"
                ),
                severity=SEVERITY[FlagType.INVERTED_PATTERN],
                affected_intervals=[],
            )
        ]
    return []


def check_stuck_detector(
    series: pd.Series,
    detector: str,
    date: str,
    same_value_threshold: float = 0.80,
    min_nonzero_count: int = 12,
) -> list[DetectorFlag]:
    """Flag detectors where >=80% of non-zero readings share the same value."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    nonzero = numeric[numeric > 0]

    if len(nonzero) < min_nonzero_count:
        return []

    mode_val = nonzero.mode().iloc[0]
    mode_frac = (nonzero == mode_val).sum() / len(nonzero)

    if mode_frac >= same_value_threshold:
        return [
            DetectorFlag(
                detector=detector,
                date=date,
                flag_type=FlagType.STUCK_DETECTOR,
                description=(
                    f"Stuck detector: {mode_frac:.0%} of non-zero readings = {mode_val:.0f} "
                    f"(threshold {same_value_threshold:.0%})"
                ),
                severity=SEVERITY[FlagType.STUCK_DETECTOR],
                affected_intervals=[],
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Unified runner
# ---------------------------------------------------------------------------

def run_all_checks(
    df: pd.DataFrame,
    detector: str,
    timestamp_col: str = "timestamp",
    date: str = "",
    flag_columns: Optional[list[str]] = None,
) -> list[DetectorFlag]:
    """Run all QC checks for a single detector column in a daily DataFrame.

    Args:
        df: DataFrame with one row per 15-min interval.
        detector: Name of the detector column.
        timestamp_col: Name of the timestamp column.
        date: Date string used in flag metadata.
        flag_columns: Optional list of KITS flag columns to inspect.

    Returns:
        List of DetectorFlag instances (may be empty if detector is clean).
    """
    if detector not in df.columns:
        return []

    series = df[detector]
    timestamps = df[timestamp_col] if timestamp_col in df.columns else pd.Series(dtype="object")

    flags: list[DetectorFlag] = []
    flags += check_system_flags(df, detector, date, flag_columns)
    flags += check_continuous_zeros(series, detector, date)
    flags += check_oscillating_pattern(series, detector, date)
    flags += check_extreme_spikes(series, detector, date)

    if len(timestamps) == len(series) and not timestamps.empty:
        flags += check_temporal_plausibility(series, timestamps, detector, date)
        flags += check_inverted_pattern(series, timestamps, detector, date)

    flags += check_stuck_detector(series, detector, date)
    return flags


def run_checks_for_all_detectors(
    df: pd.DataFrame,
    detectors: list[str],
    timestamp_col: str = "timestamp",
    date: str = "",
    flag_columns: Optional[list[str]] = None,
) -> dict[str, list[DetectorFlag]]:
    """Run all checks across multiple detectors in a single-day DataFrame."""
    return {det: run_all_checks(df, det, timestamp_col, date, flag_columns) for det in detectors}
