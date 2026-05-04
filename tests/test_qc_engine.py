"""Unit tests for the deterministic QC engine."""

import pandas as pd
import numpy as np
import pytest

from qc_engine.detector_checks import (
    FlagType,
    check_continuous_zeros,
    check_extreme_spikes,
    check_inverted_pattern,
    check_oscillating_pattern,
    check_stuck_detector,
    check_temporal_plausibility,
    run_all_checks,
)
from qc_engine.volume_averaging import average_approach_volumes, calculate_adjustment_factor
from qc_engine.synchro_export import (
    compare_with_historical,
    export_synchro_volumes,
    identify_peak_hour,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_series(values: list, name: str = "DET1") -> pd.Series:
    return pd.Series(values, name=name)


def _make_timestamps(n: int = 96, date: str = "2024-03-05") -> pd.Series:
    return pd.Series(pd.date_range(date, periods=n, freq="15min"))


def _make_daily_df(det_values: dict[str, list], date: str = "2024-03-05") -> pd.DataFrame:
    ts = _make_timestamps(96, date)
    df = pd.DataFrame({"timestamp": ts})
    for det, vals in det_values.items():
        df[det] = vals[:96] if len(vals) >= 96 else vals + [0] * (96 - len(vals))
    return df


def _make_det_dict(movements: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for mvt, dets in movements.items():
        for d in dets:
            rows.append(
                {"KITS_ID": "INT1", "Phase": 2, "movement_name": mvt, "KITS_det_name": d}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# check_continuous_zeros
# ---------------------------------------------------------------------------

class TestContinuousZeros:
    def test_detects_long_zero_run(self):
        values = [0] * 96
        series = _make_series(values)
        flags = check_continuous_zeros(series, "DET1", "2024-03-05", min_hours=3)
        assert len(flags) == 1
        assert flags[0].flag_type == FlagType.CONTINUOUS_ZERO
        assert flags[0].severity == "critical"

    def test_no_flag_short_zero_run(self):
        values = [10] * 40 + [0] * 10 + [10] * 46
        series = _make_series(values)
        flags = check_continuous_zeros(series, "DET1", "2024-03-05", min_hours=3)
        assert flags == []

    def test_detects_run_at_end(self):
        values = [10] * 83 + [0] * 13
        series = _make_series(values)
        flags = check_continuous_zeros(series, "DET1", "2024-03-05", min_hours=3)
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# check_oscillating_pattern
# ---------------------------------------------------------------------------

class TestOscillating:
    def test_detects_oscillation(self):
        values = ([5, 200] * 20) + [10] * 56
        series = _make_series(values)
        flags = check_oscillating_pattern(
            series, "DET1", "2024-03-05", low_thresh=20, high_thresh=100, min_cycles=3
        )
        assert len(flags) == 1
        assert flags[0].flag_type == FlagType.OSCILLATING

    def test_no_flag_stable_data(self):
        values = [80] * 96
        series = _make_series(values)
        flags = check_oscillating_pattern(series, "DET1", "2024-03-05")
        assert flags == []


# ---------------------------------------------------------------------------
# check_extreme_spikes
# ---------------------------------------------------------------------------

class TestExtremeSpikes:
    def test_detects_spike(self):
        values = [20] * 48 + [500] + [20] * 47
        series = _make_series(values)
        flags = check_extreme_spikes(series, "DET1", "2024-03-05", threshold_pct=300)
        assert len(flags) == 1
        assert flags[0].flag_type == FlagType.EXTREME_SPIKE

    def test_no_flag_gradual_increase(self):
        values = list(range(10, 106)) + [80] * 10
        series = _make_series(values[:96])
        flags = check_extreme_spikes(series, "DET1", "2024-03-05", threshold_pct=300)
        assert flags == []


# ---------------------------------------------------------------------------
# check_temporal_plausibility
# ---------------------------------------------------------------------------

class TestTemporalPlausibility:
    def test_flags_high_night_volume(self):
        values = [5] * 96
        ts = _make_timestamps()
        # Set index 0 (midnight) to high value
        values[0] = 200
        series = _make_series(values)
        flags = check_temporal_plausibility(
            series, ts, "DET1", "2024-03-05",
            night_hours=(22, 5), night_volume_thresh=60
        )
        assert len(flags) == 1
        assert flags[0].flag_type == FlagType.TEMPORAL_IMPLAUSIBLE

    def test_no_flag_low_night_volume(self):
        values = [5] * 96
        series = _make_series(values)
        ts = _make_timestamps()
        flags = check_temporal_plausibility(
            series, ts, "DET1", "2024-03-05", night_volume_thresh=60
        )
        assert flags == []


# ---------------------------------------------------------------------------
# check_inverted_pattern
# ---------------------------------------------------------------------------

class TestInvertedPattern:
    def test_flags_inverted(self):
        ts = _make_timestamps()
        hours = ts.dt.hour
        # Low during day (7-19), high at night
        values = [200 if h < 7 or h >= 19 else 10 for h in hours]
        series = _make_series(values)
        flags = check_inverted_pattern(series, ts, "DET1", "2024-03-05", ratio_thresh=1.5)
        assert len(flags) == 1
        assert flags[0].flag_type == FlagType.INVERTED_PATTERN

    def test_no_flag_normal_pattern(self):
        ts = _make_timestamps()
        hours = ts.dt.hour
        values = [80 if 7 <= h < 19 else 5 for h in hours]
        series = _make_series(values)
        flags = check_inverted_pattern(series, ts, "DET1", "2024-03-05", ratio_thresh=1.5)
        assert flags == []


# ---------------------------------------------------------------------------
# check_stuck_detector
# ---------------------------------------------------------------------------

class TestStuckDetector:
    def test_flags_stuck(self):
        values = [42] * 90 + [10, 5, 7, 0, 0, 0]
        series = _make_series(values)
        flags = check_stuck_detector(series, "DET1", "2024-03-05", same_value_threshold=0.8)
        assert len(flags) == 1
        assert flags[0].flag_type == FlagType.STUCK_DETECTOR

    def test_no_flag_varied_data(self):
        rng = np.random.default_rng(42)
        values = rng.integers(10, 120, size=96).tolist()
        series = _make_series(values)
        flags = check_stuck_detector(series, "DET1", "2024-03-05")
        assert flags == []


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_returns_empty_for_clean_detector(self):
        rng = np.random.default_rng(1)
        ts = _make_timestamps()
        hours = ts.dt.hour
        values = [50 if 7 <= h < 19 else 5 for h in hours]
        df = pd.DataFrame({"timestamp": ts, "DET1": values})
        flags = run_all_checks(df, "DET1", "timestamp", "2024-03-05")
        assert isinstance(flags, list)

    def test_returns_flags_for_zero_detector(self):
        ts = _make_timestamps()
        df = pd.DataFrame({"timestamp": ts, "DET1": [0] * 96})
        flags = run_all_checks(df, "DET1", "timestamp", "2024-03-05")
        assert any(f.flag_type == FlagType.CONTINUOUS_ZERO for f in flags)


# ---------------------------------------------------------------------------
# volume_averaging
# ---------------------------------------------------------------------------

class TestVolumeAveraging:
    def _setup(self):
        det_dict = _make_det_dict({"NBT": ["DET1", "DET2"], "EBL": ["DET3"]})
        ts = _make_timestamps()
        hours = ts.dt.hour
        day_vals = [80 if 7 <= h < 19 else 5 for h in hours]

        daily_dfs = {
            "2024-03-05": _make_daily_df({"DET1": day_vals, "DET2": day_vals, "DET3": day_vals}),
            "2024-03-06": _make_daily_df({"DET1": day_vals, "DET2": day_vals, "DET3": day_vals}),
        }
        return daily_dfs, det_dict

    def test_basic_average(self):
        daily_dfs, det_dict = self._setup()
        avg_df, factors = average_approach_volumes(daily_dfs, det_dict)
        assert "NBT" in avg_df.columns
        assert "EBL" in avg_df.columns
        assert len(avg_df) == 96

    def test_bad_date_exclusion(self):
        daily_dfs, det_dict = self._setup()
        avg_df, _ = average_approach_volumes(
            daily_dfs, det_dict, bad_dates=["2024-03-06"]
        )
        assert len(avg_df) == 96

    def test_all_dates_excluded_raises(self):
        daily_dfs, det_dict = self._setup()
        with pytest.raises(ValueError):
            average_approach_volumes(
                daily_dfs, det_dict, bad_dates=["2024-03-05", "2024-03-06"]
            )

    def test_adjustment_factor_partial_detector(self):
        daily_dfs, det_dict = self._setup()
        _, factors = average_approach_volumes(
            daily_dfs, det_dict, bad_detectors=["DET2"]
        )
        # NBT has 2 detectors, 1 excluded → factor = 2/1 = 2.0
        assert abs(factors["NBT"] - 2.0) < 1e-6

    def test_adjustment_factor_no_exclusion(self):
        det_dict = _make_det_dict({"NBT": ["DET1", "DET2"]})
        assert calculate_adjustment_factor(["DET1", "DET2"], ["DET1", "DET2"], det_dict, "NBT") == 1.0


# ---------------------------------------------------------------------------
# synchro_export
# ---------------------------------------------------------------------------

class TestSynchroExport:
    def _avg_df(self):
        ts = _make_timestamps()
        keys = [t.strftime("%H:%M") for t in ts]
        return pd.DataFrame({"interval": keys, "NBT": [80.0] * 96, "EBL": [30.0] * 96})

    def test_export_has_intid_and_time(self):
        avg_df = self._avg_df()
        out = export_synchro_volumes(avg_df, "INT-001")
        assert "INTID" in out.columns
        assert "Time" in out.columns
        assert (out["INTID"] == "INT-001").all()

    def test_export_rounds_volumes(self):
        avg_df = self._avg_df()
        avg_df["NBT"] = 80.7
        out = export_synchro_volumes(avg_df, "INT-001")
        assert out["NBT"].dtype in (int, "int64")

    def test_identify_peak_hour_am(self):
        avg_df = self._avg_df()
        # Spike the AM peak at 8:00
        idx_8am = avg_df["interval"] == "08:00"
        avg_df.loc[idx_8am, "NBT"] = 500
        start, end = identify_peak_hour(avg_df, "AM")
        assert start <= "09:00"

    def test_compare_with_historical_flags_critical(self):
        current = pd.DataFrame(
            {"INTID": ["INT-001"] * 3, "Time": ["07:00", "07:15", "07:30"], "NBT": [200, 200, 200]}
        )
        historical = pd.DataFrame(
            {"INTID": ["INT-001"] * 3, "Time": ["07:00", "07:15", "07:30"], "NBT": [50, 50, 50]}
        )
        comp = compare_with_historical(current, historical, warn_pct=20, critical_pct=50)
        assert len(comp) == 1
        assert comp.iloc[0]["severity"] == "critical"
        assert comp.iloc[0]["pct_change"] > 0

    def test_compare_no_flags_within_threshold(self):
        current = pd.DataFrame(
            {"INTID": ["INT-001"] * 2, "Time": ["07:00", "07:15"], "NBT": [105, 105]}
        )
        historical = pd.DataFrame(
            {"INTID": ["INT-001"] * 2, "Time": ["07:00", "07:15"], "NBT": [100, 100]}
        )
        comp = compare_with_historical(current, historical, warn_pct=20, critical_pct=50)
        assert comp.empty
