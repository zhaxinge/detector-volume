"""Streamlit frontend for Signal Detector QC AI.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import io
import os
import sys
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_engine.detector_checks import (
    DetectorFlag,
    FlagType,
    run_checks_for_all_detectors,
)
from qc_engine.volume_averaging import average_approach_volumes, network_average_volumes
from qc_engine.synchro_export import (
    compare_with_historical,
    export_network_synchro,
    export_synchro_volumes,
    identify_peak_hour,
)
from scripts.merge_folders import _read_map, merge_to_zip

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Signal Detector QC AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state ─────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    # single-intersection
    "si_daily_dfs": {},
    "si_det_dict": None,
    "si_flags": {},
    "si_bad_dates": [],
    "si_bad_detectors": [],
    "si_avg_df": None,
    "si_factors": {},
    "si_id": "INT-001",
    "si_ts_col": "timestamp",
    # network
    "net_data": {},       # {kits_id: {date: df}}
    "net_dicts": {},      # {kits_id: det_dict_df}
    "net_flags": {},      # {kits_id: {date: {det: [DetectorFlag]}}}
    "net_results": {},    # {kits_id: (avg_df, factors)}
    "net_bad_dates": {},  # {kits_id: [date]}
    "net_bad_detectors": {},  # {kits_id: [det]}
    "net_selected": None,
    "net_comparison": None,
    "net_ts_col": "timestamp",
    # ai
    "ai_summary": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── helpers ───────────────────────────────────────────────────────────────────

def _read(upload) -> pd.DataFrame:
    content = upload.read()
    name = upload.name
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _to_xlsx(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return buf.getvalue()


def _flag_counts(flags: dict[str, list[DetectorFlag]]) -> tuple[int, int]:
    """Return (critical_count, warning_count) for a {det: [flag]} dict."""
    crit = sum(1 for fs in flags.values() for f in fs if f.severity == "critical")
    warn = sum(1 for fs in flags.values() for f in fs if f.severity == "warning")
    return crit, warn


def _all_flags_for(kits_id: str) -> dict[str, list[DetectorFlag]]:
    """Merge flags across all dates for one intersection."""
    merged: dict[str, list[DetectorFlag]] = {}
    for date_flags in st.session_state.net_flags.get(kits_id, {}).values():
        for det, flist in date_flags.items():
            merged.setdefault(det, []).extend(flist)
    return merged


# ── volume plot ────────────────────────────────────────────────────────────────

_LINE_COLORS = [
    "#2563eb", "#16a34a", "#ca8a04", "#9333ea",
    "#0891b2", "#db2777", "#65a30d", "#ea580c",
]

def _approach_plot(
    daily_dfs: dict[str, pd.DataFrame],
    det_dict: pd.DataFrame,
    approach: str,
    bad_dates: list[str],
    bad_detectors: list[str],
    flags_by_date: dict[str, dict[str, list[DetectorFlag]]],
    ts_col: str = "timestamp",
) -> go.Figure:
    """Interactive plotly line chart — one line per date for a single approach."""
    app_dets = det_dict.loc[
        det_dict["movement_name"].str.strip().str.upper() == approach.strip().upper(),
        "KITS_det_name",
    ].dropna().tolist()
    active_dets = [d for d in app_dets if d not in bad_detectors]

    fig = go.Figure()
    avg_accum: list[pd.Series] = []

    for i, (date, df) in enumerate(sorted(daily_dfs.items())):
        avail = [d for d in active_dets if d in df.columns]
        if not avail:
            continue

        ts = pd.to_datetime(df[ts_col])
        x = ts.dt.strftime("%H:%M")
        vol = df[avail].apply(pd.to_numeric, errors="coerce").sum(axis=1)

        # Adjustment factor for this date
        total = len(app_dets)
        n_active = len(avail)
        if n_active < total and n_active > 0:
            vol = vol * (total / n_active)

        is_bad = date in bad_dates
        color = "#dc2626" if is_bad else _LINE_COLORS[i % len(_LINE_COLORS)]
        dash = "dot" if is_bad else "solid"
        opacity = 0.35 if is_bad else 0.8

        # Collect QC-flagged detectors for this date
        date_flags = flags_by_date.get(date, {})
        flagged_dets = [d for d in avail if date_flags.get(d)]
        flag_note = f" ⚠ {', '.join(flagged_dets)}" if flagged_dets else ""

        vol.index = range(len(vol))
        s = pd.Series(vol.values, index=x.values)

        fig.add_trace(go.Scatter(
            x=s.index,
            y=s.values,
            name=f"{date}{' [excl]' if is_bad else ''}{flag_note}",
            line=dict(color=color, dash=dash, width=1.5),
            opacity=opacity,
            mode="lines",
            hovertemplate=f"{date} %{{x}}: %{{y:.0f}} veh/15min<extra></extra>",
        ))

        if not is_bad:
            avg_accum.append(s)

    # Average line
    if avg_accum:
        avg_df = pd.concat(avg_accum, axis=1).mean(axis=1)
        fig.add_trace(go.Scatter(
            x=avg_df.index,
            y=avg_df.values,
            name="Average",
            line=dict(color="#0f172a", width=3),
            mode="lines",
            hovertemplate="Avg %{x}: %{y:.0f} veh/15min<extra></extra>",
        ))

    n_excl = len([d for d in active_dets if d not in app_dets])
    factor_note = ""
    if len(active_dets) < len(app_dets) and active_dets:
        f = len(app_dets) / len(active_dets)
        factor_note = f" (adj ×{f:.2f})"

    fig.update_layout(
        title=dict(text=f"{approach}{factor_note}", font=dict(size=14)),
        xaxis=dict(
            title="Time",
            tickmode="array",
            tickvals=[f"{h:02d}:00" for h in range(0, 24, 2)],
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            tickangle=45,
        ),
        yaxis=dict(title="veh / 15 min", rangemode="tozero"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
        margin=dict(l=40, r=20, t=40, b=120),
        height=360,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def _detector_plot(
    daily_dfs: dict[str, pd.DataFrame],
    detector: str,
    bad_dates: list[str],
    flags_by_date: dict[str, dict[str, list[DetectorFlag]]],
    ts_col: str = "timestamp",
) -> go.Figure:
    """Interactive plotly line chart — one line per date for a single detector."""
    fig = go.Figure()

    for i, (date, df) in enumerate(sorted(daily_dfs.items())):
        if detector not in df.columns:
            continue

        ts = pd.to_datetime(df[ts_col])
        x = ts.dt.strftime("%H:%M")
        vol = pd.to_numeric(df[detector], errors="coerce").fillna(0).values

        is_bad = date in bad_dates
        det_flags = flags_by_date.get(date, {}).get(detector, [])
        flag_icons = " ".join({
            FlagType.CONTINUOUS_ZERO: "⬛",
            FlagType.OSCILLATING: "〰",
            FlagType.EXTREME_SPIKE: "⚡",
            FlagType.STUCK_DETECTOR: "🔒",
            FlagType.TEMPORAL_IMPLAUSIBLE: "🌙",
            FlagType.INVERTED_PATTERN: "🔄",
            FlagType.SYSTEM_FLAG: "🚨",
        }.get(f.flag_type, "⚠") for f in det_flags)

        color = "#dc2626" if is_bad else _LINE_COLORS[i % len(_LINE_COLORS)]

        fig.add_trace(go.Scatter(
            x=list(x),
            y=vol.tolist(),
            name=f"{date}{' [excl]' if is_bad else ''}{' ' + flag_icons if flag_icons else ''}",
            line=dict(color=color, dash="dot" if is_bad else "solid", width=1.5),
            opacity=0.35 if is_bad else 0.85,
            mode="lines",
            hovertemplate=f"{date} %{{x}}: %{{y:.0f}} veh/15min<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=detector, font=dict(size=13)),
        xaxis=dict(
            tickmode="array",
            tickvals=[f"{h:02d}:00" for h in range(0, 24, 2)],
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            tickangle=45,
        ),
        yaxis=dict(title="veh / 15 min", rangemode="tozero"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
        margin=dict(l=40, r=20, t=35, b=120),
        height=320,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# ── intersection review panel ─────────────────────────────────────────────────

def _render_intersection_review(
    kits_id: str,
    daily_dfs: dict[str, pd.DataFrame],
    det_dict: pd.DataFrame,
    flags_by_date: dict[str, dict[str, list[DetectorFlag]]],
    ts_col: str,
    bad_dates_key: str,
    bad_dets_key: str,
    intersection_id: Optional[str] = None,
):
    """Render the full review panel for one intersection (shared by single + network)."""
    int_label = intersection_id or kits_id
    approaches = det_dict["movement_name"].dropna().unique().tolist()
    all_dets = det_dict["KITS_det_name"].dropna().tolist()
    all_dates = sorted(daily_dfs.keys())

    # ── QC flag summary bar ───────────────────────────────────────────────────
    all_flags: dict[str, list[DetectorFlag]] = {}
    for date_flags in flags_by_date.values():
        for det, flist in date_flags.items():
            all_flags.setdefault(det, []).extend(flist)

    crit, warn = _flag_counts(all_flags)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dates loaded", len(all_dates))
    c2.metric("Detectors", len(all_dets))
    c3.metric("🔴 Critical flags", crit)
    c4.metric("🟡 Warning flags", warn)

    # ── Exclusion controls ────────────────────────────────────────────────────
    with st.expander("🗑 Exclusion Controls", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            # Highlight dates that have critical flags
            flagged_dates = set()
            for date, date_flags in flags_by_date.items():
                if any(
                    f.severity == "critical"
                    for fs in date_flags.values()
                    for f in fs
                ):
                    flagged_dates.add(date)

            date_options = [
                f"{'⚠ ' if d in flagged_dates else ''}{d}" for d in all_dates
            ]
            date_map = dict(zip(date_options, all_dates))

            selected_excl_dates_labeled = st.multiselect(
                "Exclude dates (⚠ = QC-flagged):",
                options=date_options,
                default=[
                    f"{'⚠ ' if d in flagged_dates else ''}{d}"
                    for d in st.session_state[bad_dates_key]
                ],
                key=f"excl_dates_{kits_id}",
            )
            st.session_state[bad_dates_key] = [
                date_map[lbl] for lbl in selected_excl_dates_labeled
            ]

        with col_b:
            flagged_dets = {
                det for det, flist in all_flags.items() if flist
            }
            det_options = [
                f"{'⚠ ' if d in flagged_dets else ''}{d}" for d in sorted(all_dets)
            ]
            det_map = dict(zip(det_options, sorted(all_dets)))

            selected_excl_dets_labeled = st.multiselect(
                "Exclude detectors (⚠ = QC-flagged):",
                options=det_options,
                default=[
                    f"{'⚠ ' if d in flagged_dets else ''}{d}"
                    for d in st.session_state[bad_dets_key]
                ],
                key=f"excl_dets_{kits_id}",
            )
            st.session_state[bad_dets_key] = [
                det_map[lbl] for lbl in selected_excl_dets_labeled
            ]

    bad_dates = st.session_state[bad_dates_key]
    bad_detectors = st.session_state[bad_dets_key]

    # ── Chart tabs ────────────────────────────────────────────────────────────
    tab_approaches, tab_detectors, tab_flags = st.tabs(
        ["📈 By Approach", "📡 By Detector", "🔍 QC Flags"]
    )

    with tab_approaches:
        if not approaches:
            st.info("No detector dictionary — cannot show approach view.")
        else:
            # Two columns of charts
            n = len(approaches)
            pairs = [approaches[i : i + 2] for i in range(0, n, 2)]
            for pair in pairs:
                cols = st.columns(len(pair))
                for col, approach in zip(cols, pair):
                    with col:
                        fig = _approach_plot(
                            daily_dfs, det_dict, approach,
                            bad_dates, bad_detectors, flags_by_date, ts_col,
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"app_{kits_id}_{approach}")

    with tab_detectors:
        n = len(all_dets)
        pairs = [sorted(all_dets)[i : i + 2] for i in range(0, n, 2)]
        for pair in pairs:
            cols = st.columns(len(pair))
            for col, det in zip(cols, pair):
                with col:
                    has_flags = bool(all_flags.get(det))
                    flag_label = "⚠ " if has_flags else ""
                    with st.expander(f"{flag_label}{det}", expanded=has_flags):
                        fig = _detector_plot(
                            daily_dfs, det, bad_dates, flags_by_date, ts_col
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"det_{kits_id}_{det}")

    with tab_flags:
        rows = []
        for date, date_flags in sorted(flags_by_date.items()):
            for det, flist in date_flags.items():
                for f in flist:
                    rows.append({
                        "Date": date,
                        "Detector": det,
                        "Type": f.flag_type.value,
                        "Severity": f.severity,
                        "Description": f.description,
                    })
        if rows:
            fdf = pd.DataFrame(rows)

            def _color(row):
                bg = "#ffe5e5" if row["Severity"] == "critical" else "#fff8e1"
                return [f"background-color: {bg}"] * len(row)

            st.dataframe(fdf.style.apply(_color, axis=1), use_container_width=True)
            st.download_button(
                "⬇ Download QC Flags CSV",
                data=_to_csv(fdf),
                file_name=f"{int_label}_qc_flags.csv",
                mime="text/csv",
                key=f"dl_flags_{kits_id}",
            )
        else:
            st.success("✅ No QC flags detected for this intersection.")

    # ── Average & Export ──────────────────────────────────────────────────────
    st.divider()
    if st.button("⚡ Compute Average & Export Synchro", key=f"avg_btn_{kits_id}", type="primary"):
        with st.spinner("Averaging…"):
            try:
                avg_df, factors = average_approach_volumes(
                    daily_dfs=daily_dfs,
                    detector_dict=det_dict,
                    timestamp_col=ts_col,
                    bad_dates=bad_dates,
                    bad_detectors=bad_detectors,
                )
                synchro_df = export_synchro_volumes(avg_df, int_label, factors)
                am_s, am_e = identify_peak_hour(avg_df, "AM")
                pm_s, pm_e = identify_peak_hour(avg_df, "PM")

                st.success(
                    f"Averaged {len(all_dates) - len(bad_dates)} date(s). "
                    f"AM peak {am_s}–{am_e} · PM peak {pm_s}–{pm_e}"
                )
                if any(v != 1.0 for v in factors.values()):
                    with st.expander("Adjustment Factors"):
                        st.dataframe(
                            pd.DataFrame(factors.items(), columns=["Approach", "Factor"]),
                            use_container_width=True,
                        )

                col1, col2 = st.columns(2)
                col1.download_button(
                    "⬇ Synchro CSV",
                    data=_to_csv(synchro_df),
                    file_name=f"{int_label}_synchro.csv",
                    mime="text/csv",
                    key=f"dl_csv_{kits_id}",
                )
                col2.download_button(
                    "⬇ Synchro XLSX",
                    data=_to_xlsx(synchro_df),
                    file_name=f"{int_label}_synchro.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_xlsx_{kits_id}",
                )

                # Store back to network results if in network mode
                if bad_dates_key.startswith("net_"):
                    st.session_state.net_results[kits_id] = (avg_df, factors)

            except ValueError as exc:
                st.error(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🚦 Signal Detector QC AI")
page = st.sidebar.radio(
    "Workflow",
    ["Single Intersection", "Network Review", "Folder Organizer", "AI Assistant"],
    index=0,
)
ai_available = bool(os.getenv("ANTHROPIC_API_KEY"))
if not ai_available:
    st.sidebar.warning("ANTHROPIC_API_KEY not set — AI summaries disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Single Intersection
# ══════════════════════════════════════════════════════════════════════════════

if page == "Single Intersection":
    st.title("Single Intersection Analysis")

    with st.expander("📂 Upload Files", expanded=not bool(st.session_state.si_daily_dfs)):
        c1, c2, c3 = st.columns(3)
        with c1:
            vol_ups = st.file_uploader(
                "Volume files (one per day)",
                accept_multiple_files=True,
                type=["xlsx", "xls", "csv"],
                key="si_vols",
            )
        with c2:
            dict_up = st.file_uploader(
                "Detector dictionary  (KITS_ID · Phase · movement_name · KITS_det_name)",
                type=["xlsx", "xls", "csv"],
                key="si_dict",
            )
        with c3:
            st.session_state.si_id = st.text_input("Intersection ID", value=st.session_state.si_id)
            st.session_state.si_ts_col = st.text_input(
                "Timestamp column", value=st.session_state.si_ts_col
            )

        if vol_ups and dict_up and st.button("Load & Run QC", type="primary"):
            with st.spinner("Running QC…"):
                det_dict = _read(dict_up)
                st.session_state.si_det_dict = det_dict
                detectors = det_dict["KITS_det_name"].dropna().tolist()
                daily_dfs: dict[str, pd.DataFrame] = {}
                flags_by_date: dict[str, dict] = {}
                for up in vol_ups:
                    date_str = os.path.splitext(up.name)[0]
                    df = _read(up)
                    daily_dfs[date_str] = df
                    present = [d for d in detectors if d in df.columns]
                    flags_by_date[date_str] = run_checks_for_all_detectors(
                        df, present, st.session_state.si_ts_col, date_str
                    )
                st.session_state.si_daily_dfs = daily_dfs
                st.session_state.si_flags = flags_by_date
                st.session_state.si_bad_dates = []
                st.session_state.si_bad_detectors = []
                st.success(f"Loaded {len(daily_dfs)} day(s). QC complete.")

    if st.session_state.si_daily_dfs and st.session_state.si_det_dict is not None:
        _render_intersection_review(
            kits_id=st.session_state.si_id,
            daily_dfs=st.session_state.si_daily_dfs,
            det_dict=st.session_state.si_det_dict,
            flags_by_date=st.session_state.si_flags,
            ts_col=st.session_state.si_ts_col,
            bad_dates_key="si_bad_dates",
            bad_dets_key="si_bad_detectors",
            intersection_id=st.session_state.si_id,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Page: Folder Organizer
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Folder Organizer":
    st.title("📁 Folder Organizer")
    st.markdown(
        "Merge **April** and **October** volume files for each intersection into one "
        "organised folder, keyed by `{Synchro_ID}_{KITS_ID}`. "
        "The Synchro ID is taken from your April mapping. "
        "Download everything as a ZIP ready to drop into your project."
    )

    # ── Expected file naming ──────────────────────────────────────────────────
    with st.expander("📋 Expected file naming convention", expanded=False):
        st.markdown(
            """
| File type | Naming pattern | Example |
|---|---|---|
| April volume | `{KITS_ID}_{date}.xlsx` | `1001_2024-04-02.xlsx` |
| October volume | `{KITS_ID}_{date}.xlsx` | `1001_2024-10-01.xlsx` |
| Detector dictionary | `{KITS_ID}.xlsx` | `1001.xlsx` |
| Intersection map | any name | `intersection_map.csv` |

**Intersection map columns:** `KITS_ID` · `Synchro_ID`

**Output folder per intersection:** `{Synchro_ID}_{KITS_ID}/`
```
500_1001/
├── 1001_dict.xlsx
├── 1001_2024-04-02.xlsx
├── 1001_2024-04-03.xlsx
├── 1001_2024-10-01.xlsx
└── 1001_2024-10-07.xlsx
```
            """
        )

    # ── Upload section ────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        map_up = st.file_uploader(
            "Intersection map  (KITS_ID · Synchro_ID)",
            type=["xlsx", "xls", "csv"],
            key="fo_map",
        )
    with c2:
        dict_ups = st.file_uploader(
            "Detector dictionary files  (one per intersection, named `{KITS_ID}.xlsx`)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            key="fo_dicts",
        )

    c3, c4 = st.columns(2)
    with c3:
        april_ups = st.file_uploader(
            "April volume files  (`{KITS_ID}_{date}.xlsx`)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            key="fo_april",
        )
    with c4:
        oct_ups = st.file_uploader(
            "October volume files  (`{KITS_ID}_{date}.xlsx`)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            key="fo_oct",
        )

    have_inputs = map_up and (april_ups or oct_ups) and dict_ups

    if have_inputs and st.button("🗂 Organise & Build ZIP", type="primary"):
        with st.spinner("Organising files…"):
            try:
                # Parse intersection map
                int_map = _read_map(map_up)

                # Build in-memory file dicts: {filename: bytes}
                april_files = {up.name: up.read() for up in (april_ups or [])}
                oct_files   = {up.name: up.read() for up in (oct_ups   or [])}
                dict_files  = {up.name: up.read() for up in (dict_ups  or [])}

                zip_bytes = merge_to_zip(
                    april_files=april_files,
                    october_files=oct_files,
                    dict_files=dict_files,
                    intersection_map_df=int_map,
                )

                # Preview what was organised
                import zipfile as _zf, io as _io
                with _zf.ZipFile(_io.BytesIO(zip_bytes)) as zf:
                    names = zf.namelist()

                # Group by folder
                folders: dict[str, list[str]] = {}
                for n in sorted(names):
                    folder, fname = n.split("/", 1)
                    folders.setdefault(folder, []).append(fname)

                st.success(
                    f"Organised **{len(folders)}** intersection folder(s) · "
                    f"**{len(names)}** file(s) total"
                )

                # Preview table
                preview_rows = []
                for kits_id, row in int_map.iterrows():
                    kid = str(row["KITS_ID"])
                    sid = str(row["Synchro_ID"])
                    folder_name = f"{sid}_{kid}"
                    files_in = folders.get(folder_name, [])
                    n_april  = sum(1 for f in files_in if kid + "_" in f and "dict" not in f
                                   and any(m in f for m in ["04", "apr", "April"]))
                    n_oct    = sum(1 for f in files_in if kid + "_" in f and "dict" not in f
                                   and any(m in f for m in ["10", "oct", "Oct"]))
                    has_dict = any("dict" in f for f in files_in)
                    total    = len(files_in)
                    preview_rows.append({
                        "Folder": folder_name,
                        "KITS_ID": kid,
                        "Synchro_ID": sid,
                        "Total files": total,
                        "Dictionary": "✅" if has_dict else "❌",
                        "Status": "✅" if total > 1 else "⚠ check",
                    })

                prev_df = pd.DataFrame(preview_rows)

                def _fo_color(row):
                    return (
                        ["background-color: #ffe5e5"] * len(row)
                        if row["Status"] != "✅"
                        else ["background-color: #f0fdf4"] * len(row)
                    )

                st.dataframe(
                    prev_df.style.apply(_fo_color, axis=1),
                    use_container_width=True,
                )

                with st.expander("📂 Full file listing"):
                    for folder_name, flist in sorted(folders.items()):
                        st.markdown(f"**{folder_name}/**")
                        for f in sorted(flist):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{f}`", unsafe_allow_html=True)

                st.download_button(
                    "⬇ Download organised ZIP",
                    data=zip_bytes,
                    file_name="organised_intersections.zip",
                    mime="application/zip",
                )

            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                raise

    elif not have_inputs and (april_ups or oct_ups or dict_ups or map_up):
        missing = []
        if not map_up:       missing.append("intersection map")
        if not dict_ups:     missing.append("dictionary files")
        if not april_ups and not oct_ups:
            missing.append("at least one set of volume files (April or October)")
        st.warning(f"Still needed: {', '.join(missing)}")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Network Review
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Network Review":
    st.title("Network Review Dashboard")

    # ── Upload ────────────────────────────────────────────────────────────────
    with st.expander(
        "📂 Upload Network Files",
        expanded=not bool(st.session_state.net_results),
    ):
        st.caption(
            "**Volume files:** name as `<KITS_ID>_<date>.xlsx`  ·  "
            "**Dictionary files:** name as `<KITS_ID>.xlsx`"
        )
        c1, c2 = st.columns(2)
        with c1:
            net_vols = st.file_uploader(
                "Volume files (all intersections)",
                accept_multiple_files=True,
                type=["xlsx", "xls", "csv"],
                key="net_vols_up",
            )
        with c2:
            net_dicts = st.file_uploader(
                "Detector dictionaries (one per intersection)",
                accept_multiple_files=True,
                type=["xlsx", "xls", "csv"],
                key="net_dicts_up",
            )

        hist_up = st.file_uploader(
            "Historical Synchro baseline (optional — for comparison)",
            type=["xlsx", "xls", "csv"],
            key="hist_up",
        )

        c1b, c2b, c3b = st.columns(3)
        with c1b:
            st.session_state.net_ts_col = st.text_input(
                "Timestamp column", value=st.session_state.net_ts_col
            )
        with c2b:
            warn_pct = st.number_input("Warning threshold (%)", value=20, min_value=1)
        with c3b:
            crit_pct = st.number_input("Critical threshold (%)", value=50, min_value=1)

        if net_vols and net_dicts and st.button("Load & Run Network QC", type="primary"):
            with st.spinner("Processing network…"):
                network_data: dict[str, dict[str, pd.DataFrame]] = {}
                for up in net_vols:
                    stem = os.path.splitext(up.name)[0]
                    parts = stem.split("_", 1)
                    kits_id = parts[0]
                    date_str = parts[1] if len(parts) > 1 else stem
                    network_data.setdefault(kits_id, {})[date_str] = _read(up)

                network_dicts_map: dict[str, pd.DataFrame] = {}
                for up in net_dicts:
                    kits_id = os.path.splitext(up.name)[0]
                    network_dicts_map[kits_id] = _read(up)

                # Run QC per intersection
                net_flags: dict[str, dict] = {}
                for kits_id, daily_dfs in network_data.items():
                    dd = network_dicts_map.get(kits_id)
                    if dd is None:
                        continue
                    detectors = dd["KITS_det_name"].dropna().tolist()
                    net_flags[kits_id] = {}
                    for date, df in daily_dfs.items():
                        present = [d for d in detectors if d in df.columns]
                        net_flags[kits_id][date] = run_checks_for_all_detectors(
                            df, present, st.session_state.net_ts_col, date
                        )

                try:
                    results = network_average_volumes(
                        network_data=network_data,
                        network_dicts=network_dicts_map,
                        timestamp_col=st.session_state.net_ts_col,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    results = {}

                st.session_state.net_data = network_data
                st.session_state.net_dicts = network_dicts_map
                st.session_state.net_flags = net_flags
                st.session_state.net_results = results
                st.session_state.net_bad_dates = {k: [] for k in network_data}
                st.session_state.net_bad_detectors = {k: [] for k in network_data}
                st.session_state.net_selected = None

                if hist_up and results:
                    hist_df = _read(hist_up)
                    combined = export_network_synchro(results)
                    st.session_state.net_comparison = compare_with_historical(
                        combined, hist_df,
                        warn_pct=float(warn_pct),
                        critical_pct=float(crit_pct),
                    )

                st.success(
                    f"Loaded {len(network_data)} intersection(s), "
                    f"{sum(len(v) for v in network_data.values())} date(s) total."
                )

    # ── Network overview table ────────────────────────────────────────────────
    if st.session_state.net_data:
        st.subheader("🗺 Intersection Review Queue")

        review_rows = []
        for kits_id in sorted(st.session_state.net_data.keys()):
            all_flags = _all_flags_for(kits_id)
            crit, warn = _flag_counts(all_flags)
            n_dates = len(st.session_state.net_data[kits_id])
            n_bad = len(st.session_state.net_bad_dates.get(kits_id, []))
            status = "🔴 Needs Review" if crit else "🟡 Check" if warn else "✅ Clean"
            review_rows.append({
                "KITS_ID": kits_id,
                "Dates": n_dates,
                "Excluded": n_bad,
                "🔴 Critical": crit,
                "🟡 Warnings": warn,
                "Status": status,
            })

        overview_df = pd.DataFrame(review_rows).sort_values(
            ["🔴 Critical", "🟡 Warnings"], ascending=False
        )

        # Color critical rows
        def _status_color(row):
            if row["🔴 Critical"] > 0:
                return ["background-color: #ffe5e5"] * len(row)
            if row["🟡 Warnings"] > 0:
                return ["background-color: #fff8e1"] * len(row)
            return ["background-color: #f0fdf4"] * len(row)

        st.dataframe(
            overview_df.style.apply(_status_color, axis=1),
            use_container_width=True,
            height=min(40 + 35 * len(overview_df), 400),
        )

        # Historical comparison summary (if available)
        if st.session_state.net_comparison is not None:
            comp = st.session_state.net_comparison
            n_crit = (comp["severity"] == "critical").sum()
            n_warn = (comp["severity"] == "warning").sum()
            with st.expander(
                f"📊 Historical Comparison — {n_crit} critical, {n_warn} warning changes",
                expanded=bool(n_crit),
            ):
                def _comp_color(row):
                    bg = "#ffe5e5" if row["severity"] == "critical" else "#fff8e1"
                    return [f"background-color: {bg}"] * len(row)

                st.dataframe(
                    comp.style.apply(_comp_color, axis=1),
                    use_container_width=True,
                )
                st.download_button(
                    "⬇ Download Comparison CSV",
                    data=_to_csv(comp),
                    file_name="network_comparison.csv",
                    mime="text/csv",
                )

        # Network-wide Synchro export
        if st.session_state.net_results:
            combined = export_network_synchro(st.session_state.net_results)
            c1, c2 = st.columns(2)
            c1.download_button(
                "⬇ Full Network Synchro CSV",
                data=_to_csv(combined),
                file_name="network_synchro.csv",
                mime="text/csv",
            )
            c2.download_button(
                "⬇ Full Network Synchro XLSX",
                data=_to_xlsx(combined),
                file_name="network_synchro.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.divider()

        # ── Intersection selector & detail ────────────────────────────────────
        st.subheader("🔎 Review Intersection")

        # Order selector: critical first
        ordered_ids = [r["KITS_ID"] for _, r in overview_df.iterrows()]
        sel = st.selectbox(
            "Select intersection (sorted by severity):",
            options=ordered_ids,
            index=ordered_ids.index(st.session_state.net_selected)
            if st.session_state.net_selected in ordered_ids
            else 0,
            key="net_selector",
        )
        st.session_state.net_selected = sel

        if sel and sel in st.session_state.net_data:
            row = overview_df[overview_df["KITS_ID"] == sel].iloc[0]
            status_icon = "🔴" if row["🔴 Critical"] else "🟡" if row["🟡 Warnings"] else "✅"
            st.markdown(f"### {status_icon} {sel}")

            det_dict = st.session_state.net_dicts.get(sel)
            if det_dict is None:
                st.warning(f"No detector dictionary found for {sel}.")
            else:
                # Ensure per-intersection bad_dates / bad_detectors entries exist
                if sel not in st.session_state.net_bad_dates:
                    st.session_state.net_bad_dates[sel] = []
                if sel not in st.session_state.net_bad_detectors:
                    st.session_state.net_bad_detectors[sel] = []

                # Proxy session state keys scoped to this intersection
                _bd_key = f"_bd_{sel}"
                _bdet_key = f"_bdet_{sel}"
                if _bd_key not in st.session_state:
                    st.session_state[_bd_key] = st.session_state.net_bad_dates[sel]
                if _bdet_key not in st.session_state:
                    st.session_state[_bdet_key] = st.session_state.net_bad_detectors[sel]

                _render_intersection_review(
                    kits_id=sel,
                    daily_dfs=st.session_state.net_data[sel],
                    det_dict=det_dict,
                    flags_by_date=st.session_state.net_flags.get(sel, {}),
                    ts_col=st.session_state.net_ts_col,
                    bad_dates_key=_bd_key,
                    bad_dets_key=_bdet_key,
                    intersection_id=sel,
                )
                # Sync back to net_bad_dates / net_bad_detectors
                st.session_state.net_bad_dates[sel] = st.session_state[_bd_key]
                st.session_state.net_bad_detectors[sel] = st.session_state[_bdet_key]


# ══════════════════════════════════════════════════════════════════════════════
# Page: AI Assistant
# ══════════════════════════════════════════════════════════════════════════════

elif page == "AI Assistant":
    st.title("🤖 AI Engineering Assistant")

    if not ai_available:
        st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file and restart.")
    else:
        parts: list[str] = []

        # Pull in QC flags from whichever analysis has been run
        flag_sources = []
        if st.session_state.si_flags:
            flag_sources.append(("Single", st.session_state.si_flags))
        for kits_id, date_flags in st.session_state.net_flags.items():
            flag_sources.append((kits_id, date_flags))

        for label, flags_by_date in flag_sources:
            lines = []
            for date, det_flags in flags_by_date.items():
                for det, flist in det_flags.items():
                    for f in flist:
                        lines.append(
                            f"{label} | {date} | {det} | {f.flag_type.value} | {f.description}"
                        )
            if lines:
                parts.append("QC FLAGS:\n" + "\n".join(lines))

        if st.session_state.net_comparison is not None:
            parts.append(
                "NETWORK COMPARISON:\n"
                + st.session_state.net_comparison.to_string(index=False)
            )

        context = "\n\n".join(parts) or "No analysis run yet in this session."

        with st.expander("Session Context (auto-populated)", expanded=False):
            st.text(context[:4000] + ("…" if len(context) > 4000 else ""))

        extra = st.text_area("Append additional context (optional)", height=80)
        if extra:
            context += "\n\n" + extra

        question = st.text_area("Engineering question", height=80)
        if st.button("Ask Claude", type="primary") and question.strip():
            with st.spinner("Thinking…"):
                try:
                    from ai_layer.summarizer import QCSummarizer
                    answer = QCSummarizer().answer_engineering_question(context, question)
                    st.subheader("Answer")
                    st.info(answer)
                except Exception as exc:
                    st.error(str(exc))
