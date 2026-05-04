"""Streamlit frontend for Signal Detector QC AI.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import io
import json
import os
import sys

import pandas as pd
import streamlit as st

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_engine.detector_checks import (
    FlagType,
    run_checks_for_all_detectors,
)
from qc_engine.volume_averaging import (
    average_approach_volumes,
    network_average_volumes,
)
from qc_engine.synchro_export import (
    compare_with_historical,
    export_network_synchro,
    export_synchro_volumes,
    identify_peak_hour,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Signal Detector QC AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "daily_dfs": {},
        "det_dict": None,
        "flags_by_date": {},
        "averaged_df": None,
        "adjustment_factors": {},
        "bad_dates": [],
        "bad_detectors": [],
        "ai_summary": "",
        "network_results": {},
        "network_dicts": {},
        "network_data": {},
        "comparison_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_upload(upload) -> pd.DataFrame:
    name = upload.name
    content = upload.read()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


def _flag_color(severity: str) -> str:
    return "🔴" if severity == "critical" else "🟡"


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🚦 Signal Detector QC AI")
page = st.sidebar.radio(
    "Workflow",
    ["Single Intersection", "Network Analysis", "AI Assistant"],
    index=0,
)

ai_available = bool(os.getenv("ANTHROPIC_API_KEY"))
if not ai_available:
    st.sidebar.warning("ANTHROPIC_API_KEY not set — AI summaries disabled.")

# ---------------------------------------------------------------------------
# ── Single Intersection ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if page == "Single Intersection":
    st.title("Single Intersection Analysis")

    # ── Step 1: Upload ──────────────────────────────────────────────────────
    with st.expander("📂 Step 1 — Upload Files", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Volume Files (one per day)")
            vol_uploads = st.file_uploader(
                "Upload daily detector volume files (Excel or CSV)",
                accept_multiple_files=True,
                type=["xlsx", "xls", "csv"],
                key="vol_uploads",
            )
        with col2:
            st.subheader("Detector Dictionary")
            dict_upload = st.file_uploader(
                "Upload detector dictionary (KITS_ID, Phase, movement_name, KITS_det_name)",
                type=["xlsx", "xls", "csv"],
                key="dict_upload",
            )

        timestamp_col = st.text_input("Timestamp column name", value="timestamp")
        intersection_id = st.text_input("Intersection ID", value="INT-001")

        if vol_uploads and dict_upload:
            if st.button("Load and Run QC", type="primary"):
                with st.spinner("Running QC checks…"):
                    det_dict = _read_upload(dict_upload)
                    st.session_state.det_dict = det_dict
                    detectors = det_dict["KITS_det_name"].dropna().tolist()

                    daily_dfs: dict[str, pd.DataFrame] = {}
                    flags_by_date: dict[str, dict] = {}

                    for up in vol_uploads:
                        date_str = os.path.splitext(up.name)[0]
                        df = _read_upload(up)
                        daily_dfs[date_str] = df
                        present = [d for d in detectors if d in df.columns]
                        flags_by_date[date_str] = run_checks_for_all_detectors(
                            df, present, timestamp_col, date_str
                        )

                    st.session_state.daily_dfs = daily_dfs
                    st.session_state.flags_by_date = flags_by_date
                    st.session_state.bad_dates = []
                    st.session_state.bad_detectors = []
                    st.session_state.averaged_df = None
                    st.success(f"Loaded {len(daily_dfs)} day(s). QC complete.")

    # ── Step 2: Review QC Flags ─────────────────────────────────────────────
    if st.session_state.flags_by_date:
        with st.expander("🔍 Step 2 — Review QC Flags", expanded=True):
            all_flag_rows = []
            for date, flags_by_det in st.session_state.flags_by_date.items():
                for det, flags in flags_by_det.items():
                    for f in flags:
                        all_flag_rows.append(
                            {
                                "Date": date,
                                "Detector": det,
                                "Flag Type": f.flag_type.value,
                                "Severity": f.severity,
                                "Description": f.description,
                            }
                        )

            if all_flag_rows:
                flag_df = pd.DataFrame(all_flag_rows)
                critical = flag_df[flag_df["Severity"] == "critical"]
                warning = flag_df[flag_df["Severity"] == "warning"]

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Flags", len(flag_df))
                col2.metric("🔴 Critical", len(critical))
                col3.metric("🟡 Warning", len(warning))

                # Color rows by severity
                def _color_row(row):
                    color = "#ffe5e5" if row["Severity"] == "critical" else "#fff8e1"
                    return [f"background-color: {color}"] * len(row)

                st.dataframe(
                    flag_df.style.apply(_color_row, axis=1),
                    use_container_width=True,
                )

                # Download flags
                st.download_button(
                    "⬇ Download Flags CSV",
                    data=_to_csv_bytes(flag_df),
                    file_name="qc_flags.csv",
                    mime="text/csv",
                )
            else:
                st.success("✅ No QC flags detected — all detectors appear clean.")

    # ── Step 3: Exclude Bad Dates / Detectors ──────────────────────────────
    if st.session_state.daily_dfs:
        with st.expander("🗑 Step 3 — Exclude Bad Dates / Detectors", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                dates = list(st.session_state.daily_dfs.keys())
                bad_dates = st.multiselect(
                    "Exclude dates (entire day):",
                    options=dates,
                    default=st.session_state.bad_dates,
                )
            with col2:
                if st.session_state.det_dict is not None:
                    all_dets = st.session_state.det_dict["KITS_det_name"].dropna().tolist()
                    bad_dets = st.multiselect(
                        "Exclude detectors (all dates):",
                        options=all_dets,
                        default=st.session_state.bad_detectors,
                    )
                else:
                    bad_dets = []

            st.session_state.bad_dates = bad_dates
            st.session_state.bad_detectors = bad_dets

    # ── Step 4: Average & Export ────────────────────────────────────────────
    if st.session_state.daily_dfs and st.session_state.det_dict is not None:
        with st.expander("📊 Step 4 — Average Volumes & Export to Synchro", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                peak_period = st.selectbox("Peak period for identification", ["AM", "PM", "Both"])

            if st.button("Average Volumes", type="primary"):
                with st.spinner("Averaging…"):
                    try:
                        avg_df, factors = average_approach_volumes(
                            daily_dfs=st.session_state.daily_dfs,
                            detector_dict=st.session_state.det_dict,
                            timestamp_col=timestamp_col,
                            bad_dates=st.session_state.bad_dates,
                            bad_detectors=st.session_state.bad_detectors,
                        )
                        st.session_state.averaged_df = avg_df
                        st.session_state.adjustment_factors = factors
                        st.success("Averaging complete.")
                    except ValueError as exc:
                        st.error(str(exc))

            if st.session_state.averaged_df is not None:
                avg_df = st.session_state.averaged_df
                factors = st.session_state.adjustment_factors

                # Adjustment factors table
                if any(v != 1.0 for v in factors.values()):
                    st.subheader("Adjustment Factors")
                    st.info(
                        "Factors > 1.0 indicate excluded detectors; volumes were scaled up."
                    )
                    fac_df = pd.DataFrame(
                        [(k, f"{v:.3f}") for k, v in factors.items()],
                        columns=["Approach", "Factor"],
                    )
                    st.dataframe(fac_df, use_container_width=True)

                # Peak hour
                periods = ["AM", "PM"] if peak_period == "Both" else [peak_period]
                peak_hours: dict[str, tuple[str, str]] = {}
                for p in periods:
                    try:
                        s, e = identify_peak_hour(avg_df, p)
                        peak_hours[p] = (s, e)
                        st.write(f"**{p} Peak Hour:** {s} – {e}")
                    except Exception:
                        pass

                # Preview
                st.subheader("Averaged 15-min Approach Volumes (preview)")
                st.dataframe(avg_df.head(20), use_container_width=True)

                # Synchro export
                synchro_df = export_synchro_volumes(avg_df, intersection_id, factors)
                col1, col2 = st.columns(2)
                col1.download_button(
                    "⬇ Download Synchro CSV",
                    data=_to_csv_bytes(synchro_df),
                    file_name=f"{intersection_id}_synchro.csv",
                    mime="text/csv",
                )
                col2.download_button(
                    "⬇ Download Synchro XLSX",
                    data=_to_excel_bytes(synchro_df),
                    file_name=f"{intersection_id}_synchro.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # AI Summary
                if ai_available:
                    if st.button("🤖 Generate AI Engineering Summary"):
                        with st.spinner("Consulting Claude…"):
                            try:
                                from ai_layer.summarizer import QCSummarizer
                                from qc_engine.detector_checks import DetectorFlag, FlagType

                                flags: dict[str, list[DetectorFlag]] = {}
                                for date, fbd in st.session_state.flags_by_date.items():
                                    for det, dflags in fbd.items():
                                        flags.setdefault(det, []).extend(dflags)

                                summarizer = QCSummarizer()
                                summary = summarizer.summarize_intersection(
                                    intersection_id, flags, factors, peak_hours
                                )
                                st.session_state.ai_summary = summary
                            except Exception as exc:
                                st.error(f"AI error: {exc}")

                if st.session_state.ai_summary:
                    st.subheader("AI Engineering Summary")
                    st.info(st.session_state.ai_summary)


# ---------------------------------------------------------------------------
# ── Network Analysis ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

elif page == "Network Analysis":
    st.title("Network-Level Volume Analysis")

    with st.expander("📂 Step 1 — Upload Network Files", expanded=True):
        st.markdown(
            "**Volume file naming:** `<KITS_ID>_<date>.xlsx` (e.g. `1001_2024-03-05.xlsx`)\n\n"
            "**Dictionary file naming:** `<KITS_ID>.xlsx` (e.g. `1001.xlsx`)"
        )
        col1, col2 = st.columns(2)
        with col1:
            net_vol_uploads = st.file_uploader(
                "Upload volume files for all intersections",
                accept_multiple_files=True,
                type=["xlsx", "xls", "csv"],
                key="net_vol",
            )
        with col2:
            net_dict_uploads = st.file_uploader(
                "Upload detector dictionaries (one per intersection)",
                accept_multiple_files=True,
                type=["xlsx", "xls", "csv"],
                key="net_dict",
            )

        hist_upload = st.file_uploader(
            "Upload historical Synchro baseline (optional)",
            type=["xlsx", "xls", "csv"],
            key="hist_upload",
        )

        net_timestamp_col = st.text_input(
            "Timestamp column", value="timestamp", key="net_ts"
        )
        warn_pct = st.number_input("Warning threshold (%)", value=20, min_value=1)
        critical_pct = st.number_input("Critical threshold (%)", value=50, min_value=1)

        if net_vol_uploads and net_dict_uploads:
            if st.button("Run Network Analysis", type="primary"):
                with st.spinner("Processing network…"):
                    network_data: dict[str, dict[str, pd.DataFrame]] = {}
                    for up in net_vol_uploads:
                        stem = os.path.splitext(up.name)[0]
                        parts = stem.split("_", 1)
                        kits_id = parts[0]
                        date_str = parts[1] if len(parts) > 1 else stem
                        network_data.setdefault(kits_id, {})[date_str] = _read_upload(up)

                    network_dicts: dict[str, pd.DataFrame] = {}
                    for up in net_dict_uploads:
                        kits_id = os.path.splitext(up.name)[0]
                        network_dicts[kits_id] = _read_upload(up)

                    try:
                        results = network_average_volumes(
                            network_data=network_data,
                            network_dicts=network_dicts,
                            timestamp_col=net_timestamp_col,
                        )
                        st.session_state.network_results = results
                        st.session_state.network_data = network_data
                        st.session_state.network_dicts = network_dicts
                        st.success(f"Averaged {len(results)} intersection(s).")
                    except Exception as exc:
                        st.error(str(exc))

                    if hist_upload and st.session_state.network_results:
                        hist_df = _read_upload(hist_upload)
                        combined = export_network_synchro(st.session_state.network_results)
                        comp_df = compare_with_historical(
                            combined, hist_df,
                            warn_pct=float(warn_pct),
                            critical_pct=float(critical_pct),
                        )
                        st.session_state.comparison_df = comp_df

    if st.session_state.network_results:
        with st.expander("📊 Network Results", expanded=True):
            combined = export_network_synchro(st.session_state.network_results)
            st.subheader(f"Combined Synchro volumes — {len(st.session_state.network_results)} intersections")
            st.dataframe(combined.head(30), use_container_width=True)

            col1, col2 = st.columns(2)
            col1.download_button(
                "⬇ Download Network Synchro CSV",
                data=_to_csv_bytes(combined),
                file_name="network_synchro.csv",
                mime="text/csv",
            )
            col2.download_button(
                "⬇ Download Network Synchro XLSX",
                data=_to_excel_bytes(combined),
                file_name="network_synchro.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    if st.session_state.comparison_df is not None and not st.session_state.comparison_df.empty:
        with st.expander("⚠ Volume Change Flags vs Historical Baseline", expanded=True):
            comp_df = st.session_state.comparison_df
            critical = comp_df[comp_df["severity"] == "critical"]
            warning = comp_df[comp_df["severity"] == "warning"]

            col1, col2 = st.columns(2)
            col1.metric("🔴 Critical changes", len(critical))
            col2.metric("🟡 Warning changes", len(warning))

            def _color_comp(row):
                color = "#ffe5e5" if row["severity"] == "critical" else "#fff8e1"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                comp_df.style.apply(_color_comp, axis=1),
                use_container_width=True,
            )

            if ai_available and st.button("🤖 Generate Network AI Summary"):
                with st.spinner("Consulting Claude…"):
                    try:
                        from ai_layer.summarizer import QCSummarizer
                        summarizer = QCSummarizer()
                        summary = summarizer.summarize_network(
                            comp_df.to_dict(orient="records"),
                            len(st.session_state.network_results),
                        )
                        st.info(summary)
                    except Exception as exc:
                        st.error(str(exc))


# ---------------------------------------------------------------------------
# ── AI Assistant ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

elif page == "AI Assistant":
    st.title("🤖 AI Engineering Assistant")

    if not ai_available:
        st.error(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your `.env` file and restart the app."
        )
    else:
        st.markdown(
            "Ask engineering questions based on the analysis context from the current session."
        )

        context_parts = []
        if st.session_state.flags_by_date:
            all_flags = []
            for date, fbd in st.session_state.flags_by_date.items():
                for det, flags in fbd.items():
                    for f in flags:
                        all_flags.append(f"{date} | {det} | {f.flag_type.value} | {f.description}")
            context_parts.append("QC FLAGS:\n" + "\n".join(all_flags))

        if st.session_state.adjustment_factors:
            lines = [f"{k}: {v:.3f}" for k, v in st.session_state.adjustment_factors.items()]
            context_parts.append("ADJUSTMENT FACTORS:\n" + "\n".join(lines))

        if st.session_state.comparison_df is not None:
            context_parts.append(
                "NETWORK COMPARISON:\n"
                + st.session_state.comparison_df.to_string(index=False)
            )

        context = "\n\n".join(context_parts) or "No analysis has been run yet in this session."

        with st.expander("Session Context (auto-populated)", expanded=False):
            st.text(context)

        custom_context = st.text_area(
            "Append additional context (optional)", height=100
        )
        if custom_context:
            context += "\n\n" + custom_context

        question = st.text_area("Your engineering question", height=80)

        if st.button("Ask Claude", type="primary") and question.strip():
            with st.spinner("Thinking…"):
                try:
                    from ai_layer.summarizer import QCSummarizer
                    summarizer = QCSummarizer()
                    answer = summarizer.answer_engineering_question(context, question)
                    st.subheader("Answer")
                    st.info(answer)
                except Exception as exc:
                    st.error(str(exc))
