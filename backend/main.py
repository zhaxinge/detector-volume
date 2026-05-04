"""FastAPI backend for Signal Detector QC AI."""

from __future__ import annotations

import io
import json
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from qc_engine.detector_checks import run_checks_for_all_detectors
from qc_engine.volume_averaging import average_approach_volumes, network_average_volumes
from qc_engine.synchro_export import (
    compare_with_historical,
    export_network_synchro,
    export_synchro_volumes,
    identify_peak_hour,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Signal Detector QC AI",
    description="Deterministic traffic detector QC and Synchro volume export API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_excel_or_csv(upload: UploadFile) -> pd.DataFrame:
    content = upload.file.read()
    name = upload.filename or ""
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


def _df_to_stream(df: pd.DataFrame, filename: str) -> StreamingResponse:
    buf = io.BytesIO()
    if filename.endswith(".xlsx"):
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
    else:
        df.to_csv(buf, index=False)
    buf.seek(0)
    media = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if filename.endswith(".xlsx")
        else "text/csv"
    )
    return StreamingResponse(
        buf,
        media_type=media,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FlagSummary(BaseModel):
    detector: str
    date: str
    flag_type: str
    description: str
    severity: str


class QCResponse(BaseModel):
    intersection_id: str
    flags: list[FlagSummary]
    n_clean: int
    n_flagged: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/qc/single", response_model=QCResponse)
async def qc_single_intersection(
    intersection_id: str = Form(...),
    timestamp_col: str = Form("timestamp"),
    volume_files: list[UploadFile] = File(...),
    detector_dict_file: UploadFile = File(...),
):
    """Run QC on one or more daily volume files for a single intersection."""
    det_dict = _read_excel_or_csv(detector_dict_file)
    if "KITS_det_name" not in det_dict.columns:
        raise HTTPException(400, "detector_dict must have a 'KITS_det_name' column")

    detectors = det_dict["KITS_det_name"].dropna().tolist()
    all_flags: list[FlagSummary] = []

    for vf in volume_files:
        date_str = os.path.splitext(vf.filename or "unknown")[0]
        df = _read_excel_or_csv(vf)
        present = [d for d in detectors if d in df.columns]
        flags_by_det = run_checks_for_all_detectors(df, present, timestamp_col, date_str)

        for det, flags in flags_by_det.items():
            for f in flags:
                all_flags.append(FlagSummary(**f.to_dict()))

    n_flagged = len({f.detector for f in all_flags})
    n_clean = len(detectors) - n_flagged

    return QCResponse(
        intersection_id=intersection_id,
        flags=all_flags,
        n_clean=max(n_clean, 0),
        n_flagged=n_flagged,
    )


@app.post("/api/average/single")
async def average_single_intersection(
    intersection_id: str = Form(...),
    timestamp_col: str = Form("timestamp"),
    bad_dates: str = Form("[]"),
    bad_detectors: str = Form("[]"),
    volume_files: list[UploadFile] = File(...),
    detector_dict_file: UploadFile = File(...),
):
    """Average multi-day approach volumes; returns a Synchro-ready CSV."""
    det_dict = _read_excel_or_csv(detector_dict_file)

    daily_dfs: dict[str, pd.DataFrame] = {}
    for vf in volume_files:
        date_str = os.path.splitext(vf.filename or "unknown")[0]
        daily_dfs[date_str] = _read_excel_or_csv(vf)

    bad_d = json.loads(bad_dates)
    bad_det = json.loads(bad_detectors)

    try:
        avg_df, factors = average_approach_volumes(
            daily_dfs=daily_dfs,
            detector_dict=det_dict,
            timestamp_col=timestamp_col,
            bad_dates=bad_d,
            bad_detectors=bad_det,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    synchro_df = export_synchro_volumes(avg_df, intersection_id, factors)
    return _df_to_stream(synchro_df, f"{intersection_id}_synchro.csv")


@app.post("/api/peak-hour")
async def get_peak_hour(
    period: str = Form("AM"),
    interval_col: str = Form("interval"),
    averaged_file: UploadFile = File(...),
):
    """Identify the peak hour start/end from an averaged volume file."""
    df = _read_excel_or_csv(averaged_file)
    try:
        start, end = identify_peak_hour(df, period, interval_col)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"period": period, "peak_hour_start": start, "peak_hour_end": end}


@app.post("/api/average/network")
async def average_network(
    timestamp_col: str = Form("timestamp"),
    bad_dates_json: str = Form("{}"),
    bad_detectors_json: str = Form("{}"),
    intersection_map_json: str = Form("{}"),
    volume_files: list[UploadFile] = File(...),
    detector_dict_files: list[UploadFile] = File(...),
):
    """Average volumes for a network of intersections.

    volume_files filenames must start with '<KITS_ID>_<date>.*'
    detector_dict_files filenames must be '<KITS_ID>.*'
    """
    # Parse network-level dicts
    network_dicts: dict[str, pd.DataFrame] = {}
    for ddf in detector_dict_files:
        kits_id = os.path.splitext(ddf.filename or "")[0]
        network_dicts[kits_id] = _read_excel_or_csv(ddf)

    # Parse volume files grouped by intersection
    network_data: dict[str, dict[str, pd.DataFrame]] = {}
    for vf in volume_files:
        parts = os.path.splitext(vf.filename or "")[0].split("_", 1)
        kits_id = parts[0]
        date_str = parts[1] if len(parts) > 1 else "unknown"
        network_data.setdefault(kits_id, {})[date_str] = _read_excel_or_csv(vf)

    bad_dates = json.loads(bad_dates_json)
    bad_detectors = json.loads(bad_detectors_json)
    intersection_map = json.loads(intersection_map_json)

    try:
        results = network_average_volumes(
            network_data=network_data,
            network_dicts=network_dicts,
            bad_dates=bad_dates,
            bad_detectors=bad_detectors,
            timestamp_col=timestamp_col,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    combined = export_network_synchro(results, intersection_map)
    return _df_to_stream(combined, "network_synchro.csv")


@app.post("/api/compare/network")
async def compare_network(
    warn_pct: float = Form(20.0),
    critical_pct: float = Form(50.0),
    current_file: UploadFile = File(...),
    historical_file: UploadFile = File(...),
):
    """Compare current Synchro volumes against a historical baseline."""
    current_df = _read_excel_or_csv(current_file)
    historical_df = _read_excel_or_csv(historical_file)

    flags_df = compare_with_historical(
        current_df, historical_df, warn_pct=warn_pct, critical_pct=critical_pct
    )
    return flags_df.to_dict(orient="records")


@app.post("/api/ai/summarize")
async def ai_summarize(
    intersection_id: str = Form(...),
    flags_json: str = Form("[]"),
    adjustment_factors_json: str = Form("{}"),
    peak_hours_json: str = Form("{}"),
):
    """Generate a Claude-powered engineering summary for an intersection."""
    try:
        from ai_layer.summarizer import QCSummarizer
        from qc_engine.detector_checks import DetectorFlag, FlagType
    except ImportError as exc:
        raise HTTPException(500, f"AI layer unavailable: {exc}")

    raw_flags = json.loads(flags_json)
    flags: dict[str, list[DetectorFlag]] = {}
    for item in raw_flags:
        det = item["detector"]
        flags.setdefault(det, []).append(
            DetectorFlag(
                detector=det,
                date=item.get("date", ""),
                flag_type=FlagType(item["flag_type"]),
                description=item["description"],
                severity=item["severity"],
            )
        )

    factors = json.loads(adjustment_factors_json)
    peak_hours = json.loads(peak_hours_json)

    try:
        summarizer = QCSummarizer()
        summary = summarizer.summarize_intersection(
            intersection_id, flags, factors, peak_hours
        )
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"AI summarizer error: {exc}")

    return {"intersection_id": intersection_id, "summary": summary}


@app.post("/api/ai/question")
async def ai_question(
    context: str = Form(...),
    question: str = Form(...),
):
    """Answer an ad-hoc engineering question given analysis context."""
    try:
        from ai_layer.summarizer import QCSummarizer
        summarizer = QCSummarizer()
        answer = summarizer.answer_engineering_question(context, question)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {"answer": answer}
