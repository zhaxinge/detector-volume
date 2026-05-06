"""Claude-powered engineering summary layer.

The AI layer is strictly explanatory — it never modifies volumes or overrides
QC flags. All deterministic logic lives in qc_engine.
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Optional

import anthropic

from qc_engine.detector_checks import DetectorFlag


_DEFAULT_MODEL = os.getenv("AI_MODEL", "claude-opus-4-7")
_MAX_TOKENS = 4096

# Models that support adaptive thinking
_THINKING_MODELS = {"claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"}


def _build_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Stable system prompts — cached across requests
# ---------------------------------------------------------------------------

_INTERSECTION_SYSTEM = """\
You are a senior traffic signal engineer reviewing detector data quality.

Your summary must cover:
1. Overall data quality assessment
2. Which detectors / approaches need attention and why
3. Confidence in the averaged volumes (considering adjustment factors)
4. Recommended actions before using volumes in Synchro
5. Temporal plausibility assessment, noting any violations of expected patterns
6. A bulleted list of problematic detector, approach, and date entries

TEMPORAL PLAUSIBILITY REFERENCE — expected volume ranges by time-of-day:
  Overnight   (00:00–05:00): Very low   (0–50 veh/15-min)
  Early AM    (05:00–06:00): Low, rising (50–100 veh/15-min)
  AM peak     (07:00–09:00): Rising sharply (200–400+ veh/15-min)
  Midday      (09:00–15:00): Moderate–high (150–300+ veh/15-min)
  PM peak     (15:00–18:00): Highest (300–500+ veh/15-min)
  Evening     (19:00–22:00): Declining (100–200 veh/15-min)
  Late night  (22:00–00:00): Low (50–100 veh/15-min)

FLAGS THAT INDICATE TEMPORAL VIOLATIONS:
  • Overnight (00:00–05:00): volumes >100 veh/15-min are suspicious
  • Early AM  (05:00–06:00): volumes >200 veh/15-min are suspicious
  • Late evening (22:00–00:00): volumes >150 veh/15-min are suspicious
  • Inverted pattern: night avg > 50 % of day avg — likely stuck or mis-wired detector

Write 3–6 concise sentences in plain language suitable for a traffic engineer.
Do not invent data not explicitly provided.\
"""

_NETWORK_SYSTEM = """\
You are a senior traffic signal engineer reviewing a network-level volume comparison.

Write a 4–8 sentence summary covering:
1. Overall network data quality
2. Intersections requiring immediate attention (critical changes)
3. Possible explanations for large volume changes (construction, seasonal, detector issues)
4. Recommended next steps before importing into Synchro

Use plain language suitable for a senior traffic engineer.
Do not speculate beyond the data provided.\
"""

_QA_SYSTEM = """\
You are a traffic signal engineer assistant.
Answer questions concisely and accurately based strictly on the analysis context provided.
Do not speculate beyond the supplied data.\
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _flags_to_text(flags: list[DetectorFlag]) -> str:
    lines = []
    for f in flags:
        lines.append(
            f"  - [{f.severity.upper()}] {f.detector} | {f.flag_type.value} | {f.description}"
        )
    return "\n".join(lines) if lines else "  (none)"


def _extract_approach_from_detector(detector: str) -> str:
    match = re.search(r"_([A-Z]{2,4})", detector)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z]{2,4})\d*$", detector)
    if match:
        return match.group(1)
    return detector


def _detailed_flag_list(flags: dict[str, list[DetectorFlag]]) -> str:
    lines = []
    for detector, dflags in flags.items():
        approach = _extract_approach_from_detector(detector)
        for f in dflags:
            date_text = f.date or "unknown date"
            lines.append(
                f"  - {detector} ({approach}) | {date_text} | {f.flag_type.value} | {f.description}"
            )
    return "\n".join(lines) if lines else "  (none)"


def _flag_records(
    flags: dict[str, list[DetectorFlag]],
    kits_id: str = "",
    location: str = "",
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for detector, dflags in flags.items():
        approach = _extract_approach_from_detector(detector)
        for f in dflags:
            records.append(
                {
                    "kits_id": kits_id,
                    "location": location,
                    "detector": detector,
                    "approach": approach,
                    "date": f.date or "",
                    "flag_type": f.flag_type.value,
                    "severity": f.severity,
                    "description": f.description,
                }
            )
    return records


def save_detailed_flag_list_csv(
    flags: dict[str, list[DetectorFlag]],
    kits_id: str = "",
    location: str = "",
    filename: str = "ai_detailed_flag_list.csv",
) -> str:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "kits_id", "location", "detector", "approach",
                "date", "flag_type", "severity", "description",
            ],
        )
        writer.writeheader()
        writer.writerows(_flag_records(flags, kits_id=kits_id, location=location))
    return path


def save_flag_records_csv(
    records: list[dict[str, str]],
    filename: str = "ai_detailed_flag_list.csv",
) -> str:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    fieldnames = [
        "kits_id", "location", "detector", "approach",
        "date", "flag_type", "severity", "description",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if records:
            writer.writerows(records)
    return path


def flatten_network_flag_records(
    network_flags: dict[str, dict[str, dict[str, list[DetectorFlag]]]],
    kits_locations: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    kits_locations = kits_locations or {}
    records: list[dict[str, str]] = []
    for kits_id, date_flags in network_flags.items():
        location = kits_locations.get(kits_id, "")
        for date, det_flags in date_flags.items():
            for detector, dflags in det_flags.items():
                approach = _extract_approach_from_detector(detector)
                for f in dflags:
                    records.append(
                        {
                            "kits_id": kits_id,
                            "location": location,
                            "detector": detector,
                            "approach": approach,
                            "date": f.date or date,
                            "flag_type": f.flag_type.value,
                            "severity": f.severity,
                            "description": f.description,
                        }
                    )
    return records


# ---------------------------------------------------------------------------
# User-message builders (variable data only — not cached)
# ---------------------------------------------------------------------------

def _build_intersection_user_message(
    intersection_id: str,
    flags: dict[str, list[DetectorFlag]],
    adjustment_factors: dict[str, float],
    peak_hours: dict[str, tuple[str, str]],
) -> str:
    flagged_text = ""
    for detector, dflags in flags.items():
        if dflags:
            flagged_text += f"\nDetector {detector}:\n{_flags_to_text(dflags)}\n"

    factors_text = "\n".join(
        f"  {mvt}: {f:.3f}" for mvt, f in adjustment_factors.items()
    )
    peak_text = "\n".join(
        f"  {period}: {s} – {e}" for period, (s, e) in peak_hours.items()
    )
    detailed_text = _detailed_flag_list(flags)

    return (
        f"INTERSECTION: {intersection_id}\n\n"
        f"QC FLAGS DETECTED:\n"
        f"{flagged_text or '  (no flags — all detectors appear clean)'}\n\n"
        f"DETAILED PROBLEM LIST:\n{detailed_text}\n\n"
        f"ADJUSTMENT FACTORS (total lanes / usable lanes):\n"
        f"{factors_text or '  (no adjustments required)'}\n\n"
        f"IDENTIFIED PEAK HOURS:\n"
        f"{peak_text or '  (not computed)'}"
    )


def _build_network_user_message(
    comparison_records: list[dict],
    n_intersections: int,
) -> str:
    flagged = [r for r in comparison_records if r.get("severity")]
    flagged_text = json.dumps(flagged, indent=2) if flagged else "  (none)"
    return (
        f"Network review: {n_intersections} intersections\n\n"
        f"INTERSECTIONS WITH SIGNIFICANT VOLUME CHANGES vs. HISTORICAL BASELINE:\n"
        f"{flagged_text}"
    )


# ---------------------------------------------------------------------------
# QCSummarizer
# ---------------------------------------------------------------------------

class QCSummarizer:
    """Thin wrapper around the Claude API for engineering summaries."""

    def __init__(self, model: str = _DEFAULT_MODEL):
        self.model = model
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = _build_client()
        return self._client

    def _call(self, system_text: str, user_content: str) -> str:
        """Call Claude with a cached system prompt and variable user content.

        The system prompt is marked with cache_control so identical prefixes are
        served from the prompt cache on subsequent requests (~90 % cost reduction
        on the cached portion).  Adaptive thinking is enabled on supported models
        for better engineering reasoning.  Responses are streamed to avoid HTTP
        timeouts on longer outputs.
        """
        use_thinking = self.model in _THINKING_MODELS
        kwargs: dict = dict(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        if use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        with self.client.messages.stream(**kwargs) as stream:
            final = stream.get_final_message()

        for block in final.content:
            if block.type == "text":
                return block.text.strip()
        return ""

    def summarize_intersection(
        self,
        intersection_id: str,
        flags: dict[str, list[DetectorFlag]],
        adjustment_factors: dict[str, float],
        peak_hours: Optional[dict[str, tuple[str, str]]] = None,
    ) -> str:
        """Return a plain-English QC summary for a single intersection."""
        user_msg = _build_intersection_user_message(
            intersection_id=intersection_id,
            flags=flags,
            adjustment_factors=adjustment_factors,
            peak_hours=peak_hours or {},
        )
        return self._call(_INTERSECTION_SYSTEM, user_msg)

    def summarize_network(
        self,
        comparison_records: list[dict],
        n_intersections: int,
    ) -> str:
        """Return a plain-English summary for a network comparison."""
        user_msg = _build_network_user_message(comparison_records, n_intersections)
        return self._call(_NETWORK_SYSTEM, user_msg)

    def answer_engineering_question(self, context: str, question: str) -> str:
        """Answer an ad-hoc engineering question given analysis context."""
        user_msg = f"ANALYSIS CONTEXT:\n{context}\n\nENGINEER'S QUESTION:\n{question}"
        return self._call(_QA_SYSTEM, user_msg)
