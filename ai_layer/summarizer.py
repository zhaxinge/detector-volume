"""Claude-powered engineering summary layer.

The AI layer is strictly explanatory — it never modifies volumes or overrides
QC flags. All deterministic logic lives in qc_engine.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import anthropic

from qc_engine.detector_checks import DetectorFlag


_DEFAULT_MODEL = os.getenv("AI_MODEL", "claude-sonnet-4-6")
_MAX_TOKENS = 1024


def _build_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _flags_to_text(flags: list[DetectorFlag]) -> str:
    lines = []
    for f in flags:
        lines.append(
            f"  - [{f.severity.upper()}] {f.detector} | {f.flag_type.value} | {f.description}"
        )
    return "\n".join(lines) if lines else "  (none)"


def _build_single_intersection_prompt(
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

    return f"""You are a traffic signal engineer reviewing detector data quality for intersection {intersection_id}.

QC FLAGS DETECTED:
{flagged_text or '  (no flags — all detectors appear clean)'}

ADJUSTMENT FACTORS (total lanes / usable lanes):
{factors_text or '  (no adjustments required)'}

IDENTIFIED PEAK HOURS:
{peak_text or '  (not computed)'}

Please write a concise engineering summary (3–6 sentences) covering:
1. Overall data quality assessment
2. Which detectors / approaches need attention and why
3. Confidence in the averaged volumes (considering adjustment factors)
4. Recommended actions before using volumes in Synchro

Use plain language suitable for a traffic engineer. Do not invent data not listed above."""


def _build_network_prompt(
    comparison_records: list[dict],
    n_intersections: int,
) -> str:
    flagged = [r for r in comparison_records if r.get("severity")]
    flagged_text = json.dumps(flagged, indent=2) if flagged else "  (none)"

    return f"""You are a traffic signal engineer reviewing a network-level volume comparison for {n_intersections} intersections.

INTERSECTIONS WITH SIGNIFICANT VOLUME CHANGES vs. HISTORICAL BASELINE:
{flagged_text}

Please write a concise network analysis summary (4–8 sentences) covering:
1. Overall network data quality
2. Intersections requiring immediate attention (critical changes)
3. Possible explanations for large volume changes (construction, seasonal, detector issues)
4. Recommended next steps before importing into Synchro

Use plain language suitable for a senior traffic engineer. Do not speculate beyond the data provided."""


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

    def _call(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def summarize_intersection(
        self,
        intersection_id: str,
        flags: dict[str, list[DetectorFlag]],
        adjustment_factors: dict[str, float],
        peak_hours: Optional[dict[str, tuple[str, str]]] = None,
    ) -> str:
        """Return a plain-English QC summary for a single intersection."""
        prompt = _build_single_intersection_prompt(
            intersection_id=intersection_id,
            flags=flags,
            adjustment_factors=adjustment_factors,
            peak_hours=peak_hours or {},
        )
        return self._call(prompt)

    def summarize_network(
        self,
        comparison_records: list[dict],
        n_intersections: int,
    ) -> str:
        """Return a plain-English summary for a network comparison."""
        prompt = _build_network_prompt(comparison_records, n_intersections)
        return self._call(prompt)

    def answer_engineering_question(self, context: str, question: str) -> str:
        """Answer an ad-hoc engineering question given analysis context."""
        prompt = f"""You are a traffic signal engineer assistant.

ANALYSIS CONTEXT:
{context}

ENGINEER'S QUESTION:
{question}

Provide a concise, accurate answer based strictly on the context above."""
        return self._call(prompt)
