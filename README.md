# Signal Detector QC AI

A deterministic, auditable traffic signal detector data quality-control and volume-analysis framework for signal timing optimization.

This repository is designed for workflows using continuous detector data from traffic signal systems such as KITS, detector-to-approach dictionaries, Synchro volume exports, and multi-day volume averaging.

## Purpose

This project helps traffic signal engineers:

- Check detector data quality
- Identify malfunctioning detectors
- Compare detector data across months
- Exclude bad dates and bad detectors
- Average multi-day 15-minute approach volumes
- Export Synchro-ready volume tables
- Generate AI-assisted engineering summaries
- Support network-level signal timing optimization

## Core Principle

The QC logic is deterministic and auditable.

The AI layer should explain, summarize, and help compare results. It should not replace engineering rules or silently change volumes.

```text
Excel / CSV detector data
        ↓
Deterministic QC engine
        ↓
Engineer review and exclusions
        ↓
Adjusted multi-day averaging
        ↓
Synchro-ready exports
        ↓
Optional AI explanation
```

## Repository Structure

```text
signal-detector-qc-ai/
├── backend/                 # FastAPI backend
├── frontend/                # Placeholder for HTML / Streamlit / React UI
├── qc_engine/               # Deterministic detector QC and volume logic
├── ai_layer/                # Optional OpenAI / LangChain summaries
├── examples/                # Example input/output templates
├── tests/                   # Unit tests
├── docs/                    # Technical documentation
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/signal-detector-qc-ai.git
cd signal-detector-qc-ai

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

Run API:

```bash
uvicorn backend.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Run Streamlit prototype:

```bash
streamlit run frontend/streamlit_app.py
```

## Suggested GitHub Repository Name

```text
signal-detector-qc-ai
```

Alternative names:

```text
traffic-signal-qc-engine
kits-detector-qc
synchro-volume-qc
signalops-qc-ai
```

## Recommended Development Path

1. Keep the deterministic QC engine in Python.
2. Keep the frontend simple first, using Streamlit or your existing HTML + Chart.js tool.
3. Add FastAPI once file upload and export workflows are stable.
4. Add OpenAI summaries using direct API calls.
5. Add LangChain/RAG later for historical work orders, split logs, and recurring issue analysis.

## Disclaimer

This repository is a technical prototype. Final signal timing decisions should be reviewed by a qualified traffic signal engineer.


## Run Included Example Volume

This repo includes an example KITS-style detector volume file:

```text
examples/data/example_kits_volume.xlsx
```

Run QC on it:

```bash
python examples/run_example_volume.py
```

The script writes results to:

```text
outputs/example_qc_flags.xlsx
```

## Example Data and Folder Structure

The repository uses this preferred working structure:

```text
examples/data/network/volume/SynchroID_KITSID_IntersectionName/
```

Example:

```text
examples/data/network/volume/SynID_92950110_Route50_MajesticLn/
```

Corridor-level files are stored separately:

```text
examples/data/network/route50_west/
```

See:

```text
docs/folder_structure.md
examples/data/FOLDER_TREE.txt
```

## Auto Discovery

The repo now includes a flexible scanner that does not depend on a fixed folder structure.

Run:

```bash
python scripts/scan_project.py examples/data
```

The scanner identifies:

- KITS ID
- Synchro ID
- intersection name
- raw volume files
- detector dictionary files
- Synchro/historical volume files
- bad date and bad detector exclusion files
- weekday/weekend/saturday/sunday analysis period

See:

```text
docs/auto_discovery.md
```
