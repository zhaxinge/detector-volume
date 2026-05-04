"""Build organised intersection folders driven by a masterlist.

The masterlist is the single source of truth.  Every time you update
Synchro IDs, re-run this tool and it rebuilds the folder structure.

Masterlist format (Excel or CSV)
---------------------------------
Required columns (flexible names accepted):
    KITS_ID      – stable KITS system identifier
    Synchro_ID   – Synchro intersection number (may change between studies)

Any extra columns (Location, Street, Notes, …) are preserved in the
masterlist but ignored during folder building.

Output folder per intersection
-------------------------------
    {Synchro_ID}_{KITS_ID}/
        {KITS_ID}_dict.xlsx          ← detector dictionary
        {KITS_ID}_2024-04-02.xlsx    ← all volume files that match this KITS_ID
        {KITS_ID}_2024-10-01.xlsx
        …

Volume files are matched purely by the file stem starting with
``{KITS_ID}_``.  There is no concept of April vs. October — just drop
all your volume files into one place and the masterlist assigns them.

Importable API
--------------
    from scripts.merge_folders import read_masterlist, build_from_masterlist

    masterlist = read_masterlist("masterlist.xlsx")
    zip_bytes  = build_from_masterlist(masterlist, volume_files, dict_files)
    # volume_files / dict_files are {filename: bytes} dicts (Streamlit uploads)

CLI
---
    python scripts/merge_folders.py \\
        --masterlist masterlist.xlsx \\
        --volumes    data/volumes \\
        --dicts      data/dicts \\
        --output     data/organized \\
        [--overwrite]
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


# ── column detection ──────────────────────────────────────────────────────────

_KITS_ALIASES    = {"kits_id", "kits", "kitsid", "kits id"}
_SYNCHRO_ALIASES = {"synchro_id", "synchroid", "synchro", "synchro id", "syncid"}


def _detect_col(columns: list[str], aliases: set[str]) -> Optional[str]:
    for col in columns:
        if col.strip().lower().replace(" ", "_") in aliases:
            return col
        if col.strip().lower().replace(" ", "") in {a.replace("_", "") for a in aliases}:
            return col
    return None


# ── public: read masterlist ───────────────────────────────────────────────────

def read_masterlist(source: str | Path | io.IOBase) -> pd.DataFrame:
    """Read and normalise a masterlist file.

    Accepts a file path or a file-like object (e.g. Streamlit UploadedFile).
    Returns a DataFrame with at least ``KITS_ID`` and ``Synchro_ID`` columns,
    plus any extra columns from the original file.  Rows with missing
    KITS_ID or Synchro_ID are dropped with a warning.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        df = pd.read_excel(p) if p.suffix in (".xlsx", ".xls") else pd.read_csv(p)
    else:
        # file-like — try excel first, fall back to csv
        raw = source.read()
        try:
            df = pd.read_excel(io.BytesIO(raw))
        except Exception:
            df = pd.read_csv(io.BytesIO(raw))

    df.columns = df.columns.str.strip()

    kits_col    = _detect_col(list(df.columns), _KITS_ALIASES)
    synchro_col = _detect_col(list(df.columns), _SYNCHRO_ALIASES)

    if not kits_col or not synchro_col:
        missing = []
        if not kits_col:    missing.append("KITS_ID")
        if not synchro_col: missing.append("Synchro_ID")
        raise ValueError(
            f"Masterlist is missing required columns: {missing}.\n"
            f"Columns found: {list(df.columns)}\n"
            f"Expected names like: KITS_ID / KITS, Synchro_ID / Synchro"
        )

    df = df.rename(columns={kits_col: "KITS_ID", synchro_col: "Synchro_ID"})
    df["KITS_ID"]    = df["KITS_ID"].astype(str).str.strip()
    df["Synchro_ID"] = df["Synchro_ID"].astype(str).str.strip()

    # Drop rows where either key is empty / NaN
    before = len(df)
    df = df[(df["KITS_ID"].str.len() > 0) & (df["Synchro_ID"].str.len() > 0)]
    df = df.dropna(subset=["KITS_ID", "Synchro_ID"])
    dropped = before - len(df)
    if dropped:
        print(f"[masterlist] dropped {dropped} rows with missing KITS_ID or Synchro_ID")

    # Deduplicate on KITS_ID, keep first (so the masterlist is the authority)
    df = df.drop_duplicates("KITS_ID", keep="first").reset_index(drop=True)
    return df


def folder_name(kits_id: str, synchro_id: str) -> str:
    return f"{synchro_id}_{kits_id}"


# ── shared matching helpers ───────────────────────────────────────────────────

def _stem(name: str) -> str:
    return Path(name).stem


def _volume_matches(files: dict[str, bytes], kits_id: str) -> list[tuple[str, bytes]]:
    """Files whose stem starts with kits_id + '_'."""
    return [
        (name, data)
        for name, data in files.items()
        if _stem(name).startswith(kits_id + "_")
    ]


def _dict_match(files: dict[str, bytes], kits_id: str) -> Optional[tuple[str, bytes]]:
    """File whose stem equals exactly kits_id."""
    for name, data in files.items():
        if _stem(name) == kits_id:
            return name, data
    return None


# ── in-memory build (Streamlit) ───────────────────────────────────────────────

def build_from_masterlist(
    masterlist: pd.DataFrame,
    volume_files: dict[str, bytes],
    dict_files: dict[str, bytes],
) -> tuple[bytes, pd.DataFrame]:
    """Build organised folder structure in memory and return a ZIP.

    Parameters
    ----------
    masterlist:
        Output of :func:`read_masterlist` — must have ``KITS_ID`` and
        ``Synchro_ID`` columns.
    volume_files:
        ``{filename: bytes}`` — all volume files for any/all periods.
        Each file is matched to an intersection by ``{KITS_ID}_`` prefix.
    dict_files:
        ``{filename: bytes}`` — one dictionary file per intersection,
        named ``{KITS_ID}.xlsx`` (stem must equal the KITS_ID exactly).

    Returns
    -------
    (zip_bytes, summary_df)
    summary_df has columns: KITS_ID, Synchro_ID, Folder, Volume files,
    Dictionary, Missing, plus any extra columns from the masterlist.
    """
    zip_buf = io.BytesIO()
    summary_rows: list[dict] = []

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, row in masterlist.iterrows():
            kits_id    = str(row["KITS_ID"]).strip()
            synchro_id = str(row["Synchro_ID"]).strip()
            folder     = folder_name(kits_id, synchro_id)

            vol_hits  = _volume_matches(volume_files, kits_id)
            dict_hit  = _dict_match(dict_files, kits_id)
            missing   = []

            for name, data in vol_hits:
                zf.writestr(f"{folder}/{name}", data)

            if dict_hit:
                orig_name, data = dict_hit
                ext = Path(orig_name).suffix
                zf.writestr(f"{folder}/{kits_id}_dict{ext}", data)
            else:
                missing.append("dictionary")

            if not vol_hits:
                missing.append("volume files")

            # Preserve extra masterlist columns in the summary
            extra = {
                k: v for k, v in row.items()
                if k not in ("KITS_ID", "Synchro_ID")
            }
            summary_rows.append({
                "KITS_ID":       kits_id,
                "Synchro_ID":    synchro_id,
                "Folder":        folder,
                "Volume files":  len(vol_hits),
                "Dictionary":    "✅" if dict_hit else "❌",
                "Missing":       ", ".join(missing) if missing else "",
                **extra,
            })

    summary_df = pd.DataFrame(summary_rows)
    return zip_buf.getvalue(), summary_df


# ── filesystem build (CLI / batch) ───────────────────────────────────────────

def build_folders_from_masterlist(
    masterlist: pd.DataFrame,
    volumes_dir: str | Path,
    dict_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build organised intersection folders on the filesystem.

    Scans *volumes_dir* for all ``.xlsx/.xls/.csv`` files and routes each
    one into the correct ``{Synchro_ID}_{KITS_ID}/`` folder based on its
    ``{KITS_ID}_`` prefix.

    Parameters
    ----------
    masterlist:     Output of :func:`read_masterlist`.
    volumes_dir:    Flat directory containing all volume files (any period).
    dict_dir:       Directory with one ``{KITS_ID}.xlsx`` per intersection.
    output_dir:     Root output directory.
    overwrite:      Overwrite existing files if True.

    Returns
    -------
    summary_df with one row per intersection (same schema as the in-memory version).
    """
    volumes_dir = Path(volumes_dir)
    dict_dir    = Path(dict_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all volume + dict files into memory maps for uniform matching
    vol_exts  = {".xlsx", ".xls", ".csv"}
    vol_files = {
        p.name: p.read_bytes()
        for p in sorted(volumes_dir.iterdir())
        if p.is_file() and p.suffix.lower() in vol_exts
    }
    dict_files_map = {
        p.name: p.read_bytes()
        for p in sorted(dict_dir.iterdir())
        if p.is_file() and p.suffix.lower() in vol_exts
    }

    summary_rows: list[dict] = []

    for _, row in masterlist.iterrows():
        kits_id    = str(row["KITS_ID"]).strip()
        synchro_id = str(row["Synchro_ID"]).strip()
        dest       = output_dir / folder_name(kits_id, synchro_id)
        dest.mkdir(parents=True, exist_ok=True)

        vol_hits = _volume_matches(vol_files, kits_id)
        dict_hit = _dict_match(dict_files_map, kits_id)
        missing  = []

        for name, data in vol_hits:
            dst = dest / name
            if not dst.exists() or overwrite:
                dst.write_bytes(data)

        if dict_hit:
            orig_name, data = dict_hit
            ext = Path(orig_name).suffix
            dst = dest / f"{kits_id}_dict{ext}"
            if not dst.exists() or overwrite:
                dst.write_bytes(data)
        else:
            missing.append("dictionary")

        if not vol_hits:
            missing.append("volume files")

        extra = {k: v for k, v in row.items() if k not in ("KITS_ID", "Synchro_ID")}
        summary_rows.append({
            "KITS_ID":      kits_id,
            "Synchro_ID":   synchro_id,
            "Folder":       str(dest),
            "Volume files": len(vol_hits),
            "Dictionary":   "✅" if dict_hit else "❌",
            "Missing":      ", ".join(missing) if missing else "",
            **extra,
        })

    return pd.DataFrame(summary_rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build {Synchro_ID}_{KITS_ID} intersection folders from a masterlist.\n"
            "Volume files from all periods go in one --volumes directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--masterlist", required=True,
                        help="Masterlist file (KITS_ID + Synchro_ID columns)")
    parser.add_argument("--volumes",    required=True,
                        help="Directory containing all volume files (any period)")
    parser.add_argument("--dicts",      required=True,
                        help="Directory containing detector dictionary files")
    parser.add_argument("--output",     required=True,
                        help="Root output directory")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Overwrite existing files")
    args = parser.parse_args()

    masterlist = read_masterlist(args.masterlist)
    print(f"Masterlist loaded: {len(masterlist)} intersections")

    summary = build_folders_from_masterlist(
        masterlist   = masterlist,
        volumes_dir  = args.volumes,
        dict_dir     = args.dicts,
        output_dir   = args.output,
        overwrite    = args.overwrite,
    )

    n_ok      = (summary["Missing"] == "").sum()
    n_missing = (summary["Missing"] != "").sum()
    print(f"\n{'─'*60}")
    print(f"Folders built : {len(summary)}")
    print(f"Complete      : {n_ok}")
    print(f"Incomplete    : {n_missing}")
    print(f"{'─'*60}")

    for _, r in summary.sort_values("Missing", ascending=False).iterrows():
        icon = "✓" if not r["Missing"] else "⚠"
        print(f"\n{icon}  {r['Folder']}  ({r['Volume files']} vol file(s))")
        if r["Missing"]:
            print(f"   ! missing: {r['Missing']}")


if __name__ == "__main__":
    _cli()
