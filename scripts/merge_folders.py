"""Merge April + October volume files for each intersection into one organised folder.

Expected input layout
---------------------
april_dir/
    {kits_id}_{date}.xlsx       e.g.  1001_2024-04-02.xlsx
october_dir/
    {kits_id}_{date}.xlsx       e.g.  1001_2024-10-01.xlsx
dict_dir/
    {kits_id}.xlsx              e.g.  1001.xlsx   (detector dictionary)
intersection_map.csv / .xlsx
    columns: KITS_ID, Synchro_ID

Output layout
-------------
output_dir/
    {synchro_id}_{kits_id}/     e.g.  500_1001/
        1001_dict.xlsx
        1001_2024-04-02.xlsx
        1001_2024-04-03.xlsx
        1001_2024-10-01.xlsx
        1001_2024-10-07.xlsx

CLI usage
---------
python scripts/merge_folders.py \\
    --april  data/april \\
    --october data/october \\
    --dicts  data/dicts \\
    --map    data/intersection_map.csv \\
    --output data/organized

Importable function
-------------------
from scripts.merge_folders import merge_intersection_folders
result = merge_intersection_folders(
    source_dirs={"april": "data/april", "october": "data/october"},
    dict_dir="data/dicts",
    intersection_map_df=df,   # pandas DataFrame with KITS_ID, Synchro_ID
    output_dir="data/organized",
)
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_map(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    df.columns = df.columns.str.strip()

    # Accept flexible column names
    col_map: dict[str, str] = {}
    for col in df.columns:
        lc = col.lower()
        if "kits" in lc or lc == "kits_id":
            col_map[col] = "KITS_ID"
        elif "synchro" in lc or lc in ("synchro_id", "synchroid"):
            col_map[col] = "Synchro_ID"
    if col_map:
        df = df.rename(columns=col_map)

    missing = [c for c in ("KITS_ID", "Synchro_ID") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Intersection map must have KITS_ID and Synchro_ID columns. "
            f"Missing: {missing}. Found: {list(df.columns)}"
        )
    df["KITS_ID"] = df["KITS_ID"].astype(str).str.strip()
    df["Synchro_ID"] = df["Synchro_ID"].astype(str).str.strip()
    return df[["KITS_ID", "Synchro_ID"]].drop_duplicates("KITS_ID")


def _files_for_kits(directory: Path, kits_id: str) -> list[Path]:
    """Return all files in directory whose stem starts with kits_id + '_'."""
    matches = [
        p for p in directory.iterdir()
        if p.is_file()
        and p.stem.startswith(kits_id + "_")
        and p.suffix.lower() in (".xlsx", ".xls", ".csv")
    ]
    return sorted(matches)


def _dict_file_for_kits(dict_dir: Path, kits_id: str) -> Optional[Path]:
    """Return the dictionary file for this KITS_ID (stem == kits_id)."""
    for ext in (".xlsx", ".xls", ".csv"):
        candidate = dict_dir / f"{kits_id}{ext}"
        if candidate.exists():
            return candidate
    return None


# ── core function ─────────────────────────────────────────────────────────────

def merge_intersection_folders(
    source_dirs: dict[str, str | Path],
    dict_dir: str | Path,
    intersection_map_df: pd.DataFrame,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, dict]:
    """Organise volume + dictionary files into per-intersection folders.

    Parameters
    ----------
    source_dirs:
        Mapping of label → directory.  E.g. ``{"april": "data/april", "october": "data/oct"}``.
        Files from every source directory are collected for each intersection.
    dict_dir:
        Directory containing one dictionary file per intersection, named ``{kits_id}.xlsx``.
    intersection_map_df:
        DataFrame with columns ``KITS_ID`` and ``Synchro_ID``.
    output_dir:
        Root directory where ``{synchro_id}_{kits_id}/`` sub-folders will be created.
    overwrite:
        If True, existing destination files are overwritten.

    Returns
    -------
    Summary dict: ``{kits_id: {"folder": str, "copied": [filenames], "missing": [labels]}}``.
    """
    dict_dir = Path(dict_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dirs_p = {label: Path(d) for label, d in source_dirs.items()}
    summary: dict[str, dict] = {}

    for _, row in intersection_map_df.iterrows():
        kits_id = str(row["KITS_ID"]).strip()
        synchro_id = str(row["Synchro_ID"]).strip()

        folder_name = f"{synchro_id}_{kits_id}"
        dest = output_dir / folder_name
        dest.mkdir(parents=True, exist_ok=True)

        copied: list[str] = []
        missing: list[str] = []

        # Volume files from every source directory
        for label, src_dir in source_dirs_p.items():
            if not src_dir.exists():
                missing.append(f"{label} dir not found: {src_dir}")
                continue
            files = _files_for_kits(src_dir, kits_id)
            if not files:
                missing.append(f"no files for {kits_id} in {label}")
            for src_file in files:
                dst_file = dest / src_file.name
                if dst_file.exists() and not overwrite:
                    copied.append(f"{src_file.name} (skipped, exists)")
                else:
                    shutil.copy2(src_file, dst_file)
                    copied.append(src_file.name)

        # Dictionary file
        dict_file = _dict_file_for_kits(dict_dir, kits_id)
        if dict_file:
            dict_dest = dest / f"{kits_id}_dict{dict_file.suffix}"
            if dict_dest.exists() and not overwrite:
                copied.append(f"{dict_dest.name} (skipped, exists)")
            else:
                shutil.copy2(dict_file, dict_dest)
                copied.append(dict_dest.name)
        else:
            missing.append(f"no dictionary file for {kits_id} in {dict_dir}")

        summary[kits_id] = {
            "folder": str(dest),
            "synchro_id": synchro_id,
            "copied": copied,
            "missing": missing,
        }

    return summary


# ── in-memory version for Streamlit (returns ZIP bytes) ──────────────────────

def merge_to_zip(
    april_files: dict[str, bytes],
    october_files: dict[str, bytes],
    dict_files: dict[str, bytes],
    intersection_map_df: pd.DataFrame,
) -> bytes:
    """Same logic as merge_intersection_folders but entirely in-memory.

    Parameters
    ----------
    april_files / october_files / dict_files:
        ``{filename: file_bytes}`` mappings uploaded via Streamlit.
    intersection_map_df:
        DataFrame with KITS_ID, Synchro_ID columns.

    Returns
    -------
    bytes of a ZIP archive with the organised folder structure.
    """
    zip_buf = io.BytesIO()

    def _stem(name: str) -> str:
        return Path(name).stem

    def _files_matching(file_dict: dict[str, bytes], kits_id: str) -> list[tuple[str, bytes]]:
        return [
            (name, data)
            for name, data in file_dict.items()
            if _stem(name).startswith(kits_id + "_")
        ]

    def _dict_for(file_dict: dict[str, bytes], kits_id: str) -> Optional[tuple[str, bytes]]:
        for name, data in file_dict.items():
            if _stem(name) == kits_id:
                return name, data
        return None

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, row in intersection_map_df.iterrows():
            kits_id = str(row["KITS_ID"]).strip()
            synchro_id = str(row["Synchro_ID"]).strip()
            folder = f"{synchro_id}_{kits_id}"

            # April volumes
            for name, data in _files_matching(april_files, kits_id):
                zf.writestr(f"{folder}/{name}", data)

            # October volumes
            for name, data in _files_matching(october_files, kits_id):
                zf.writestr(f"{folder}/{name}", data)

            # Dictionary
            hit = _dict_for(dict_files, kits_id)
            if hit:
                orig_name, data = hit
                ext = Path(orig_name).suffix
                zf.writestr(f"{folder}/{kits_id}_dict{ext}", data)

    return zip_buf.getvalue()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Merge April + October detector volumes into organised folders."
    )
    parser.add_argument("--april",   required=True, help="Directory with April volume files")
    parser.add_argument("--october", required=True, help="Directory with October volume files")
    parser.add_argument("--dicts",   required=True, help="Directory with detector dictionary files")
    parser.add_argument("--map",     required=True, help="Intersection map file (KITS_ID, Synchro_ID)")
    parser.add_argument("--output",  required=True, help="Output root directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    int_map = _read_map(args.map)
    print(f"Intersection map loaded: {len(int_map)} rows")

    summary = merge_intersection_folders(
        source_dirs={"april": args.april, "october": args.october},
        dict_dir=args.dicts,
        intersection_map_df=int_map,
        output_dir=args.output,
        overwrite=args.overwrite,
    )

    total_copied = sum(len(v["copied"]) for v in summary.values())
    total_missing = sum(len(v["missing"]) for v in summary.values())
    print(f"\n{'─'*60}")
    print(f"Organised {len(summary)} intersection(s).")
    print(f"Files copied : {total_copied}")
    print(f"Missing/warn : {total_missing}")
    print(f"{'─'*60}")

    for kits_id, info in sorted(summary.items()):
        status = "✓" if not info["missing"] else "⚠"
        print(f"\n{status} {info['folder']}")
        for f in info["copied"]:
            print(f"    + {f}")
        for m in info["missing"]:
            print(f"    ! {m}")


if __name__ == "__main__":
    _cli()
