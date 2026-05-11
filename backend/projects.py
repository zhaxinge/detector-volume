"""Project management router for cloud-hosted traffic review projects."""

from __future__ import annotations

import io
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/projects", tags=["projects"])

DATA_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))


def _project_dir(pid: str) -> Path:
    return DATA_DIR / "projects" / pid


def _meta_path(pid: str) -> Path:
    return _project_dir(pid) / "meta.json"


def _states_dir(pid: str) -> Path:
    return _project_dir(pid) / "states"


def _files_dir(pid: str) -> Path:
    return _project_dir(pid) / "files"


def _load_meta(pid: str) -> dict[str, Any]:
    path = _meta_path(pid)
    if not path.exists():
        raise HTTPException(404, f"Project {pid!r} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_meta(pid: str, meta: dict[str, Any]) -> None:
    path = _meta_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _list_project_ids() -> list[str]:
    projects_dir = DATA_DIR / "projects"
    if not projects_dir.exists():
        return []
    return [
        d.name for d in projects_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("")
async def create_project(name: str = Form(...)):
    """Create a new named project and return its metadata."""
    pid = uuid.uuid4().hex[:12]
    meta: dict[str, Any] = {
        "id": pid,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intersections": [],
    }
    _save_meta(pid, meta)
    _states_dir(pid).mkdir(parents=True, exist_ok=True)
    _files_dir(pid).mkdir(parents=True, exist_ok=True)
    return meta


@router.get("")
async def list_projects():
    """Return all projects ordered by creation date (newest first)."""
    result = []
    for pid in _list_project_ids():
        try:
            result.append(_load_meta(pid))
        except Exception:
            pass
    result.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return result


@router.delete("/{pid}")
async def delete_project(pid: str):
    """Delete a project and all its data."""
    proj_dir = _project_dir(pid)
    if not proj_dir.exists():
        raise HTTPException(404, f"Project {pid!r} not found")
    shutil.rmtree(proj_dir)
    return {"deleted": pid}


@router.post("/{pid}/upload")
async def upload_project_zip(
    pid: str,
    year: int = Form(2026),
    zip_file: UploadFile = File(...),
):
    """Accept a ZIP (from merge_folders.py) and extract to files/.

    Expected ZIP structure:
        {seq}_{kits_id}/
            {kits_id}_{mmdd}.xlsx
            {kits_id}_dict.xlsx   (optional)
    """
    meta = _load_meta(pid)
    files_root = _files_dir(pid)
    files_root.mkdir(parents=True, exist_ok=True)

    content = await zip_file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(400, "Uploaded file is not a valid ZIP archive")

    intersections: dict[str, dict[str, Any]] = {}
    for member in zf.namelist():
        parts = Path(member).parts
        if len(parts) < 2:
            continue
        folder = parts[0]
        fname = parts[-1]
        if not fname or fname.startswith("."):
            continue

        dest = files_root / folder / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(member))

        if folder not in intersections:
            folder_parts = folder.split("_", 1)
            seq_str = folder_parts[0]
            kits_id = folder_parts[1] if len(folder_parts) > 1 else folder
            try:
                seq = int(seq_str)
            except ValueError:
                seq = len(intersections) + 1
            intersections[folder] = {
                "folder": folder,
                "kits_id": kits_id,
                "synchro_id": seq_str,
                "seq": seq,
            }

    zf.close()

    meta["intersections"] = sorted(intersections.values(), key=lambda x: x["seq"])
    meta["year"] = year
    _save_meta(pid, meta)
    return {"uploaded": len(intersections), "intersections": meta["intersections"]}


@router.get("/{pid}/intersections")
async def list_intersections(pid: str):
    """List intersections with their file lists."""
    meta = _load_meta(pid)
    files_root = _files_dir(pid)
    result = []
    for entry in meta.get("intersections", []):
        folder = entry["folder"]
        folder_path = files_root / folder
        files = (
            sorted(f.name for f in folder_path.iterdir() if f.is_file())
            if folder_path.exists()
            else []
        )
        result.append({**entry, "files": files})
    return result


@router.get("/{pid}/intersections/{kits_id}/files")
async def list_intersection_files(pid: str, kits_id: str):
    """List files for a specific intersection matched by kits_id."""
    meta = _load_meta(pid)
    files_root = _files_dir(pid)
    for entry in meta.get("intersections", []):
        if entry["kits_id"] == kits_id:
            folder_path = files_root / entry["folder"]
            files = (
                sorted(f.name for f in folder_path.iterdir() if f.is_file())
                if folder_path.exists()
                else []
            )
            return {"kits_id": kits_id, "folder": entry["folder"], "files": files}
    raise HTTPException(404, f"Intersection {kits_id!r} not found")


@router.get("/{pid}/intersections/{kits_id}/files/{filename}")
async def get_intersection_file(pid: str, kits_id: str, filename: str):
    """Serve a single file for an intersection."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "Invalid filename")

    meta = _load_meta(pid)
    files_root = _files_dir(pid)

    for entry in meta.get("intersections", []):
        if entry["kits_id"] == kits_id:
            file_path = files_root / entry["folder"] / filename
            if not file_path.exists():
                raise HTTPException(404, f"File {filename!r} not found")
            try:
                file_path.resolve().relative_to(files_root.resolve())
            except ValueError:
                raise HTTPException(400, "Invalid file path")
            return FileResponse(
                str(file_path),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )

    raise HTTPException(404, f"Intersection {kits_id!r} not found")


@router.get("/{pid}/states")
async def get_all_states(pid: str):
    """Return all saved reviewer states keyed by kits_id."""
    _load_meta(pid)
    states_dir = _states_dir(pid)
    result: dict[str, Any] = {}
    if states_dir.exists():
        for f in states_dir.iterdir():
            if f.suffix == ".json":
                kits_id = f.stem
                try:
                    with open(f, encoding="utf-8") as fp:
                        result[kits_id] = json.load(fp)
                except Exception:
                    pass
    return result


@router.get("/{pid}/intersections/{kits_id}/state")
async def get_state(pid: str, kits_id: str):
    """Return saved state for one intersection."""
    _load_meta(pid)
    state_path = _states_dir(pid) / f"{kits_id}.json"
    if not state_path.exists():
        return {}
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


@router.put("/{pid}/intersections/{kits_id}/state")
async def save_state(pid: str, kits_id: str, request: Request):
    """Persist reviewer state (status, notes, exclusions) for one intersection."""
    _load_meta(pid)
    body = await request.json()
    state_path = _states_dir(pid) / f"{kits_id}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    return {"saved": kits_id}
