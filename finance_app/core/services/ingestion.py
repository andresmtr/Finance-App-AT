import logging
import os
from pathlib import Path

import pandas as pd
from smb.SMBConnection import SMBConnection

from django.conf import settings

from finance_app.core.services.parsers import COLUMNS, normalize_frame

LOGGER = logging.getLogger(__name__)


def _parse_samba_path(remote_path: str) -> tuple[str, str]:
    cleaned = remote_path.strip()
    if not cleaned:
        raise ValueError("SAMBA_REMOTE_PATH is empty")

    if cleaned.startswith("smb://"):
        cleaned = cleaned[len("smb://") :]
        parts = cleaned.split("/", 1)
        if len(parts) < 2:
            raise ValueError("SAMBA_REMOTE_PATH must include share name")
        cleaned = parts[1]

    cleaned = cleaned.lstrip("/")
    parts = cleaned.split("/", 1)
    share = parts[0]
    subpath = parts[1] if len(parts) > 1 else ""
    return share, subpath


def _connect_samba() -> SMBConnection:
    if not settings.SAMBA_HOST:
        raise ValueError("SAMBA_HOST is not set")
    if not settings.SAMBA_USERNAME:
        raise ValueError("SAMBA_USERNAME is not set")
    if not settings.SAMBA_PASSWORD:
        raise ValueError("SAMBA_PASSWORD is not set")
    if not settings.SAMBA_REMOTE_PATH:
        raise ValueError("SAMBA_REMOTE_PATH is not set")

    connection = SMBConnection(
        settings.SAMBA_USERNAME,
        settings.SAMBA_PASSWORD,
        "finance-app",
        settings.SAMBA_HOST,
        use_ntlm_v2=True,
        is_direct_tcp=True,
    )
    if not connection.connect(settings.SAMBA_HOST, 445, timeout=10):
        raise ConnectionError("Could not connect or authenticate with Samba host")
    return connection


def fetch_csvs_from_samba() -> list[Path]:
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = _connect_samba()
    share, subpath = _parse_samba_path(settings.SAMBA_REMOTE_PATH)

    try:
        entries = connection.listPath(share, subpath or "/")
        csv_entries = []
        for entry in entries:
            if entry.isDirectory:
                continue
            filename = entry.filename
            if filename.startswith(".") or filename.startswith("._"):
                continue
            if not filename.lower().endswith(".csv"):
                continue
            csv_entries.append(entry)
        local_paths = []
        for entry in csv_entries:
            remote_file = f"{subpath}/{entry.filename}" if subpath else entry.filename
            local_path = settings.RAW_DATA_DIR / entry.filename
            LOGGER.info("Descargando %s", remote_file)
            with open(local_path, "wb") as handle:
                connection.retrieveFile(share, remote_file, handle)
            local_paths.append(local_path)
        return local_paths
    finally:
        connection.close()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            sep=None,
            engine="python",
            on_bad_lines="skip",
            encoding_errors="replace",
        )
        return frame
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("No se pudo leer %s: %s", path, exc)
        return pd.DataFrame(columns=COLUMNS)


def consolidate_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = _read_csv(path)
        if frame.empty:
            continue
        frames.append(normalize_frame(frame))

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    consolidated = pd.concat(frames, ignore_index=True)
    consolidated = normalize_frame(consolidated)
    return consolidated


def update_consolidated() -> dict:
    csv_paths = fetch_csvs_from_samba()
    LOGGER.info("CSV encontrados: %s", len(csv_paths))

    consolidated = consolidate_csvs(csv_paths)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(settings.CONSOLIDATED_PATH, index=False)

    return {
        "rows": len(consolidated.index),
        "files": len(csv_paths),
        "path": str(settings.CONSOLIDATED_PATH),
    }


def load_consolidated() -> pd.DataFrame:
    if not settings.CONSOLIDATED_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        frame = pd.read_csv(settings.CONSOLIDATED_PATH, dtype=str)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("No se pudo leer consolidado: %s", exc)
        return pd.DataFrame(columns=COLUMNS)
    return normalize_frame(frame)
