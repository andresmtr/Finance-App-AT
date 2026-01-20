import hashlib
import re
from datetime import date

import pandas as pd

COLUMNS = [
    "bank",
    "account_last4",
    "date",
    "time",
    "amount",
    "currency",
    "movement_type",
    "reference",
    "merchant",
    "location",
    "channel",
    "description",
    "id",
]


def _normalize_strings(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str).str.strip()
            frame[column] = frame[column].replace({"nan": None, "None": None})
    return frame


def _normalize_account_last4(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if not text.isdigit():
        return text

    cleaned = text.lstrip("0")
    cleaned = cleaned.rstrip("0")
    if cleaned == "":
        return "0"
    return cleaned


def _normalize_bank(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if "daviv" in text.lower():
        return "Banco Davivienda"
    return text


def _parse_amount(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    text = re.sub(r"[^0-9,\.-]", "", text)
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        value_float = float(text)
    except ValueError:
        return None
    return -value_float if negative else value_float


def _fallback_hash(row: pd.Series) -> str:
    parts = [
        str(row.get("bank", "")),
        str(row.get("account_last4", "")),
        str(row.get("date", "")),
        str(row.get("time", "")),
        str(row.get("amount", "")),
        str(row.get("reference", "")),
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[COLUMNS]
    frame = _normalize_strings(frame)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce").dt.time

    frame["amount"] = frame["amount"].apply(_parse_amount)
    frame["account_last4"] = frame["account_last4"].apply(_normalize_account_last4)
    frame["bank"] = frame["bank"].apply(_normalize_bank)

    frame["dedup_key"] = frame["id"].where(frame["id"].notna() & (frame["id"] != ""))
    missing_mask = frame["dedup_key"].isna()
    frame.loc[missing_mask, "dedup_key"] = frame[missing_mask].apply(_fallback_hash, axis=1)
    frame = frame.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"])

    return frame


def split_date_parts(value: date | None) -> tuple[int | None, int | None, int | None]:
    if not value:
        return None, None, None
    return value.year, value.month, value.day
