# finance_app/core/services/pdf_parser.py
# -*- coding: utf-8 -*-

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytesseract
import torch
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader, PdfWriter
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

# ===========================
# CONFIG (ALINEADO AL NOTEBOOK)
# ===========================
DEFAULT_BANK = "Banco"
DEFAULT_CURRENCY = "COP"
DEFAULT_PASSWORDS: List[str] = []

DEFAULT_DPI = 300
LANG_OCR = "spa+eng"
TESS_CONFIG = "--psm 6"

TABLE_DET_MODEL = "microsoft/table-transformer-detection"
DETECTION_SCORE_THRESHOLD = 0.40
SECONDARY_THRESHOLD = 0.30

# Debug vía env
DEBUG_TABLES = os.getenv("PDF_DEBUG", "0") == "1"
DEBUG_SHOW_SAMPLES = 2

HEADER_SYNONYMS = {
    "date": ["fecha", "date", "fec", "fecha mov", "fecha movimiento"],
    "time": ["hora", "time"],
    "amount": ["valor", "value", "amount", "importe", "monto", "total", "transacción", "transaccion"],
    "debit": ["debito", "débito", "cargo", "retiro", "egreso", "salida", "dr"],
    "credit": ["credito", "crédito", "abono", "ingreso", "entrada", "cr"],
    "description": [
        "descripcion", "descripción", "description", "movimiento", "movimientos",
        "clase de movimiento", "detalle", "concepto"
    ],
    "reference": [
        "documento", "doc", "ref", "referencia", "cod", "cod trans", "cód trans",
        "número", "numero", "no.", "num", "nro", "trans", "codtrans", "aut", "autoriz"
    ],
    "location": ["ciudad", "city", "lugar"],
    "channel": ["oficina", "canal", "oficina/canal", "channel", "sucursal", "app", "cajero"],
    "balance": ["saldo", "balance", "sldo", "saldo disponible", "saldo total"],
}

MONTHS_ES = {
    "ene": 1, "enero": 1,
    "feb": 2, "febrero": 2,
    "mar": 3, "marzo": 3,
    "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "junio": 6,
    "jul": 7, "julio": 7,
    "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "septiembre": 9,
    "oct": 10, "octubre": 10,
    "nov": 11, "noviembre": 11,
    "dic": 12, "diciembre": 12,
}

# ===========================
# Helpers
# ===========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _contains_any(text: str, keys: List[str]) -> bool:
    t = _norm(text)
    return any(k in t for k in keys)


def _looks_like_date_token(s: str) -> bool:
    """
    Detecta tokens tipo fecha: 03/02, 03-02, 03\02 (con o sin espacios).
    Esto evita que el fallback de montos convierta '03/02 0686' en 03020686.
    """
    if not s:
        return False
    return bool(re.search(r"\b\d{1,2}\s*[\/\\-]\s*\d{1,2}\b", str(s)))


# ===========================
# Banco + last4
# ===========================
def detect_bank_and_last4(text: str) -> Tuple[str, str]:
    t = _norm(text)
    bank = DEFAULT_BANK
    if "banco de bogotá" in t or "banco de bogota" in t:
        bank = "Banco de Bogotá"
    elif "bancolombia" in t:
        bank = "Bancolombia"
    elif "davivienda" in t:
        bank = "Davivienda"
    elif "bbva" in t:
        bank = "BBVA"

    last4 = ""
    m = re.search(r"(?:cuenta|account|cta|tarjeta|card).{0,80}?(\d{4})\b", text, re.IGNORECASE)
    if m:
        last4 = m.group(1)
    return bank, last4


# ===========================
# Movement type
# ===========================
def classify_movement_type(description: str) -> str:
    d = _norm(description)
    rules = [
        ("cdt", ["cdt", "certificado de deposito", "certificado de depósito", "título", "titulo"]),
        ("intereses", ["interes", "interés", "rendimiento", "rendimientos"]),
        ("impuesto", ["gmf", "4x1000", "impuesto", "retencion", "retención", "iva"]),
        ("retiro", ["retiro", "atm", "cajero", "withdrawal"]),
        ("transferencia", ["transferencia", "transf", "pse", "ach", "envio", "envío"]),
        ("pago", ["pago", "cuota", "tarj", "tarjeta", "credito", "crédito", "servicio", "manejo"]),
        ("compra", ["compra", "pos", "datáfono", "datafono", "comercio", "apple.com", "bill", "supermercado", "mercado"]),
        ("abono", ["abono", "consignacion", "consignación", "deposito", "depósito", "ingreso", "recaudo"]),
    ]
    for label, kws in rules:
        if any(k in d for k in kws):
            return label
    return "pago"


# ===========================
# Moneda + monto robusto
# ===========================
CURRENCY_HINTS = {
    "USD": ["usd", "us$", "u$s", "dolar", "dólar", "dollars"],
    "COP": ["cop", "peso", "pesos", "col$"],
}


def infer_currency(text: str, default: str = DEFAULT_CURRENCY) -> str:
    t = _norm(text)
    if any(k in t for k in CURRENCY_HINTS["USD"]):
        return "USD"
    if any(k in t for k in CURRENCY_HINTS["COP"]):
        return "COP"
    return default


def parse_amount_string(raw: str) -> Optional[float]:
    """
    Parser monetario robusto.
    FIX CLAVE: si el string parece fecha (03/02) => NO lo interpretamos como monto.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # 🚫 Evita '03/02 0686' -> 03020686
    if _looks_like_date_token(s):
        return None

    neg = False
    if re.search(r"(^\s*-\s*)|(\(\s*)", s):
        neg = True

    s2 = re.sub(r"[^\d,.\-()]", "", s)
    s2 = s2.replace("(", "").replace(")", "").replace("-", "")

    if not re.search(r"\d", s2):
        return None

    comma = "," in s2
    dot = "." in s2

    if comma and dot:
        last_comma = s2.rfind(",")
        last_dot = s2.rfind(".")
        if last_dot > last_comma:
            s2 = s2.replace(",", "")
        else:
            s2 = s2.replace(".", "")
            s2 = s2.replace(",", ".")
    elif comma and not dot:
        if re.search(r",\d{1,2}$", s2):
            s2 = s2.replace(",", ".")
        else:
            s2 = s2.replace(",", "")
    elif dot and not comma:
        if not re.search(r"\.\d{1,2}$", s2):
            s2 = s2.replace(".", "")

    try:
        val = float(s2)
        return -val if neg else val
    except Exception:
        return None


def parse_amount_and_currency(
    amount_cell: str,
    desc: str,
    default_currency: str = DEFAULT_CURRENCY
) -> Tuple[Optional[float], str]:
    cur = infer_currency((amount_cell or "") + " " + (desc or ""), default=default_currency)
    amt = parse_amount_string(amount_cell)
    return amt, cur


# Extra: fallback montos desde texto
AMT_FALLBACK_RE = re.compile(r"(?<!\w)(\(?-?\$?\s*\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?\)?)(?!\w)")


def _score_amount_candidate(cand: str, full: str) -> tuple:
    """
    Heurística para no escoger IDs/fechas:
    - prioriza 2 decimales
    - prioriza separadores de miles
    - prioriza candidatos hacia el final (donde suelen estar valor/saldo)
    """
    cand_s = cand or ""
    has_decimals = bool(re.search(r"[.,]\d{2}\b", cand_s))
    has_thousands = bool(re.search(r"\d{1,3}([.,\s]\d{3})+", cand_s))
    pos = (full or "").rfind(cand_s)
    digits = len(re.sub(r"\D", "", cand_s))
    return (int(has_decimals), int(has_thousands), pos, digits)


def parse_amount_from_text_fallback(row_text: str) -> Optional[float]:
    cands = AMT_FALLBACK_RE.findall(row_text or "")
    if not cands:
        return None

    # 🚫 filtra candidatos que parecen fecha
    cands = [c for c in cands if not _looks_like_date_token(c)]
    if not cands:
        return None

    # Si hay >=2 montos con decimales, típicamente:
    # ... <valor_movimiento> <saldo>
    # En muchos extractos, el movimiento es el penúltimo.
    money_with_dec = [c for c in cands if re.search(r"[.,]\d{2}\b", c)]
    if len(money_with_dec) >= 2:
        v = parse_amount_string(money_with_dec[-2])
        if v is not None and abs(v) >= 0.01:
            return v

    # Si no, usamos scoring robusto
    cands_sorted = sorted(cands, key=lambda c: _score_amount_candidate(c, row_text or ""), reverse=True)
    for c in cands_sorted[:8]:
        v = parse_amount_string(c)
        if v is not None and abs(v) >= 0.01:
            return v
    return None


# ===========================
# Fechas
# ===========================
def infer_statement_year(text: str) -> Optional[int]:
    t = text or ""
    m = re.search(r"(periodo|período|extracto|mes)\D{0,40}((19|20)\d{2})", t, re.IGNORECASE)
    if m:
        return int(m.group(2))
    m = re.search(
        r"(ene|enero|feb|febrero|mar|marzo|abr|abril|may|mayo|jun|junio|jul|julio|ago|agosto|sep|sept|septiembre|oct|octubre|nov|noviembre|dic|diciembre)\D{0,10}((19|20)\d{2})",
        t,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(2))
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", t)
    if years:
        from collections import Counter
        y = Counter(years).most_common(1)[0][0]
        return int(y)
    return None


def normalize_date_str(date_raw: str, statement_year: Optional[int]) -> str:
    if not date_raw:
        return ""
    s = str(date_raw).strip()
    if not s:
        return ""

    s_clean = s.replace("\\", "/").replace("-", "/")
    s_clean = re.sub(r"\s+", " ", s_clean).strip()

    m = re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})(?:\s*/\s*(\d{2,4}))?\b", s_clean)
    if m:
        d = int(m.group(1))
        mo = int(m.group(2))
        y = m.group(3)
        if y:
            y = int(y)
            if y < 100:
                y += 2000
        else:
            y = statement_year
        if y:
            try:
                return pd.Timestamp(year=y, month=mo, day=d).date().isoformat()
            except Exception:
                return ""

    m = re.search(r"\b(\d{1,2})\s+(\d{1,2})\b", s_clean)
    if m and not re.search(r"\d{1,2}:\d{2}", s_clean):
        d = int(m.group(1))
        mo = int(m.group(2))
        y = statement_year
        if y:
            try:
                return pd.Timestamp(year=y, month=mo, day=d).date().isoformat()
            except Exception:
                return ""

    s2 = _norm(s_clean)
    m = re.search(
        r"\b(\d{1,2})\s*(ene|enero|feb|febrero|mar|marzo|abr|abril|may|mayo|jun|junio|jul|julio|ago|agosto|sep|sept|septiembre|oct|octubre|nov|noviembre|dic|diciembre)\s*((19|20)\d{2})?\b",
        s2,
    )
    if m:
        d = int(m.group(1))
        mo = MONTHS_ES.get(m.group(2))
        y = int(m.group(3)) if m.group(3) else statement_year
        if mo and y:
            try:
                return pd.Timestamp(year=y, month=mo, day=d).date().isoformat()
            except Exception:
                return ""

    return ""


DATE_FALLBACK_RE = re.compile(r"\b(\d{1,2}\s*/\s*\d{1,2}(?:\s*/\s*\d{2,4})?)\b")


def date_from_row_fallback(row_txt: str, statement_year: Optional[int]) -> str:
    if not row_txt:
        return ""
    m = DATE_FALLBACK_RE.search(row_txt)
    if m:
        return normalize_date_str(m.group(1), statement_year)
    m2 = re.search(
        r"\b(\d{1,2})\s*(ene|enero|feb|febrero|mar|marzo|abr|abril|may|mayo|jun|junio|jul|julio|ago|agosto|sep|sept|septiembre|oct|octubre|nov|noviembre|dic|diciembre)\b",
        _norm(row_txt),
    )
    if m2:
        return normalize_date_str(m2.group(0), statement_year)
    return ""


# ===========================
# PDF decrypt + render
# ===========================
def decrypt_pdf_to_tempfile(pdf_path: str, passwords: List[str]) -> Tuple[Optional[str], Optional[str], str]:
    try:
        reader = PdfReader(pdf_path)
        if not reader.is_encrypted:
            return pdf_path, None, ""

        used = None
        opened = False
        for pw in passwords:
            try:
                r2 = PdfReader(pdf_path)
                ok = r2.decrypt(pw)
                if ok:
                    reader = r2
                    used = pw
                    opened = True
                    break
            except Exception:
                pass

        if not opened:
            return None, None, "No se pudo abrir el PDF con PASSWORDS."

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        tmp.close()

        writer = PdfWriter()
        for p in reader.pages:
            writer.add_page(p)
        with open(tmp_path, "wb") as f:
            writer.write(f)

        return tmp_path, used, ""
    except Exception as exc:
        return None, None, f"decrypt_tempfile error: {type(exc).__name__}: {exc}"


def render_pdf_pages(
    pdf_path: str,
    passwords: List[str],
    dpi: int,
    max_pages: Optional[int],
) -> Tuple[List[Image.Image], Optional[str], str]:
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        if max_pages:
            pages = pages[:max_pages]
        return pages, None, "plain"
    except Exception:
        pass

    for pw in passwords:
        try:
            pages = convert_from_path(pdf_path, dpi=dpi, userpw=pw)
            if max_pages:
                pages = pages[:max_pages]
            return pages, pw, "userpw"
        except Exception:
            continue

    tmp_path, used_pw, err = decrypt_pdf_to_tempfile(pdf_path, passwords)
    if err or not tmp_path:
        return [], None, "fail"

    try:
        pages = convert_from_path(tmp_path, dpi=dpi)
        if max_pages:
            pages = pages[:max_pages]
        return pages, used_pw, "tmp"
    finally:
        try:
            if tmp_path != pdf_path and Path(tmp_path).exists():
                os.remove(tmp_path)
        except Exception:
            pass


# ===========================
# Table detector cache (Django-safe)
# ===========================
_TABLE_PROCESSOR: Optional[AutoImageProcessor] = None
_TABLE_MODEL: Optional[TableTransformerForObjectDetection] = None


def _get_table_detector() -> Tuple[AutoImageProcessor, TableTransformerForObjectDetection]:
    global _TABLE_PROCESSOR, _TABLE_MODEL
    if _TABLE_PROCESSOR is None or _TABLE_MODEL is None:
        device = torch.device("cpu")
        _TABLE_PROCESSOR = AutoImageProcessor.from_pretrained(TABLE_DET_MODEL)
        _TABLE_MODEL = TableTransformerForObjectDetection.from_pretrained(TABLE_DET_MODEL).to(device)
        _TABLE_MODEL.eval()
    return _TABLE_PROCESSOR, _TABLE_MODEL


@torch.no_grad()
def detect_tables(pil_img: Image.Image, score_thr: float = DETECTION_SCORE_THRESHOLD) -> List[Dict[str, Any]]:
    processor, model = _get_table_detector()
    inputs = processor(images=pil_img, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]])
    res = processor.post_process_object_detection(outputs, threshold=score_thr, target_sizes=target_sizes)[0]

    out: List[Dict[str, Any]] = []
    for score, box in zip(res["scores"], res["boxes"]):
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]
        out.append({"bbox": (x0, y0, x1, y1), "score": float(score)})

    out.sort(
        key=lambda d: (d["score"], (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])),
        reverse=True,
    )
    return out


def pad_bbox(bbox, w, h, pad: int = 18):
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return (x0, y0, x1, y1)


# ===========================
# OCR structured
# ===========================
def ocr_data_df(img: Image.Image) -> pd.DataFrame:
    d = pytesseract.image_to_data(
        img,
        lang=LANG_OCR,
        output_type=pytesseract.Output.DATAFRAME,
        config=TESS_CONFIG,
    )
    d = d.dropna(subset=["text"])
    d["text"] = d["text"].astype(str)
    d = d[(d["text"].str.strip() != "") & (d["conf"] > 0)]
    return d


def cluster_rows(words: pd.DataFrame) -> pd.DataFrame:
    if words.empty:
        words["row_id"] = []
        return words
    w = words.copy()
    w["y_mid"] = w["top"] + (w["height"] / 2.0)
    w = w.sort_values(["y_mid", "left"]).reset_index(drop=True)
    h_med = float(w["height"].median()) if len(w) else 10.0
    thr = max(6.0, 0.75 * h_med)
    row_ids = []
    cur = 0
    prev_y = None
    for ym in w["y_mid"].tolist():
        if prev_y is None:
            row_ids.append(cur)
            prev_y = ym
            continue
        if abs(ym - prev_y) > thr:
            cur += 1
            prev_y = ym
        row_ids.append(cur)
    w["row_id"] = row_ids
    return w


def row_text(w: pd.DataFrame, rid: int) -> str:
    r = w[w["row_id"] == rid].sort_values("left")
    return " ".join(r["text"].tolist()).strip()


def pick_header_row(w: pd.DataFrame) -> Optional[int]:
    if w.empty:
        return None
    best, best_score = None, -1
    for rid in w["row_id"].unique().tolist()[:140]:
        t = row_text(w, rid)
        score = 0
        if _contains_any(t, HEADER_SYNONYMS["date"]):
            score += 3
        if _contains_any(t, HEADER_SYNONYMS["description"]):
            score += 3
        if _contains_any(t, HEADER_SYNONYMS["amount"]):
            score += 2
        if _contains_any(t, HEADER_SYNONYMS["debit"]):
            score += 2
        if _contains_any(t, HEADER_SYNONYMS["credit"]):
            score += 2
        if _contains_any(t, HEADER_SYNONYMS["reference"]):
            score += 1
        if _contains_any(t, HEADER_SYNONYMS["channel"]):
            score += 1
        if _contains_any(t, HEADER_SYNONYMS["location"]):
            score += 1
        if _contains_any(t, HEADER_SYNONYMS["balance"]):
            score += 1
        if score > best_score:
            best_score = score
            best = rid
    return best if best_score >= 4 else None


def infer_columns_from_header(w: pd.DataFrame, header_rid: int) -> List[Tuple[float, float, str]]:
    hdr = w[w["row_id"] == header_rid].sort_values("left").copy()
    if hdr.empty:
        return []
    hdr["x0"] = hdr["left"]
    hdr["x1"] = hdr["left"] + hdr["width"]
    xs = hdr[["x0", "x1", "text"]].values.tolist()

    groups = []
    cur = [xs[0]]
    for prev, nxt in zip(xs, xs[1:]):
        gap = nxt[0] - prev[1]
        if gap > 18:
            groups.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    groups.append(cur)

    cols = []
    for g in groups:
        x0 = float(min(v[0] for v in g))
        x1 = float(max(v[1] for v in g))
        txt = " ".join(v[2] for v in g).strip()
        cols.append((x0, x1, txt))

    cols = sorted(cols, key=lambda z: z[0])

    fixed = []
    for i, (x0, x1, txt) in enumerate(cols):
        left = x0 - 10
        right = x1 + 10
        if i > 0:
            _, prev_x1, _ = cols[i - 1]
            left = (prev_x1 + x0) / 2.0
        if i < len(cols) - 1:
            next_x0, _, _ = cols[i + 1]
            right = (x1 + next_x0) / 2.0
        fixed.append((left, right, txt))
    return fixed


def map_header_to_field(htxt: str) -> Optional[str]:
    ht = _norm(htxt)
    if _contains_any(ht, HEADER_SYNONYMS["date"]):
        return "date"
    if _contains_any(ht, HEADER_SYNONYMS["time"]):
        return "time"
    if _contains_any(ht, HEADER_SYNONYMS["description"]):
        return "description"
    if _contains_any(ht, HEADER_SYNONYMS["reference"]):
        return "reference"
    if _contains_any(ht, HEADER_SYNONYMS["location"]):
        return "location"
    if _contains_any(ht, HEADER_SYNONYMS["channel"]):
        return "channel"
    if _contains_any(ht, HEADER_SYNONYMS["balance"]):
        return "balance"
    # IMPORTANT: debit/credit antes que amount
    if _contains_any(ht, HEADER_SYNONYMS["debit"]):
        return "debit"
    if _contains_any(ht, HEADER_SYNONYMS["credit"]):
        return "credit"
    if _contains_any(ht, HEADER_SYNONYMS["amount"]):
        return "amount"
    return None


def assign_words_to_cols(w: pd.DataFrame, cols: List[Tuple[float, float, str]]) -> pd.DataFrame:
    if w.empty or not cols:
        w["col_id"] = -1
        return w
    ww = w.copy()
    ww["x_mid"] = ww["left"] + (ww["width"] / 2.0)

    def col_for_x(x):
        for i, (x0, x1, _) in enumerate(cols):
            if x0 <= x <= x1:
                return i
        return -1

    ww["col_id"] = ww["x_mid"].apply(col_for_x)
    return ww


def parse_table_image_to_transactions(
    table_img: Image.Image,
    bank: str,
    last4: str,
    statement_year: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    words = cluster_rows(ocr_data_df(table_img))
    header_rid = pick_header_row(words)
    if header_rid is None:
        return [], {"error": "No header detected"}

    cols = infer_columns_from_header(words, header_rid)

    col_map: Dict[str, int] = {}
    for i, (_, _, htxt) in enumerate(cols):
        f = map_header_to_field(htxt)
        if f and f not in col_map:
            col_map[f] = i

    words2 = assign_words_to_cols(words, cols)

    def cell_text(rid: int, colid: int) -> str:
        c = words2[(words2["row_id"] == rid) & (words2["col_id"] == colid)].sort_values("left")
        return " ".join(c["text"].tolist()).strip()

    txs: List[Dict[str, Any]] = []
    prev_tx = None
    row_ids = sorted([rid for rid in words2["row_id"].unique().tolist() if rid > header_rid])

    skipped_no_amount = 0
    continued = 0
    kept = 0
    samples_skipped: List[str] = []

    header_txt = row_text(words2, header_rid)

    for rid in row_ids:
        date_raw = cell_text(rid, col_map["date"]) if "date" in col_map else ""
        time = cell_text(rid, col_map["time"]) if "time" in col_map else ""
        desc = cell_text(rid, col_map["description"]) if "description" in col_map else row_text(words2, rid)

        ref = cell_text(rid, col_map["reference"]) if "reference" in col_map else ""
        loc = cell_text(rid, col_map["location"]) if "location" in col_map else ""
        chan = cell_text(rid, col_map["channel"]) if "channel" in col_map else ""

        debit_raw = cell_text(rid, col_map["debit"]) if "debit" in col_map else ""
        credit_raw = cell_text(rid, col_map["credit"]) if "credit" in col_map else ""
        amt_raw = cell_text(rid, col_map["amount"]) if "amount" in col_map else ""

        date_iso = normalize_date_str(date_raw, statement_year)
        if not date_iso:
            date_iso = date_from_row_fallback(row_text(words2, rid), statement_year)

        currency = infer_currency(
            (debit_raw or "") + " " + (credit_raw or "") + " " + (amt_raw or "") + " " + (desc or ""),
            default=DEFAULT_CURRENCY,
        )

        amount = None
        deb = parse_amount_string(debit_raw) if debit_raw else None
        cre = parse_amount_string(credit_raw) if credit_raw else None

        if cre is not None and (deb is None or abs(cre) >= 0.01):
            amount = abs(cre)
        elif deb is not None and (cre is None or abs(deb) >= 0.01):
            amount = -abs(deb)
        else:
            amount, _ = parse_amount_and_currency(amt_raw, desc, default_currency=DEFAULT_CURRENCY)
            if amount is None:
                amount = parse_amount_from_text_fallback(row_text(words2, rid))

        row_full_txt = row_text(words2, rid)

        # Continuación de línea
        if (not date_iso) and (amount is None) and desc and prev_tx is not None:
            prev_tx["description"] = (prev_tx["description"] + " " + desc).strip()
            if ref and not prev_tx.get("reference"):
                prev_tx["reference"] = ref
            if loc and not prev_tx.get("location"):
                prev_tx["location"] = loc
            if chan and not prev_tx.get("channel"):
                prev_tx["channel"] = chan
            continued += 1
            continue

        # último intento por texto completo
        if amount is None:
            amount = parse_amount_from_text_fallback(row_full_txt)

        if amount is None:
            skipped_no_amount += 1
            if len(samples_skipped) < DEBUG_SHOW_SAMPLES:
                samples_skipped.append(row_full_txt)
            continue

        tx = {
            "bank": bank,
            "account_last4": last4,
            "date": date_iso,
            "time": time,
            "amount": amount,
            "currency": currency,
            "movement_type": classify_movement_type(desc),
            "reference": ref,
            "merchant": "",
            "location": loc,
            "channel": chan,
            "description": desc if desc else row_full_txt,
            "id": uuid.uuid4().hex,
        }
        txs.append(tx)
        prev_tx = tx
        kept += 1

    dbg = {
        "header_row_id": header_rid,
        "header_text": header_txt,
        "col_map": col_map,
        "n_cols": len(cols),
        "rows_total_after_header": len(row_ids),
        "kept": kept,
        "continued": continued,
        "skipped_no_amount": skipped_no_amount,
        "skipped_samples": samples_skipped,
    }
    return txs, dbg


def fullpage_fallback_transactions(
    page_img: Image.Image,
    bank: str,
    last4: str,
    statement_year: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    txs, dbg = parse_table_image_to_transactions(page_img, bank, last4, statement_year)
    dbg["fullpage_fallback"] = True
    return txs, dbg


# ===========================
# API principal para Django
# ===========================
def parse_pdf_file(
    pdf_path: str,
    passwords: Optional[List[str]] = None,
    max_pages: Optional[int] = None,
    dpi: int = DEFAULT_DPI,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retorna:
      - txs: lista de movimientos
      - dbg_list: lista de dicts debug por tabla/página
    """
    passwords = passwords or DEFAULT_PASSWORDS

    pages, used_pw, method = render_pdf_pages(pdf_path, passwords, dpi, max_pages)
    if not pages:
        return [], [{
            "error": "No pages rendered",
            "pdf": Path(pdf_path).name,
            "method": method,
            "used_pw": bool(used_pw)
        }]

    first_txt = ""
    try:
        first_txt = pytesseract.image_to_string(pages[0].convert("L"), lang=LANG_OCR, config=TESS_CONFIG)
    except Exception:
        pass

    bank, last4 = detect_bank_and_last4(first_txt)
    statement_year = infer_statement_year(first_txt)

    if DEBUG_TABLES:
        print(
            f"\n📄 {Path(pdf_path).name} | pages={len(pages)} | render={method}{' (pw)' if used_pw else ''} "
            f"| bank={bank} | last4={last4 or 'NA'} | year={statement_year or 'NA'}"
        )

    txs: List[Dict[str, Any]] = []
    dbg_list: List[Dict[str, Any]] = []

    for p_idx, page_img in enumerate(pages, start=1):
        try:
            dets = detect_tables(page_img, score_thr=DETECTION_SCORE_THRESHOLD)
            if len(dets) == 0:
                dets = detect_tables(page_img, score_thr=SECONDARY_THRESHOLD)
        except Exception:
            dets = []

        page_got_rows = 0

        if DEBUG_TABLES:
            print(
                f"  - page {p_idx}: tablas_detectadas={len(dets)} "
                f"(thr={DETECTION_SCORE_THRESHOLD}/{SECONDARY_THRESHOLD}) size={page_img.size}"
            )

        for t_idx, d in enumerate(dets, start=1):
            x0, y0, x1, y1 = pad_bbox(d["bbox"], page_img.size[0], page_img.size[1], pad=18)
            crop = page_img.crop((x0, y0, x1, y1))

            rows, dbg = parse_table_image_to_transactions(crop, bank, last4, statement_year)
            dbg_list.append({
                "pdf": Path(pdf_path).name,
                "page": p_idx,
                "table": t_idx,
                "bbox": (x0, y0, x1, y1),
                "score": d["score"],
                **dbg,
            })

            if DEBUG_TABLES:
                print(f"    table {t_idx}: score={d['score']:.2f} bbox={(x0, y0, x1, y1)} col_map={dbg.get('col_map')}")
                if dbg.get("skipped_samples"):
                    for i, smp in enumerate(dbg["skipped_samples"], 1):
                        print(f"      sample_skipped_{i}: {smp[:180]}")

            if rows:
                for tx in rows:
                    tx["source_pdf"] = Path(pdf_path).name
                    tx["source_page"] = p_idx
                    tx["source_table"] = t_idx
                txs.extend(rows)
                page_got_rows += len(rows)

        # fallback página completa
        if (len(dets) == 0) or (page_got_rows == 0):
            fallback_rows, fdbg = fullpage_fallback_transactions(page_img, bank, last4, statement_year)
            dbg_list.append({
                "pdf": Path(pdf_path).name,
                "page": p_idx,
                "table": 0,
                "fallback": True,
                **fdbg,
            })
            if fallback_rows:
                for tx in fallback_rows:
                    tx["source_pdf"] = Path(pdf_path).name
                    tx["source_page"] = p_idx
                    tx["source_table"] = 0
                txs.extend(fallback_rows)

    return txs, dbg_list