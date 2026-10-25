from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from django.db.models import Q
from django.utils import timezone

from finance_app.transactions.models import MovementType, Transaction


MONTHS_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
MONTH_PATTERN = "|".join(MONTHS_ES)
DOMAIN_TERMS = {
    "abono",
    "banco",
    "balance",
    "bolsillo",
    "categoria",
    "cdt",
    "compra",
    "compras",
    "concepto",
    "cuenta",
    "descripcion",
    "finanzas",
    "gasto",
    "gaste",
    "gastos",
    "ingreso",
    "ingresos",
    "movimiento",
    "movimientos",
    "pago",
    "pagos",
    "retiro",
    "saldo",
    "transaccion",
    "transacciones",
    "transferencia",
}
SEARCH_SYNONYMS = {
    "comida": [
        "comida",
        "restaurante",
        "restaurant",
        "cafe",
        "pizza",
        "burger",
        "presto",
        "wok",
        "tostao",
        "supermercado",
        "mercado",
        "olimpica",
        "exito",
        "d1",
        "ara",
        "jumbo",
    ],
    "alimentacion": [
        "restaurante",
        "restaurant",
        "cafe",
        "pizza",
        "burger",
        "supermercado",
        "mercado",
        "olimpica",
        "exito",
        "d1",
        "ara",
        "jumbo",
    ],
    "restaurante": ["restaurante", "restaurant", "wok", "presto", "pizza", "burger", "cafe"],
    "supermercado": ["supermercado", "mercado", "olimpica", "exito", "d1", "ara", "jumbo"],
}
VALID_INTENTS = {
    "resumen_general",
    "total_gastos",
    "total_ingresos",
    "balance",
    "conteo",
    "top_descripciones",
    "top_categorias",
    "listar_transacciones",
    "promedio",
}
VALID_CLASSIFICATIONS = {"gasto", "ingreso", "cdt", "bolsillo", "all"}


@dataclass
class FinanceChatIntent:
    is_valid: bool
    intent: str = "resumen_general"
    classification: str = "gasto"
    start_date: date | None = None
    end_date: date | None = None
    bank: str = ""
    account_last4: str = ""
    movement_type: str = ""
    search_text: str = ""
    limit: int = 5


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text)


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        return start, date(year, 12, 31)
    return start, date(year, month + 1, 1) - timedelta(days=1)


def _parse_period(question: str, anchor_date: date | None = None) -> tuple[date | None, date | None]:
    text = normalize_text(question)
    today = timezone.localdate()
    data_today = anchor_date or today

    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if len(iso_dates) >= 2:
        return _parse_iso_date(iso_dates[0]), _parse_iso_date(iso_dates[1])
    if len(iso_dates) == 1:
        parsed = _parse_iso_date(iso_dates[0])
        return parsed, parsed

    slash_dates = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    parsed_slash = []
    for day, month, year in slash_dates[:2]:
        try:
            parsed_slash.append(date(int(year), int(month), int(day)))
        except ValueError:
            pass
    if len(parsed_slash) >= 2:
        return parsed_slash[0], parsed_slash[1]
    if len(parsed_slash) == 1:
        return parsed_slash[0], parsed_slash[0]

    if "hoy" in text:
        return today, today
    if "ayer" in text:
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday

    last_days = re.search(r"\bultim(?:os|as)?\s+(\d{1,3})\s+dias\b", text)
    if last_days:
        days = max(1, min(int(last_days.group(1)), 3650))
        return data_today - timedelta(days=days - 1), data_today

    if "semana pasada" in text:
        start_this_week = data_today - timedelta(days=data_today.weekday())
        start = start_this_week - timedelta(days=7)
        return start, start + timedelta(days=6)
    if "esta semana" in text:
        start = data_today - timedelta(days=data_today.weekday())
        return start, data_today

    if "mes pasado" in text:
        first_this_month = data_today.replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)
        return _month_bounds(last_prev_month.year, last_prev_month.month)
    if "este mes" in text:
        return data_today.replace(day=1), data_today

    if "ano pasado" in text or "año pasado" in text:
        year = data_today.year - 1
        return date(year, 1, 1), date(year, 12, 31)
    if "este ano" in text or "este año" in text:
        return date(data_today.year, 1, 1), data_today

    month_match = re.search(rf"\b({MONTH_PATTERN})(?:\s+de)?\s*(\d{{4}})?\b", text)
    if month_match:
        month = MONTHS_ES[month_match.group(1)]
        year = int(month_match.group(2)) if month_match.group(2) else data_today.year
        return _month_bounds(year, month)

    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        year = int(year_match.group(1))
        return date(year, 1, 1), date(year, 12, 31)

    return None, None


def _extract_limit(text: str) -> int:
    normalized = normalize_text(text)
    match = re.search(r"\b(?:top|ultim(?:os|as)?|primer(?:os|as)?)\s+(\d{1,2})\b", normalized)
    if not match:
        return 5
    return max(1, min(int(match.group(1)), 20))


def _extract_bank(text: str) -> str:
    normalized = normalize_text(text)
    if "davivienda" in normalized:
        return "Davivienda"
    if "bogota" in normalized or "banco de bog" in normalized:
        return "Banco de Bogotá"
    return ""


def _extract_account(text: str) -> str:
    normalized = normalize_text(text)
    explicit = re.search(r"\b(?:cuenta|termina(?:da)? en|finaliza(?:da)? en)\s*(\d{4})\b", normalized)
    if explicit:
        return explicit.group(1)
    return ""


def _extract_movement_type(text: str) -> str:
    normalized = normalize_text(text)
    for name in MovementType.objects.filter(is_active=True).values_list("name", flat=True):
        if normalize_text(name) in normalized:
            return name
    return ""


def _strip_period_words(text: str) -> str:
    text = normalize_text(text)
    replacements = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        rf"\b({MONTH_PATTERN})(?:\s+de)?\s*\d{{0,4}}\b",
        r"\b(?:hoy|ayer|este mes|mes pasado|esta semana|semana pasada|este ano|este año|ano pasado|año pasado)\b",
        r"\bultim(?:os|as)?\s+\d{1,3}\s+dias\b",
    ]
    for pattern in replacements:
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_search_candidate(candidate: str) -> str:
    candidate = _strip_period_words(candidate)
    candidate = re.sub(
        r"\b(?:davivienda|banco de bogota|banco de bogotá|gasto|gastos|ingreso|ingresos|transacciones|transaccion|compras|compra|pagos|pago|top|cuanto|cuantos|cuantas|total|suma|en|de|por|para|la|el|los|las|mi|mis)\b",
        " ",
        candidate,
    )
    candidate = re.sub(r"[^a-z0-9 ._-]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ._-")
    if len(candidate) < 3:
        return ""
    if candidate in MONTHS_ES:
        return ""
    return candidate[:80]


def _extract_search_text(question: str) -> str:
    normalized = normalize_text(question)
    patterns = [
        r"\b(?:descripcion|descripción|concepto)\s+(.+)$",
        r"\b(?:comercio|establecimiento|tienda)\s+(.+)$",
        r"\b(?:gaste|gast[eé]|pague|pagu[eé]|compre|compr[eé])\s+(?:en|a|de)?\s+(.+)$",
        r"\b(?:en|de|por)\s+([a-z0-9 ._-]{3,80})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            candidate = _clean_search_candidate(match.group(1))
            if candidate:
                return candidate
    return ""


def classify_finance_question(question: str, anchor_date: date | None = None) -> FinanceChatIntent:
    normalized = normalize_text(question)
    start_date, end_date = _parse_period(question, anchor_date=anchor_date)
    bank = _extract_bank(question)
    account_last4 = _extract_account(question)
    movement_type = _extract_movement_type(question)
    search_text = _extract_search_text(question)
    limit = _extract_limit(question)

    intent = "total_gastos"
    classification = "gasto"

    if any(k in normalized for k in ("resumen", "dashboard", "general")):
        intent = "resumen_general"
        classification = "all"
    elif any(k in normalized for k in ("top", "ranking", "mayores")) and any(
        k in normalized for k in ("categoria", "tipo", "movimiento")
    ):
        intent = "top_categorias"
        classification = "all"
    elif any(k in normalized for k in ("top", "ranking", "mayores")):
        intent = "top_descripciones"
        classification = "gasto"
    elif any(k in normalized for k in ("ultimas", "ultimos", "listar", "lista", "muestrame", "mostrar")):
        intent = "listar_transacciones"
        classification = "all"
    elif any(k in normalized for k in ("cuantas", "cuantos", "cantidad", "conteo", "numero")):
        intent = "conteo"
        classification = "all"
    elif any(k in normalized for k in ("promedio", "media")):
        intent = "promedio"
        classification = "gasto"
    elif any(k in normalized for k in ("balance", "saldo", "neto", "diferencia")):
        intent = "balance"
        classification = "all"
    elif any(k in normalized for k in ("ingreso", "ingresos", "recibi", "recibido", "abono", "abonos")):
        intent = "total_ingresos"
        classification = "ingreso"
    elif any(k in normalized for k in ("gasto", "gastos", "gaste", "pague", "pagos", "compra", "compras")):
        intent = "total_gastos"
        classification = "gasto"

    if "cdt" in normalized:
        classification = "cdt"
    elif "bolsillo" in normalized:
        classification = "bolsillo"
    elif any(k in normalized for k in ("todas", "todos", "todo el historial")) and intent != "total_gastos":
        classification = "all"

    is_valid = (
        intent in VALID_INTENTS
        and (
            any(term in normalized for term in DOMAIN_TERMS)
            or bool(start_date)
            or bool(bank)
            or bool(account_last4)
            or bool(movement_type)
            or bool(search_text)
        )
    )

    return FinanceChatIntent(
        is_valid=is_valid,
        intent=intent,
        classification=classification,
        start_date=start_date,
        end_date=end_date,
        bank=bank,
        account_last4=account_last4,
        movement_type=movement_type,
        search_text=search_text,
        limit=limit,
    )


def _period_label(intent: FinanceChatIntent) -> str:
    if intent.start_date and intent.end_date:
        if intent.start_date == intent.end_date:
            return f"el {intent.start_date.isoformat()}"
        return f"del {intent.start_date.isoformat()} al {intent.end_date.isoformat()}"
    if intent.start_date:
        return f"desde {intent.start_date.isoformat()}"
    if intent.end_date:
        return f"hasta {intent.end_date.isoformat()}"
    return "en todo el historial"


def _money(value: Decimal | int | float | None) -> str:
    value = Decimal(value or 0)
    return f"${float(value):,.0f} COP"


def _decimal_to_number(value: Decimal | int | float | None) -> float:
    return float(value or 0)


def _transaction_label(row: Transaction) -> str:
    return (row.merchant or row.description or row.reference or "Sin detalle").strip()


def _base_queryset(intent: FinanceChatIntent, user_id: int):
    queryset = Transaction.objects.filter(user_id=user_id)
    if intent.start_date:
        queryset = queryset.filter(date__gte=intent.start_date)
    if intent.end_date:
        queryset = queryset.filter(date__lte=intent.end_date)
    if intent.classification != "all":
        queryset = queryset.filter(classification=intent.classification)
    if intent.bank:
        queryset = queryset.filter(bank__icontains=intent.bank)
    if intent.account_last4:
        queryset = queryset.filter(account_last4=intent.account_last4)
    if intent.movement_type:
        queryset = queryset.filter(movement_type__name__icontains=intent.movement_type)
    if intent.search_text:
        search_terms = _expanded_search_terms(intent.search_text)
        search_query = Q()
        for term in search_terms:
            search_query |= Q(merchant__icontains=term)
            search_query |= Q(description__icontains=term)
            search_query |= Q(reference__icontains=term)
        queryset = queryset.filter(
            search_query
        )
    return queryset


def _expanded_search_terms(search_text: str) -> list[str]:
    normalized = normalize_text(search_text)
    terms = SEARCH_SYNONYMS.get(normalized, [search_text])
    return [term for term in terms if len(term.strip()) >= 2]


def _filters_payload(intent: FinanceChatIntent) -> dict[str, Any]:
    return {
        "classification": intent.classification,
        "start_date": intent.start_date.isoformat() if intent.start_date else "",
        "end_date": intent.end_date.isoformat() if intent.end_date else "",
        "bank": intent.bank,
        "account_last4": intent.account_last4,
        "movement_type": intent.movement_type,
        "search_text": intent.search_text,
        "search_terms": _expanded_search_terms(intent.search_text) if intent.search_text else [],
        "limit": intent.limit,
    }


def _top_descriptions(queryset, limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in queryset.only("amount", "merchant", "description", "reference").iterator(chunk_size=500):
        label = _transaction_label(row)[:90]
        if label not in buckets:
            buckets[label] = {"descripcion": label, "total": Decimal("0"), "transacciones": 0}
        buckets[label]["total"] += abs(row.amount or Decimal("0"))
        buckets[label]["transacciones"] += 1
    ordered = sorted(buckets.values(), key=lambda item: item["total"], reverse=True)[:limit]
    for item in ordered:
        item["total"] = _decimal_to_number(item["total"])
    return ordered


def _top_categories(queryset, limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"categoria": "Sin categoria", "total": Decimal("0"), "transacciones": 0})
    for row in queryset.select_related("movement_type").only("amount", "movement_type__name").iterator(chunk_size=500):
        label = row.movement_type.name if row.movement_type else "Sin categoria"
        buckets[label]["categoria"] = label
        buckets[label]["total"] += abs(row.amount or Decimal("0"))
        buckets[label]["transacciones"] += 1
    ordered = sorted(buckets.values(), key=lambda item: item["total"], reverse=True)[:limit]
    for item in ordered:
        item["total"] = _decimal_to_number(item["total"])
    return ordered


def _list_transactions(queryset, limit: int) -> list[dict[str, Any]]:
    rows = (
        queryset.select_related("movement_type")
        .order_by("-date", "-time", "-id")
        .only(
            "id",
            "date",
            "amount",
            "classification",
            "bank",
            "account_last4",
            "description",
            "merchant",
            "movement_type__name",
        )[:limit]
    )
    return [
        {
            "id": row.id,
            "date": row.date.isoformat() if row.date else "",
            "amount": _decimal_to_number(row.amount),
            "classification": row.classification,
            "bank": row.bank,
            "account_last4": row.account_last4,
            "category": row.movement_type.name if row.movement_type else "",
            "description": _transaction_label(row),
        }
        for row in rows
    ]


def _summary_by_classification(queryset) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"classification": "", "total": Decimal("0"), "transacciones": 0})
    for row in queryset.only("classification", "amount").iterator(chunk_size=500):
        label = row.classification or "sin_clasificacion"
        buckets[label]["classification"] = label
        buckets[label]["total"] += row.amount or Decimal("0")
        buckets[label]["transacciones"] += 1
    ordered = sorted(buckets.values(), key=lambda item: item["classification"])
    for item in ordered:
        item["total"] = _decimal_to_number(item["total"])
    return ordered


def _fallback_answer(intent: FinanceChatIntent, payload: dict[str, Any]) -> str:
    period = _period_label(intent)
    filters = []
    if intent.bank:
        filters.append(intent.bank)
    if intent.account_last4:
        filters.append(f"cuenta {intent.account_last4}")
    if intent.movement_type:
        filters.append(intent.movement_type)
    if intent.search_text:
        filters.append(f"descripcion contiene '{intent.search_text}'")
    filter_text = f" ({', '.join(filters)})" if filters else ""

    if intent.intent == "total_gastos":
        return f"Gastos {period}{filter_text}: {_money(payload.get('total'))} en {payload.get('row_count', 0)} transacciones."
    if intent.intent == "total_ingresos":
        return f"Ingresos {period}{filter_text}: {_money(payload.get('total'))} en {payload.get('row_count', 0)} transacciones."
    if intent.intent == "balance":
        return (
            f"Balance {period}{filter_text}: neto {_money(payload.get('net_total'))}. "
            f"Ingresos {_money(payload.get('income_total'))} y gastos {_money(payload.get('expense_total'))}."
        )
    if intent.intent == "conteo":
        return f"Transacciones {period}{filter_text}: {payload.get('row_count', 0)}."
    if intent.intent == "promedio":
        return f"Promedio {period}{filter_text}: {_money(payload.get('average'))} sobre {payload.get('row_count', 0)} transacciones."
    if intent.intent == "top_categorias":
        rows = payload.get("top_categories") or []
        if not rows:
            return f"No encontre categorias {period}{filter_text}."
        joined = ", ".join(f"{row['categoria']}: {_money(row['total'])}" for row in rows[:5])
        return f"Top categorias {period}{filter_text}: {joined}."
    if intent.intent == "top_descripciones":
        rows = payload.get("top_descriptions") or []
        if not rows:
            return f"No encontre movimientos {period}{filter_text}."
        joined = ", ".join(f"{row['descripcion']}: {_money(row['total'])}" for row in rows[:5])
        return f"Top descripciones {period}{filter_text}: {joined}."
    if intent.intent == "listar_transacciones":
        rows = payload.get("transactions") or []
        if not rows:
            return f"No encontre transacciones {period}{filter_text}."
        lines = [f"Ultimas {len(rows)} transacciones {period}{filter_text}:"]
        for row in rows:
            lines.append(
                f"- {row['date']}: {_money(row['amount'])} ({row['classification']}) {row['description'][:70]}"
            )
        return "\n".join(lines)

    by_class = payload.get("by_classification") or []
    class_text = ", ".join(f"{row['classification']}: {_money(row['total'])}" for row in by_class[:5]) or "sin datos"
    return f"Resumen {period}{filter_text}: {payload.get('row_count', 0)} transacciones. Totales por clasificacion: {class_text}."


def resolve_finance_question(question: str, *, user_id: int) -> dict[str, Any]:
    latest_date = (
        Transaction.objects.filter(user_id=user_id, date__isnull=False)
        .order_by("-date")
        .values_list("date", flat=True)
        .first()
    )
    intent = classify_finance_question(question, anchor_date=latest_date)
    if not intent.is_valid:
        return {
            "valid": False,
            "intent": intent.intent,
            "answer": "Solo puedo responder preguntas sobre tus transacciones, gastos, ingresos, cuentas, bancos, CDT o bolsillos.",
            "payload": {"filters": _filters_payload(intent)},
        }

    queryset = _base_queryset(intent, user_id)
    payload: dict[str, Any] = {
        "filters": _filters_payload(intent),
        "period": _period_label(intent),
        "data_anchor_date": latest_date.isoformat() if latest_date else "",
        "row_count": queryset.count(),
    }

    if intent.intent in {"total_gastos", "total_ingresos"}:
        total = Decimal("0")
        for amount in queryset.values_list("amount", flat=True).iterator(chunk_size=500):
            total += abs(amount or Decimal("0"))
        payload["total"] = _decimal_to_number(total)
    elif intent.intent == "balance":
        income_total = Decimal("0")
        expense_total = Decimal("0")
        net_total = Decimal("0")
        for row in queryset.only("amount", "classification").iterator(chunk_size=500):
            amount = row.amount or Decimal("0")
            net_total += amount
            if row.classification == "ingreso" or amount > 0:
                income_total += abs(amount)
            elif row.classification == "gasto" or amount < 0:
                expense_total += abs(amount)
        payload.update(
            {
                "income_total": _decimal_to_number(income_total),
                "expense_total": _decimal_to_number(expense_total),
                "net_total": _decimal_to_number(net_total),
            }
        )
    elif intent.intent == "promedio":
        total = Decimal("0")
        count = 0
        for amount in queryset.values_list("amount", flat=True).iterator(chunk_size=500):
            total += abs(amount or Decimal("0"))
            count += 1
        payload["average"] = _decimal_to_number(total / count) if count else 0
    elif intent.intent == "top_categorias":
        payload["top_categories"] = _top_categories(queryset, intent.limit)
    elif intent.intent == "top_descripciones":
        payload["top_descriptions"] = _top_descriptions(queryset, intent.limit)
    elif intent.intent == "listar_transacciones":
        payload["transactions"] = _list_transactions(queryset, intent.limit)
    elif intent.intent == "resumen_general":
        payload["by_classification"] = _summary_by_classification(queryset)
        payload["top_descriptions"] = _top_descriptions(queryset, min(intent.limit, 5))

    answer = _fallback_answer(intent, payload)
    return {
        "valid": True,
        "intent": intent.intent,
        "answer": answer,
        "payload": payload,
    }
