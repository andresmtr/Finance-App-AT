from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from django.conf import settings
from django.utils import timezone

from finance_app.transactions.models import TelegramChatSession, Transaction

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


VALID_METRICS = {
    "expense_total",
    "income_total",
    "net_total",
    "count",
    "top_merchants",
    "top_movement_types",
    "list_transactions",
}
VALID_CLASSIFICATIONS = {"gasto", "ingreso", "cdt", "bolsillo", "all"}
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


def build_help_text() -> str:
    return (
        "Asistente de finanzas activo.\n"
        "Flujo:\n"
        "1) Enviame una fecha (YYYY-MM-DD, hoy, ayer)\n"
        "2) Haz preguntas de esa fecha\n"
        "3) Escribe finalizar para cerrar la sesion\n"
        "Ejemplos:\n"
        "- 2026-01-12\n"
        "- cuanto gaste\n"
        "- top 5 comercios\n"
        "- finalizar"
    )


@dataclass
class QueryIntent:
    metric: str = "expense_total"
    classification: str = "gasto"
    start_date: date | None = None
    end_date: date | None = None
    bank: str = ""
    merchant: str = ""
    limit: int = 5


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_date_router_rules(text: str) -> date | None:
    normalized = _normalize(text)
    today = timezone.localdate()

    if normalized == "hoy":
        return today
    if normalized == "ayer":
        return today - timedelta(days=1)

    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
    if iso_dates:
        return _parse_iso_date(iso_dates[0])

    slash_dates = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", normalized)
    if slash_dates:
        day, month, year = slash_dates[0]
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            return None

    month_match = re.search(
        r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)(?:\s+de)?\s*(\d{4})\b",
        normalized,
    )
    if month_match:
        month = MONTHS_ES[month_match.group(1)]
        year = int(month_match.group(2))
        return date(year, month, 1)

    return None


def _looks_like_date_command(text: str) -> bool:
    normalized = _normalize(text).strip()
    if normalized in {"hoy", "ayer"}:
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", normalized):
        return True
    if re.fullmatch(
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)( de)? \d{4}",
        normalized,
    ):
        return True
    return False


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return start, end


def _money(value: Decimal) -> str:
    return f"${float(value):,.0f} COP"


def _period_label(start: date | None, end: date | None) -> str:
    if start and end:
        return f"del {start.isoformat()} al {end.isoformat()}"
    if start:
        return f"desde {start.isoformat()}"
    if end:
        return f"hasta {end.isoformat()}"
    return "en todo el historial"


class FinanceAssistant:
    def __init__(self) -> None:
        api_key = settings.OPENAI_API_KEY.strip() if settings.OPENAI_API_KEY else ""
        self._openai_client = OpenAI(api_key=api_key) if (api_key and OpenAI is not None) else None
        self._model = settings.OPENAI_CHAT_MODEL
        self._known_banks = [
            "davivienda",
            "bancolombia",
            "bbva",
            "banco de bogota",
            "nequi",
            "nu",
            "lulo",
            "itau",
        ]

    def answer(self, question: str, *, user_id: int) -> str:
        question = (question or "").strip()
        if not question:
            return "Escribe una pregunta. Ejemplo: cuanto gaste en enero 2026."

        intent = self._resolve_intent(question)
        return self._execute_intent(intent, user_id=user_id)

    def answer_with_default_date(self, question: str, *, user_id: int, default_date: date | None) -> str:
        intent = self._resolve_intent(question)
        if default_date and not intent.start_date and not intent.end_date:
            intent.start_date = default_date
            intent.end_date = default_date
        return self._execute_intent(intent, user_id=user_id)

    def extract_date(self, text: str) -> date | None:
        parsed = self._extract_date_from_openai(text)
        if parsed is not None:
            return parsed
        return _parse_date_router_rules(text)

    def _resolve_intent(self, question: str) -> QueryIntent:
        return self._intent_from_openai(question) or self._intent_from_rules(question)

    def _extract_date_from_openai(self, text: str) -> date | None:
        if self._openai_client is None:
            return None

        today = timezone.localdate().isoformat()
        system_prompt = (
            "Devuelve SOLO una fecha en formato YYYY-MM-DD o null. "
            f"Hoy es {today} (America/Bogota, UTC-05:00). "
            "Si el usuario dice hoy, devuelve hoy. "
            "Si dice ayer, devuelve ayer. "
            "Si menciona un rango, usa solo la fecha inicial. "
            "Sin texto adicional."
        )
        try:
            completion = self._openai_client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            raw = (completion.choices[0].message.content or "").strip().splitlines()[0].strip()
            if raw == "null":
                return None
            return _parse_iso_date(raw)
        except Exception:
            return None

    def _intent_from_openai(self, question: str) -> QueryIntent | None:
        if self._openai_client is None:
            return None

        today = timezone.localdate().isoformat()
        system_prompt = (
            "Eres un parser de consultas financieras. "
            "Convierte preguntas en JSON valido con llaves exactas: "
            "metric, classification, start_date, end_date, bank, merchant, limit. "
            f"Hoy es {today}. "
            "metric debe ser uno de: expense_total, income_total, net_total, count, top_merchants, top_movement_types, list_transactions. "
            "classification debe ser: gasto, ingreso, cdt, bolsillo, all. "
            "start_date/end_date en YYYY-MM-DD o null. "
            "Si no hay fechas, usa null en ambos. "
            "Responde solo JSON."
        )

        try:
            completion = self._openai_client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            data = json.loads(raw)
        except Exception:
            return None

        metric = str(data.get("metric", "expense_total")).strip()
        classification = str(data.get("classification", "gasto")).strip()
        limit = data.get("limit", 5)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5

        if metric not in VALID_METRICS:
            metric = "expense_total"
        if classification not in VALID_CLASSIFICATIONS:
            classification = "gasto"
        limit = max(1, min(limit, 20))

        return QueryIntent(
            metric=metric,
            classification=classification,
            start_date=_parse_iso_date(data.get("start_date")),
            end_date=_parse_iso_date(data.get("end_date")),
            bank=str(data.get("bank", "")).strip(),
            merchant=str(data.get("merchant", "")).strip(),
            limit=limit,
        )

    def _intent_from_rules(self, question: str) -> QueryIntent:
        intent = QueryIntent()
        text = _normalize(question)
        today = timezone.localdate()

        if any(k in text for k in ("top", "ranking")) and any(
            k in text for k in ("comercio", "tienda", "merchant", "establecimiento")
        ):
            intent.metric = "top_merchants"
            intent.classification = "gasto"
        elif any(k in text for k in ("top", "ranking")) and "tipo" in text:
            intent.metric = "top_movement_types"
            intent.classification = "all"
        elif any(
            k in text
            for k in ("cuantas", "cuantos", "numero de transacciones", "cantidad de transacciones", "conteo")
        ):
            intent.metric = "count"
            intent.classification = "all"
        elif any(k in text for k in ("ultimas", "ultimos", "listar", "lista de transacciones", "muestrame")):
            intent.metric = "list_transactions"
            intent.classification = "all"
        elif any(k in text for k in ("neto", "balance", "saldo", "diferencia")):
            intent.metric = "net_total"
            intent.classification = "all"
        elif any(k in text for k in ("ingreso", "ingresos", "recibi", "recibido", "ganancia", "gane")):
            intent.metric = "income_total"
            intent.classification = "ingreso"
        else:
            intent.metric = "expense_total"
            intent.classification = "gasto"

        if "cdt" in text:
            intent.classification = "cdt"
        elif "bolsillo" in text:
            intent.classification = "bolsillo"
        elif "gasto" in text or "gaste" in text:
            intent.classification = "gasto"
        elif "ingreso" in text or "ingresos" in text:
            intent.classification = "ingreso"
        elif "todo" in text or "todas" in text:
            intent.classification = "all"

        date_tokens = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if len(date_tokens) >= 2:
            intent.start_date = _parse_iso_date(date_tokens[0])
            intent.end_date = _parse_iso_date(date_tokens[1])
        elif len(date_tokens) == 1:
            parsed = _parse_iso_date(date_tokens[0])
            intent.start_date = parsed
            intent.end_date = parsed
        elif "hoy" in text:
            intent.start_date = today
            intent.end_date = today
        elif "ayer" in text:
            intent.start_date = today - timedelta(days=1)
            intent.end_date = intent.start_date
        elif "mes pasado" in text:
            first_this_month = today.replace(day=1)
            last_prev_month = first_this_month - timedelta(days=1)
            intent.start_date, intent.end_date = _month_bounds(last_prev_month.year, last_prev_month.month)
        elif "este mes" in text:
            intent.start_date = today.replace(day=1)
            intent.end_date = today
        elif "ano pasado" in text or "año pasado" in text:
            prev_year = today.year - 1
            intent.start_date = date(prev_year, 1, 1)
            intent.end_date = date(prev_year, 12, 31)
        elif "este ano" in text or "este año" in text:
            intent.start_date = date(today.year, 1, 1)
            intent.end_date = today
        else:
            month_match = re.search(
                r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)(?:\s+de)?\s*(\d{4})?\b",
                text,
            )
            if month_match:
                month = MONTHS_ES[month_match.group(1)]
                year = int(month_match.group(2)) if month_match.group(2) else today.year
                intent.start_date, intent.end_date = _month_bounds(year, month)

        top_match = re.search(r"\btop\s+(\d{1,2})\b", text)
        if top_match:
            intent.limit = max(1, min(int(top_match.group(1)), 20))
        last_match = re.search(r"\bultim(?:as|os)?\s+(\d{1,2})\b", text)
        if last_match:
            intent.limit = max(1, min(int(last_match.group(1)), 20))

        for bank in self._known_banks:
            if bank in text:
                intent.bank = bank
                break

        merchant_match = re.search(r"(?:en|de)\s+([a-z0-9 ._-]{3,40})$", text)
        if merchant_match:
            intent.merchant = merchant_match.group(1).strip()

        return intent

    def _base_queryset(self, intent: QueryIntent, *, user_id: int):
        queryset = Transaction.objects.filter(user_id=user_id)
        if intent.start_date:
            queryset = queryset.filter(date__gte=intent.start_date)
        if intent.end_date:
            queryset = queryset.filter(date__lte=intent.end_date)
        if intent.classification != "all":
            queryset = queryset.filter(classification=intent.classification)
        if intent.bank:
            queryset = queryset.filter(bank__icontains=intent.bank)
        if intent.merchant:
            queryset = queryset.filter(merchant__icontains=intent.merchant)
        return queryset

    def _execute_intent(self, intent: QueryIntent, *, user_id: int) -> str:
        queryset = self._base_queryset(intent, user_id=user_id)
        period = _period_label(intent.start_date, intent.end_date)

        if intent.metric == "count":
            total = queryset.count()
            return f"Transacciones {period}: {total}."

        if intent.metric == "net_total":
            net = Decimal("0")
            for amount in queryset.values_list("amount", flat=True):
                net += amount or Decimal("0")
            return f"Neto {period}: {_money(net)}."

        if intent.metric == "income_total":
            total = Decimal("0")
            tx_count = 0
            for amount in queryset.values_list("amount", flat=True):
                amount = amount or Decimal("0")
                if amount > 0:
                    total += amount
                    tx_count += 1
            return f"Ingresos {period}: {_money(total)} en {tx_count} transacciones."

        if intent.metric == "expense_total":
            total = Decimal("0")
            tx_count = 0
            for amount in queryset.values_list("amount", flat=True):
                amount = amount or Decimal("0")
                if amount < 0:
                    total += abs(amount)
                    tx_count += 1
                elif intent.classification == "gasto":
                    total += abs(amount)
                    tx_count += 1
            return f"Gastos {period}: {_money(total)} en {tx_count} transacciones."

        if intent.metric == "top_movement_types":
            buckets: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
            rows = queryset.select_related("movement_type").only("amount", "movement_type__name")
            for row in rows.iterator(chunk_size=500):
                label = row.movement_type.name if row.movement_type else "sin_tipo"
                buckets[label] += abs(row.amount or Decimal("0"))
            ordered = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[: intent.limit]
            if not ordered:
                return f"No encontré transacciones {period}."
            lines = [f"Top tipos {period}:"]
            for idx, (name, total) in enumerate(ordered, start=1):
                lines.append(f"{idx}. {name}: {_money(total)}")
            return "\n".join(lines)

        if intent.metric == "list_transactions":
            rows = (
                queryset.select_related("movement_type")
                .order_by("-date", "-time", "-id")
                .values("date", "amount", "merchant", "description", "classification")[: intent.limit]
            )
            if not rows:
                return f"No encontré transacciones {period}."
            lines = [f"Ultimas {len(rows)} transacciones {period}:"]
            for row in rows:
                label = (row.get("merchant") or row.get("description") or "").strip() or "sin_detalle"
                lines.append(
                    f"- {row.get('date')}: {_money(row.get('amount') or Decimal('0'))} "
                    f"({row.get('classification')}) {label[:50]}"
                )
            return "\n".join(lines)

        buckets = defaultdict(lambda: Decimal("0"))
        rows = queryset.only("merchant", "description", "amount")
        for row in rows.iterator(chunk_size=500):
            name = (row.merchant or "").strip()
            if not name:
                name = (row.description or "").strip()[:45] or "sin_detalle"
            buckets[name] += abs(row.amount or Decimal("0"))

        ordered = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[: intent.limit]
        if not ordered:
            return f"No encontré transacciones {period}."
        lines = [f"Top comercios {period}:"]
        for idx, (name, total) in enumerate(ordered, start=1):
            lines.append(f"{idx}. {name}: {_money(total)}")
        return "\n".join(lines)


class TelegramBotClient:
    def __init__(self, token: str, *, timeout: int = 30) -> None:
        self._token = token
        self._timeout = timeout

    def _request(self, method: str, payload: dict[str, Any]) -> Any:
        encoded = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"https://api.telegram.org/bot{self._token}/{method}",
            data=encoded,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout + 10) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not parsed.get("ok"):
            raise RuntimeError(f"Telegram API error: {parsed}")
        return parsed.get("result", [])

    def get_updates(self, *, offset: int, timeout: int) -> list[dict[str, Any]]:
        return self._request(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": json.dumps(["message"]),
            },
        )

    def send_message(self, *, chat_id: int, text: str) -> None:
        self._request("sendMessage", {"chat_id": chat_id, "text": text})


def _session_for_chat(chat_id: str) -> TelegramChatSession | None:
    return TelegramChatSession.objects.filter(chat_id=chat_id).first()


def clear_chat_session(chat_id: str) -> None:
    TelegramChatSession.objects.filter(chat_id=chat_id).delete()


def build_reply_for_chat_message(
    *,
    chat_id: str,
    text: str,
    assistant: FinanceAssistant,
    user_id: int,
) -> str:
    normalized = _normalize(text).strip()
    if normalized == "finalizar":
        clear_chat_session(chat_id)
        return "Proceso finalizado ✅\nCuando quieras empezar de nuevo, envíame una fecha."

    session = _session_for_chat(chat_id)
    date_candidate = assistant.extract_date(text)

    if session is None or session.selected_date is None:
        if date_candidate is None:
            return "Envíame primero una fecha (YYYY-MM-DD, hoy o ayer)."
        TelegramChatSession.objects.update_or_create(
            chat_id=chat_id,
            defaults={"selected_date": date_candidate},
        )
        return (
            f"Fecha registrada: {date_candidate.isoformat()} ✅\n"
            "Ahora haz tu pregunta sobre ese día. Escribe 'finalizar' para cerrar."
        )

    if date_candidate and _looks_like_date_command(text):
        session.selected_date = date_candidate
        session.save(update_fields=["selected_date", "updated_at"])
        return (
            f"Fecha actualizada: {date_candidate.isoformat()} ✅\n"
            "Ahora haz tu pregunta sobre ese día. Escribe 'finalizar' para cerrar."
        )

    answer = assistant.answer_with_default_date(
        text,
        user_id=user_id,
        default_date=session.selected_date,
    )
    return f"Fecha en sesión: {session.selected_date.isoformat()}\n{answer}"
