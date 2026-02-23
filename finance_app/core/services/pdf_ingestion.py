from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
from pathlib import Path
from typing import Iterable, List

from django.db import transaction

from finance_app.core.services.pdf_parser import parse_pdf_file
from finance_app.transactions.models import ImportBatch, MovementType, StagedTransaction, Transaction


def _to_decimal(value, max_digits: int = 14, decimal_places: int = 2) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        decimal_value = value
    else:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
    if not decimal_value.is_finite():
        return None
    try:
        decimal_value = decimal_value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    digits = decimal_value.as_tuple().digits
    if not digits:
        return None
    if len(digits) > max_digits:
        return None
    return decimal_value


def _to_date(value) -> datetime.date | None:
    if not value:
        return None
    if hasattr(value, "year"):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def _to_time(value) -> datetime.time | None:
    if not value:
        return None
    if hasattr(value, "hour"):
        return value
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(str(value), fmt).time()
        except ValueError:
            continue
    return None


def _movement_type_map() -> dict[str, MovementType]:
    return {mt.name: mt for mt in MovementType.objects.filter(is_active=True)}


def stage_pdf_files(user, pdf_paths: Iterable[Path], passwords: List[str] | None = None, extraction_method: str = "auto") -> ImportBatch:
    batch = ImportBatch.objects.create(user=user, source_label="PDF")
    type_map = _movement_type_map()
    staged_rows = []

    for pdf_path in pdf_paths:
        txs, dbg_list = parse_pdf_file(str(pdf_path), passwords=passwords or [], extraction_method=extraction_method)
        for row in txs:
            movement_name = row.get("movement_type") or ""
            amount = _to_decimal(row.get("amount"))
            if amount is None:
                continue

            external_id = row.get("external_id") or row.get("id") or ""
            
            # Lazily create missing movement types
            m_type = type_map.get(movement_name)
            if not m_type and movement_name:
                m_type, _ = MovementType.objects.get_or_create(name=movement_name, defaults={"is_active": True})
                type_map[movement_name] = m_type

            staged_rows.append(
                StagedTransaction(
                    user=user,
                    batch=batch,
                    bank=row.get("bank", "") or "",
                    account_last4=row.get("account_last4", "") or "",
                    date=_to_date(row.get("date")),
                    time=_to_time(row.get("time")),
                    amount=amount,
                    currency=row.get("currency", "") or "",
                    movement_type=type_map.get(movement_name),
                    reference=row.get("reference", "") or "",
                    merchant=row.get("merchant", "") or "",
                    location=row.get("location", "") or "",
                    channel=row.get("channel", "") or "",
                    description=row.get("description", "") or "",
                    external_id=external_id,
                    source_pdf=row.get("source_pdf", "") or "",
                    source_page=row.get("source_page", None),
                    source_table=row.get("source_table", None),
                )
            )

    if staged_rows:
        StagedTransaction.objects.bulk_create(staged_rows, batch_size=200)
    return batch


@transaction.atomic
def approve_staged_transaction(staged: StagedTransaction) -> Transaction:
    tx = Transaction.objects.create(
        user=staged.user,
        bank=staged.bank,
        account_last4=staged.account_last4,
        date=staged.date,
        time=staged.time,
        amount=staged.amount or Decimal("0.00"),
        currency=staged.currency,
        movement_type=staged.movement_type,
        reference=staged.reference,
        merchant=staged.merchant,
        location=staged.location,
        channel=staged.channel,
        description=staged.description,
        external_id=staged.external_id,
        source_pdf=staged.source_pdf,
        source_page=staged.source_page,
        source_table=staged.source_table,
    )
    staged.status = StagedTransaction.STATUS_APPROVED
    staged.save(update_fields=["status"])
    return tx
