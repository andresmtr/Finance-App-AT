from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4
import re

import pandas as pd
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import F
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_http_methods

from finance_app.core.services.pdf_ingestion import approve_staged_transaction, stage_pdf_files, get_classification
from finance_app.core.services.parsers import COLUMNS
from finance_app.transactions.models import ImportBatch, MovementType, StagedTransaction, Transaction


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_time(value: str | None):
    if not value:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            continue
    return None


def _parse_decimal(value: str | None):
    if value is None or value == "":
        return None
    text = str(value).strip()
    neg = False
    if re.search(r"(^\s*-\s*)|(\(\s*)", text):
        neg = True
    text = re.sub(r"[^\d,.\-()]", "", text)
    text = text.replace("(", "").replace(")", "").replace("-", "")
    if not re.search(r"\d", text):
        return None
    comma = "," in text
    dot = "." in text
    if comma and dot:
        last_comma = text.rfind(",")
        last_dot = text.rfind(".")
        if last_dot > last_comma:
            text = text.replace(",", "")
        else:
            text = text.replace(".", "")
            text = text.replace(",", ".")
    elif comma and not dot:
        if re.search(r",\d{1,2}$", text):
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif dot and not comma:
        if not re.search(r"\.\d{1,2}$", text):
            text = text.replace(".", "")
    try:
        val = Decimal(text)
        if not val.is_finite():
            return None
        return -val if neg else val
    except (InvalidOperation, ValueError):
        return None


def _apply_filters(queryset, params: dict):
    start_date = _parse_date(params.get("start_date"))
    end_date = _parse_date(params.get("end_date"))
    if start_date:
        queryset = queryset.filter(date__gte=start_date)
    if end_date:
        queryset = queryset.filter(date__lte=end_date)

    bank = params.get("bank")
    if bank:
        queryset = queryset.filter(bank=bank)

    account_last4 = params.get("account_last4")
    if account_last4:
        queryset = queryset.filter(account_last4=account_last4)

    classification = params.get("classification")
    if classification:
        queryset = queryset.filter(classification=classification)

    movement_type = params.get("movement_type")
    if movement_type:
        queryset = queryset.filter(movement_type__name=movement_type)

    merchant = params.get("merchant")
    if merchant:
        queryset = queryset.filter(merchant__icontains=merchant)

    amount_min = _parse_float(params.get("amount_min"))
    if amount_min is not None:
        queryset = queryset.filter(amount__gte=amount_min)

    amount_max = _parse_float(params.get("amount_max"))
    if amount_max is not None:
        queryset = queryset.filter(amount__lte=amount_max)

    return queryset


@login_required
def transactions(request: HttpRequest) -> HttpResponse:
    base_qs = Transaction.objects.filter(user=request.user).select_related("movement_type")
    filtered = _apply_filters(base_qs, request.GET)

    if request.GET.get("export") == "1":
        export_rows = (
            filtered.annotate(movement_type_name=F("movement_type__name"))
            .values(
                "bank",
                "account_last4",
                "date",
                "time",
                "amount",
                "currency",
                "movement_type_name",
                "classification",
                "reference",
                "merchant",
                "location",
                "channel",
                "description",
                "external_id",
            )
            .order_by("-date", "-time")
        )
        export_frame = [
            {
                **row,
                "movement_type": row.pop("movement_type_name"),
                "classification": row.pop("classification", ""),
                "id": row.pop("external_id"),
            }
            for row in export_rows
        ]
        if export_frame:
            import io
            df = pd.DataFrame(export_frame)[COLUMNS]
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            file_data = output.read()
        else:
            file_data = b""
        response = HttpResponse(file_data, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        response["Content-Disposition"] = "attachment; filename=transacciones.xlsx"
        return response

    if request.GET.get("show_all") == "1":
        page_obj = type("PageObj", (), {"object_list": filtered.order_by("-date", "-time"), "has_previous": False, "has_next": False, "number": 1})()
    else:
        paginator = Paginator(filtered.order_by("-date", "-time"), 25)
        page_number = request.GET.get("page") or 1
        page_obj = paginator.get_page(page_number)

    banks = (
        base_qs.exclude(bank="").order_by().values_list("bank", flat=True).distinct()
    )
    accounts = (
        base_qs.exclude(account_last4="")
        .exclude(account_last4="6176")
        .order_by()
        .values_list("account_last4", flat=True)
        .distinct()
    )
    movement_types = MovementType.objects.filter(is_active=True).order_by("name")

    context = {
        "page_obj": page_obj,
        "filters": request.GET,
        "banks": list(banks),
        "accounts": list(accounts),
        "movement_types": [mt.name for mt in movement_types],
        "classifications": [c[0] for c in StagedTransaction.CLASSIFICATION_CHOICES],
    }
    return render(request, "transactions/transactions.html", context)


@login_required
@require_http_methods(["GET", "POST"])
def import_pdfs(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        files = request.FILES.getlist("pdfs")
        passwords = request.POST.get("passwords", "")
        extraction_method = request.POST.get("extraction_method", "auto")
        password_list = [pw.strip() for pw in passwords.split(",") if pw.strip()]
        if not files:
            return render(
                request,
                "transactions/import.html",
                {"error": "Selecciona al menos un PDF."},
            )

        saved_paths = []
        upload_root = Path(settings.MEDIA_ROOT) / "pdfs"
        upload_root.mkdir(parents=True, exist_ok=True)
        for file_obj in files:
            filename = f"{request.user.id}_{uuid4().hex}_{file_obj.name}"
            dest = upload_root / filename
            with dest.open("wb") as handle:
                for chunk in file_obj.chunks():
                    handle.write(chunk)
            saved_paths.append(dest)

        batch = stage_pdf_files(request.user, saved_paths, passwords=password_list, extraction_method=extraction_method)
        for path in saved_paths:
            try:
                path.unlink()
            except OSError:
                pass
        return redirect("review_next", batch_id=batch.id)

    return render(request, "transactions/import.html")


@login_required
@require_http_methods(["GET", "POST"])
def review_next(request: HttpRequest, batch_id: int) -> HttpResponse:
    batch = get_object_or_404(ImportBatch, id=batch_id, user=request.user)
    movement_types = list(MovementType.objects.filter(is_active=True).order_by("name"))
    movement_map = {mt.name: mt for mt in movement_types}

    pending_qs = (
        StagedTransaction.objects.filter(
            user=request.user,
            batch=batch,
            status=StagedTransaction.STATUS_PENDING,
        )
        .select_related("movement_type")
        .order_by("id")
    )
    pending_list = list(pending_qs)
    pending_count = len(pending_list)
    status_log = []
    status_map = {}

    if request.method == "POST" and pending_list:
        selected_ids = [int(x) for x in request.POST.getlist("row_id")]
        action = request.POST.get("action")

        for staged in pending_list:
            row_id = staged.id
            staged.bank = request.POST.get(f"bank_{row_id}", "") or ""
            staged.account_last4 = request.POST.get(f"account_last4_{row_id}", "") or ""
            staged.date = _parse_date(request.POST.get(f"date_{row_id}"))
            staged.time = _parse_time(request.POST.get(f"time_{row_id}"))
            staged.amount = _parse_decimal(request.POST.get(f"amount_{row_id}"))
            staged.currency = request.POST.get(f"currency_{row_id}", "") or ""
            staged.reference = request.POST.get(f"reference_{row_id}", "") or ""
            staged.merchant = request.POST.get(f"merchant_{row_id}", "") or ""
            staged.location = request.POST.get(f"location_{row_id}", "") or ""
            staged.channel = request.POST.get(f"channel_{row_id}", "") or ""
            staged.description = request.POST.get(f"description_{row_id}", "") or ""
            movement_type_name = request.POST.get(f"movement_type_{row_id}")
            staged.movement_type = movement_map.get(movement_type_name) if movement_type_name else None
            staged.classification = get_classification(
                staged.movement_type.name if staged.movement_type else "",
                staged.amount,
            )
            staged.save()
            if action == "save_all":
                status_map[row_id] = _("Saved")

        if action in {"approve_selected", "delete_selected", "skip_selected"}:
            selected_set = set(selected_ids)
            for staged in pending_list:
                if staged.id not in selected_set:
                    continue
                if action == "approve_selected":
                    approve_staged_transaction(staged)
                    status_log.append({"id": staged.id, "status": _("Approved")})
                elif action == "skip_selected":
                    staged.status = StagedTransaction.STATUS_SKIPPED
                    staged.save(update_fields=["status"])
                    status_log.append({"id": staged.id, "status": _("Skipped")})
                elif action == "delete_selected":
                    staged.status = StagedTransaction.STATUS_DELETED
                    staged.save(update_fields=["status"])
                    status_log.append({"id": staged.id, "status": _("Deleted")})

        pending_list = list(pending_qs)
        pending_count = len(pending_list)

    if not pending_list:
        return render(
            request,
            "transactions/review_done.html",
            {"batch": batch, "pending_count": pending_count, "status_log": status_log},
        )

    context = {
        "batch": batch,
        "staged_rows": pending_list,
        "movement_types": movement_types,
        "pending_count": pending_count,
        "status_log": status_log,
        "status_map": status_map,
    }
    return render(request, "transactions/review.html", context)


@login_required
@require_http_methods(["GET", "POST"])
def manual_transaction(request: HttpRequest) -> HttpResponse:
    movement_types = MovementType.objects.filter(is_active=True).order_by("name")
    if request.method == "POST":
        movement_type_name = request.POST.get("movement_type")
        movement_type = (
            movement_types.filter(name=movement_type_name).first()
            if movement_type_name
            else None
        )
        amount = _parse_decimal(request.POST.get("amount")) or Decimal("0.00")
        Transaction.objects.create(
            user=request.user,
            bank=request.POST.get("bank", "") or "",
            account_last4=request.POST.get("account_last4", "") or "",
            date=_parse_date(request.POST.get("date")),
            time=_parse_time(request.POST.get("time")),
            amount=amount,
            currency=request.POST.get("currency", "") or "",
            movement_type=movement_type,
            classification=request.POST.get("classification") or get_classification(movement_type.name if movement_type else "", amount),
            reference=request.POST.get("reference", "") or "",
            merchant=request.POST.get("merchant", "") or "",
            location=request.POST.get("location", "") or "",
            channel=request.POST.get("channel", "") or "",
            description=request.POST.get("description", "") or "",
            external_id=request.POST.get("external_id", "") or "",
        )
        return redirect("transactions")

    return render(
        request,
        "transactions/manual_form.html",
        {"movement_types": movement_types, "classifications": StagedTransaction.CLASSIFICATION_CHOICES,
        "title": _("Manual Transaction")},
    )


@login_required
@require_http_methods(["GET", "POST"])
def edit_transaction(request: HttpRequest, pk: int) -> HttpResponse:
    tx = get_object_or_404(Transaction, pk=pk, user=request.user)
    movement_types = MovementType.objects.filter(is_active=True).order_by("name")

    if request.method == "POST":
        movement_type_name = request.POST.get("movement_type")
        movement_type = (
            movement_types.filter(name=movement_type_name).first()
            if movement_type_name
            else None
        )

        tx.bank = request.POST.get("bank", "") or ""
        tx.account_last4 = request.POST.get("account_last4", "") or ""
        tx.date = _parse_date(request.POST.get("date"))
        tx.time = _parse_time(request.POST.get("time"))
        tx.amount = _parse_decimal(request.POST.get("amount")) or Decimal("0.00")
        tx.currency = request.POST.get("currency", "") or ""
        tx.movement_type = movement_type
        tx.classification = request.POST.get("classification") or get_classification(movement_type.name if movement_type else "", tx.amount)
        tx.reference = request.POST.get("reference", "") or ""
        tx.merchant = request.POST.get("merchant", "") or ""
        tx.location = request.POST.get("location", "") or ""
        tx.channel = request.POST.get("channel", "") or ""
        tx.description = request.POST.get("description", "") or ""
        tx.external_id = request.POST.get("external_id", "") or ""
        tx.save()

        return redirect("transactions")

    return render(
        request,
        "transactions/manual_form.html",
        {
            "transaction": tx,
            "classifications": StagedTransaction.CLASSIFICATION_CHOICES,
            "movement_types": movement_types,
            "title": _("Edit Transaction"),
        },
    )


@login_required
@require_http_methods(["POST"])
def delete_transaction(request: HttpRequest, pk: int) -> HttpResponse:
    tx = get_object_or_404(Transaction, pk=pk, user=request.user)
    tx.delete()
    return redirect("transactions")
