from datetime import date, datetime

import pandas as pd
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from finance_app.core.services.ingestion import load_consolidated
from finance_app.core.services.parsers import COLUMNS


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


def _apply_filters(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    filtered = frame.copy()

    start_date = _parse_date(params.get("start_date"))
    end_date = _parse_date(params.get("end_date"))
    if start_date:
        filtered = filtered[filtered["date"] >= start_date]
    if end_date:
        filtered = filtered[filtered["date"] <= end_date]

    bank = params.get("bank")
    if bank:
        filtered = filtered[filtered["bank"] == bank]

    account_last4 = params.get("account_last4")
    if account_last4:
        filtered = filtered[filtered["account_last4"] == account_last4]

    movement_type = params.get("movement_type")
    if movement_type:
        filtered = filtered[filtered["movement_type"] == movement_type]

    merchant = params.get("merchant")
    if merchant:
        filtered = filtered[filtered["merchant"].str.contains(merchant, case=False, na=False)]

    amount_min = _parse_float(params.get("amount_min"))
    if amount_min is not None:
        filtered = filtered[filtered["amount"] >= amount_min]

    amount_max = _parse_float(params.get("amount_max"))
    if amount_max is not None:
        filtered = filtered[filtered["amount"] <= amount_max]

    return filtered


@login_required
def transactions(request: HttpRequest) -> HttpResponse:
    frame = load_consolidated()
    frame = frame.dropna(subset=["amount"]) if not frame.empty else frame

    filtered = _apply_filters(frame, request.GET)
    filtered = filtered.sort_values(by=["date", "time"], ascending=False, na_position="last")

    if request.GET.get("export") == "1":
        csv_data = filtered[COLUMNS].to_csv(index=False)
        response = HttpResponse(csv_data, content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=transacciones.csv"
        return response

    paginator = Paginator(filtered.to_dict(orient="records"), 25)
    page_number = request.GET.get("page") or 1
    page_obj = paginator.get_page(page_number)

    context = {
        "page_obj": page_obj,
        "filters": request.GET,
        "banks": sorted(frame["bank"].dropna().unique().tolist()) if not frame.empty else [],
        "accounts": sorted(
            frame["account_last4"]
            .dropna()
            .loc[frame["account_last4"] != "6176"]
            .unique()
            .tolist()
        )
        if not frame.empty
        else [],
        "movement_types": sorted(frame["movement_type"].dropna().unique().tolist())
        if not frame.empty
        else [],
    }
    return render(request, "transactions/transactions.html", context)
