import json

import pandas as pd
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from finance_app.core.services.ingestion import load_consolidated


def _safe_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.fillna(0).sum())


def _apply_filters(
    frame: pd.DataFrame,
    account_last4: str,
    movement_type: str,
    bank: str,
    reference: str,
) -> pd.DataFrame:
    filtered = frame
    if bank:
        filtered = filtered[filtered["bank"] == bank]
    if account_last4:
        filtered = filtered[filtered["account_last4"] == account_last4]
    if movement_type:
        filtered = filtered[filtered["movement_type"] == movement_type]
    if reference:
        filtered = filtered[filtered["reference"].str.contains(reference, case=False, na=False)]
    return filtered


def _format_period_labels(index: pd.DatetimeIndex, granularity: str) -> list[str]:
    if granularity == "monthly":
        return index.to_period("M").astype(str).tolist()
    return [value.date().isoformat() for value in index]


@login_required
def dashboard(request: HttpRequest) -> HttpResponse:
    frame_all = load_consolidated()
    frame_all = frame_all.dropna(subset=["amount"]) if not frame_all.empty else frame_all

    bank = request.GET.get("bank", "")
    account_last4 = request.GET.get("account_last4", "")
    movement_type = request.GET.get("movement_type", "")
    reference = request.GET.get("reference", "")
    granularity = request.GET.get("granularity", "daily")
    granularity_map = {"daily": "D", "weekly": "W", "monthly": "M"}
    freq = granularity_map.get(granularity, "D")

    frame = _apply_filters(frame_all, account_last4, movement_type, bank, reference)
    frame = frame.copy()

    if not frame.empty:
        frame = frame.dropna(subset=["date"])
        frame["date_dt"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date_dt"])

    movement = frame["movement_type"].fillna("").str.lower() if not frame.empty else pd.Series()
    income_mask = movement.str.contains("abono|intereses", na=False)
    expense_mask = movement.str.contains("compra|retiro|pago|impuesto", na=False)
    if not frame.empty:
        transfer_mask = movement.str.contains("transferencia", na=False)
        transfer_account_mask = frame["account_last4"] == "9974"
        expense_mask = expense_mask | (transfer_mask & transfer_account_mask)
        frame["income_flag"] = income_mask
        frame["expense_flag"] = expense_mask

    income = frame[income_mask] if not frame.empty else frame
    expense = frame[expense_mask] if not frame.empty else frame

    total_income = _safe_sum(income["amount"]) if not income.empty else 0.0
    total_expense = _safe_sum(expense["amount"].abs()) if not expense.empty else 0.0
    net = total_income - total_expense

    tx_count = int(len(frame.index)) if not frame.empty else 0

    merchant_spend = (
        expense.assign(amount_abs=expense["amount"].abs())
        .groupby("merchant")["amount_abs"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        if not expense.empty
        else pd.Series()
    )
    merchant_labels = merchant_spend.index.fillna("N/A").tolist()
    merchant_values = merchant_spend.fillna(0).astype(float).tolist()

    movement_group = (
        frame.groupby("movement_type")["amount"].sum().sort_values(ascending=False)
        if not frame.empty
        else pd.Series()
    )
    movement_labels = movement_group.index.fillna("N/A").astype(str).tolist()
    movement_values = movement_group.fillna(0).astype(float).tolist()

    bank_group = (
        frame.groupby("bank")["amount"].sum().sort_values(ascending=False)
        if not frame.empty
        else pd.Series()
    )
    bank_labels = bank_group.index.fillna("N/A").astype(str).tolist()
    bank_values = bank_group.fillna(0).astype(float).tolist()

    if not frame.empty:
        indexed = frame.set_index("date_dt").sort_index()
        net_series = indexed["amount"].resample(freq).sum()
        income_series = indexed[indexed["income_flag"]]["amount"].resample(freq).sum()
        expense_series = indexed[indexed["expense_flag"]]["amount"].resample(freq).sum().abs()
        income_series = income_series.reindex(net_series.index, fill_value=0)
        expense_series = expense_series.reindex(net_series.index, fill_value=0)
        ts_labels = _format_period_labels(net_series.index, granularity)
    else:
        net_series = pd.Series()
        income_series = pd.Series()
        expense_series = pd.Series()
        ts_labels = []

    daily_net = (
        frame.groupby("date")["amount"].sum().sort_index() if not frame.empty else pd.Series()
    )
    if not daily_net.empty:
        daily_index = pd.to_datetime(daily_net.index, errors="coerce")
        daily_net.index = daily_index
        daily_net = daily_net.dropna()
        monthly_avg = daily_net.groupby(daily_net.index.to_period("M")).mean()
    else:
        monthly_avg = pd.Series()
    monthly_avg_labels = monthly_avg.index.astype(str).tolist()
    monthly_avg_values = monthly_avg.fillna(0).astype(float).tolist()
    avg_monthly_net = float(monthly_avg.mean()) if not monthly_avg.empty else 0.0

    chart_data = {
        "time_series": {
            "labels": ts_labels,
            "net": net_series.fillna(0).astype(float).tolist() if not net_series.empty else [],
            "income": income_series.fillna(0).astype(float).tolist() if not income_series.empty else [],
            "expense": expense_series.fillna(0).astype(float).tolist()
            if not expense_series.empty
            else [],
        },
        "movement": {"labels": movement_labels, "values": movement_values},
        "merchants": {"labels": merchant_labels, "values": merchant_values},
        "monthly_avg": {"labels": monthly_avg_labels, "values": monthly_avg_values},
        "banks": {"labels": bank_labels, "values": bank_values},
    }

    context = {
        "total_income": total_income,
        "total_expense": total_expense,
        "net": net,
        "tx_count": tx_count,
        "avg_monthly_net": avg_monthly_net,
        "chart_data_json": json.dumps(chart_data),
        "filters": request.GET,
        "granularity": granularity,
        "banks": sorted(frame_all["bank"].dropna().unique().tolist()) if not frame_all.empty else [],
        "accounts": sorted(
            frame_all["account_last4"]
            .dropna()
            .loc[frame_all["account_last4"] != "6176"]
            .unique()
            .tolist()
        )
        if not frame_all.empty
        else [],
        "movement_types": sorted(frame_all["movement_type"].dropna().unique().tolist())
        if not frame_all.empty
        else [],
    }
    return render(request, "dashboard/dashboard.html", context)
